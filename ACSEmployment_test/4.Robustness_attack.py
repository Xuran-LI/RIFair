import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ACSEmployment_test.model import MLP, WideDeep, DeepFM, AutoInt, TabTransformer


def infer_input_type(model):
    """
    根据 forward 签名判断模型输入格式：
    - ('x_value', ) → MLP
    - ('x_index','x_value') → WideDeep / DeepFM / AutoInt
    - ('x',) → TabTransformer
    """
    import inspect
    sig = inspect.signature(model.forward)
    args = list(sig.parameters.keys())

    if len(args) == 2:
        return "cat_num"      # 含 index_input + value_input
    elif len(args) == 1:
        return "numerical"    # 单输入 (MLP / TabTransformer)
    else:
        raise ValueError("不支持的模型输入格式: forward 参数 = ", args)


def compute_input_gradients(model, sample, device="cpu"):
    # ===============================================================
    # 计算模型输入梯度的函数
    # ===============================================================
    """
    sample: tuple 或 numpy
       - 对 embedding 模型: (index_input, value_input, label)
       - 对 MLP / TabTransformer: (value_input, label)
    """
    model.eval()
    model.to(device)

    input_type = infer_input_type(model)

    # ---------------------- 准备数据 ----------------------
    if input_type == "cat_num":      # index + value
        index_input, value_input, label = sample
        index_input = torch.tensor(index_input).long().unsqueeze(0).to(device)
        value_input = torch.tensor(value_input).float().unsqueeze(0).to(device)
        label = torch.tensor(label).float().unsqueeze(0).to(device)
        value_input.requires_grad = True

    else:  # numerical only
        value_input, label = sample
        value_input = torch.tensor(value_input).float().unsqueeze(0).to(device)
        label = torch.tensor(label).float().unsqueeze(0).to(device)
        value_input.requires_grad = True

    # ---------------------- Forward ----------------------
    if input_type == "cat_num":
        logits = model(index_input, value_input)
    else:
        logits = model(value_input)

    # ---------------------- Loss ----------------------
    if logits.shape[1] == 1:
        target = label[:, 1].unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, target)
    else:
        target = label.argmax(dim=1)
        loss = F.cross_entropy(logits, target)

    # ---------------------- Backward ----------------------
    model.zero_grad()
    loss.backward()

    # ---------------------- 返回梯度 ----------------------
    grad_index = None  # index 无法求梯度
    grad_value = value_input.grad.detach().cpu().numpy()

    return grad_index, grad_value


def compute_embedding_gradients(model, sample, device='cpu'):
    # ===============================================================
    # 计算 embedding 输出梯度的函数
    # ===============================================================
    """
    获取模型 embedding 输出对 loss 的梯度
    """
    model.eval()
    input_type = infer_input_type(model)

    grads_dict = {}
    hooks = []

    # ---------------- Hook function ----------------
    def save_grad(name):
        def hook(module, grad_in, grad_out):
            grads_dict[name] = grad_out[0].detach()
        return hook

    # ---------------- 注册 hook ----------------
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            hooks.append(module.register_full_backward_hook(save_grad(name)))

    # ---------------- 准备输入 ----------------
    if input_type == "cat_num":
        index_input, value_input, label = sample
        index_input = torch.tensor(index_input).long().unsqueeze(0).to(device)
        value_input = torch.tensor(value_input).float().unsqueeze(0).to(device)
        label = torch.tensor(label).float().unsqueeze(0).to(device)
        logits = model(index_input, value_input)
    else:
        value_input, label = sample
        value_input = torch.tensor(value_input).float().unsqueeze(0).to(device)
        label = torch.tensor(label).float().unsqueeze(0).to(device)
        logits = model(value_input)

    # ---------------- Loss ----------------
    if logits.shape[1] == 1:
        target = label[:, 1].unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, target)
    else:
        target = label.argmax(dim=1)
        loss = F.cross_entropy(logits, target)

    # ---------------- Backward ----------------
    model.zero_grad()
    loss.backward()

    # ---------------- 移除 hooks ----------------
    for h in hooks:
        h.remove()

    return grads_dict


def load_pytorch_model(model_class, model_path, init_args: dict, device="cpu"):
    # ===============================================================
    # 加载PyTorch模型
    # ===============================================================
    """
    model_class: 类 (MLP, WideDeep, DeepFM, AutoInt...)
    model_path: .pt / .pth
    init_args: 初始化参数字典，例如：
        {"input_dim":100}  或 {"num_fields":20,"num_categories":num_categories}
    """
    model = model_class(**init_args)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def calculate_replaced_value_each_feature(model, vocab_size=100, device='cpu'):
    # ===============================================================
    # 计算每个离散属性的可替换token扰动幅度
    # ===============================================================
    """
    计算每个离散属性的可替换 token 扰动方向
    对于每个 token embedding，返回它与 vocab 中所有其他 embedding 的差向量
    :param model: 已训练的模型，包含 embedding 层 model.emb_layers
    :param vocab_size: embedding 的总 vocab 数量
    :return: directions: list of shape [vocab_size, vocab_size, emb_dim]
             directions[i, j] = emb[j] - emb[i]
    """
    model.eval()
    with torch.no_grad():
        # 获取 embedding 权重
        if isinstance(model, WideDeep):
            emb_weights = model.emb_layers.weight.data.clone().to(device)
        elif isinstance(model, DeepFM):
            emb_weights = model.second_order_emb.weight.data.clone().to(device)
        elif isinstance(model, AutoInt):
            emb_weights = model.embedding.index_emb.weight.data.clone().to(device)
        else:
            return None

        directions = []
        for i in range(vocab_size):
            # 当前 token embedding
            emb_i = emb_weights[i]  # [emb_dim]
            # 所有其他 token embedding - 当前 embedding
            diff = emb_weights - emb_i.unsqueeze(0)  # [vocab_size, emb_dim]
            directions.append(diff)  # 每个 i 对应一个 [vocab_size, emb_dim]

        directions = torch.stack(directions, dim=0)  # [vocab_size, vocab_size, emb_dim]

    return directions


def select_max_gradient_feature(grad_emb, grad_val, discrete_mask, continuous_mask, none_protect_mask):
    """
    选择梯度最大的离散、连续属性进行扰动
    """
    if grad_emb is not None:
        grad_norm_index = numpy.linalg.norm(grad_emb, axis=1)  # [F]
    else:
        grad_norm_index = None
    grad_norm_value =  numpy.linalg.norm(grad_val, axis=0)  # [F]

    # 离散属性可扰动
    discrete_idx = numpy.where(discrete_mask & none_protect_mask)[0]
    continuous_idx = numpy.where(continuous_mask & none_protect_mask)[0]

    disc_id = None
    cont_id = None

    def pick_top3_random(indices, grad_norm):
        """
        从 grad_norm 中选出 indices 指定的元素中梯度最大的三个之一
        """
        if len(indices) == 0:
            return None

        vals = grad_norm[indices]  # 对应梯度
        topk = min(3, len(vals))  # 取 min(3, n)
        top_idx = numpy.argpartition(-vals, topk - 1)[:topk]  # top-k 的乱序索引
        top_real_indices = [indices[i] for i in top_idx]  # 转换回原始坐标

        # 随机返回其中一个
        return random.choice(top_real_indices)

    if len(discrete_idx) > 0 and grad_emb is not None:
        # disc_id = discrete_idx[numpy.argmax(grad_norm_index[discrete_idx])]
        disc_id = pick_top3_random(discrete_idx, grad_norm_index)

    if len(continuous_idx) > 0:
        # cont_id = continuous_idx[numpy.argmax(grad_norm_value[continuous_idx])]
        cont_id = pick_top3_random(continuous_idx, grad_norm_value)

    return disc_id, cont_id


def optimize_discrete_attribute(dl_dx, dir_unit, x, attr_id):
    """
    选择扰动离散属性的替换 token
    dl_dx: [F, emb_dim] 梯度
    dir_unit: 预计算的方向向量列表 delta_vectors[F][num_tokens][emb_dim]
    x: 当前输入 x_v
    attr_id: 要扰动的属性索引
    返回 adv_token_id
    """
    dot_products = [numpy.dot(dl_dx[attr_id], dir_unit[attr_id][j]) for j in range(len(dir_unit[attr_id]))]
    adv_token_id = int(numpy.argmax(dot_products))
    return adv_token_id


def false(model, x_i1, x_v1, y, x_i2, x_v2, times, dir_unit):
    """
    生成 false-bias 对抗样本（连续 + 离散）
    """

    # ------------------ 初始化 ------------------
    has_discrete = x_i1 is not None and x_i2 is not None

    adv_xi1 = x_i1.copy() if has_discrete else None
    adv_xi2 = x_i2.copy() if has_discrete else None
    adv_xv1, adv_xv2 = x_v1.copy(), x_v2.copy()

    # 掩码
    if has_discrete:
        continuous_mask = (x_v1 != 1)
        discrete_mask   = ~continuous_mask
    else:
        continuous_mask = numpy.ones_like(x_v1, dtype=bool)
        discrete_mask   = numpy.zeros_like(x_v1, dtype=bool)  # 无离散特征

    history = [{
        "iter": 0,
        "adv_xi1": None if not has_discrete else adv_xi1.copy(),
        "adv_xv1": adv_xv1.copy(),
        "adv_xi2": None if not has_discrete else adv_xi2.copy(),
        "adv_xv2": adv_xv2.copy(),
        "disc_id": -1,
        "cont_id": -1
    }]

    # =========================================================
    #                    主循环
    # =========================================================
    for t in range(times):

        # ------------ 计算 embedding 梯度（离散部分） ------------
        if has_discrete:
            grad_emb1 = compute_embedding_gradients(model, (adv_xi1, adv_xv1, y))
            grad_emb2 = compute_embedding_gradients(model, (adv_xi2, adv_xv2, y))

            dl_dx_emb = {}
            if isinstance(model, WideDeep):
                dl_dx_emb = {k: grad_emb1[k] for k in grad_emb1}
            elif isinstance(model, DeepFM):
                dl_dx_emb = {"second_order_emb": grad_emb1["second_order_emb"]}
            elif isinstance(model, AutoInt):
                dl_dx_emb = {k: grad_emb1[k] for k in grad_emb1}
            else:
                dl_dx_emb = None

        # ------------ 连续梯度 -----------------------------------
        if has_discrete:
            _, grad1 = compute_input_gradients(model, (adv_xi1, adv_xv1, y))
            _, grad2 = compute_input_gradients(model, (adv_xi2, adv_xv2, y))
            none_protect = adv_xi1 == adv_xi2
        else:
            _, grad1 = compute_input_gradients(model, (adv_xv1, y))
            _, grad2 = compute_input_gradients(model, (adv_xv2, y))
            none_protect = adv_xv1 == adv_xv2

        dl_dx_val = grad1

        # ------------ 选特征：离散 + 连续 --------------------------
        if has_discrete:
            disc_id, cont_id = None, None
            for k in dl_dx_emb:
                disc_id, cont_id = select_max_gradient_feature(dl_dx_emb[k][0], dl_dx_val,discrete_mask, continuous_mask,none_protect)
        else:
            disc_id, cont_id = select_max_gradient_feature(None, dl_dx_val,discrete_mask, continuous_mask,none_protect)

        # =========================================================
        #                执行扰动
        # =========================================================

        # ------------ 离散扰动 -----------------------------------
        if has_discrete and disc_id is not None:
            for k in dl_dx_emb:
                adv_token = optimize_discrete_attribute(dl_dx_emb[k][0], dir_unit, adv_xi1, disc_id)
                adv_xi1[disc_id] = adv_token
                adv_xi2[disc_id] = adv_token

        # ------------ 连续扰动 -----------------------------------
        if cont_id is not None:
            grad_sign = dl_dx_val[0][cont_id] if has_discrete else numpy.sign(dl_dx_val[0][cont_id])
            step = 0.1 if has_discrete else 1.0
            adv_xv1[cont_id] += step * grad_sign
            adv_xv2[cont_id] += step * grad_sign

        # ------------ 保存历史 -----------------------------------
        history.append({
            "iter": t+1,
            "adv_xi1": None if not has_discrete else adv_xi1.copy(),
            "adv_xv1": adv_xv1.copy(),
            "adv_xi2": None if not has_discrete else adv_xi2.copy(),
            "adv_xv2": adv_xv2.copy(),
            "disc_id": disc_id,
            "cont_id": cont_id
        })

    # =========================================================
    #                 返回最终结果
    # =========================================================
    if has_discrete:
        return [adv_xi1, adv_xv1], [adv_xi2, adv_xv2], [y], history
    else:
        return [adv_xv1], [adv_xv2], [y], history


def false_attack(model, model_name, encode_tag, data_tag):
    """
    进行准确公平性测试，并保存攻击结果
    """

    # ----------------------- 1. 加载数据 -----------------------
    index1 = numpy.load(f"../dataset/ACS/employment/data/N_{encode_tag}_{data_tag}_i.npy", allow_pickle=True)
    value1 = numpy.load(f"../dataset/ACS/employment/data/N_{encode_tag}_{data_tag}_V.npy", allow_pickle=True)

    index2 = numpy.load(f"../dataset/ACS/employment/data/N_{encode_tag}_aug_{data_tag}_i.npy", allow_pickle=True)
    value2 = numpy.load(f"../dataset/ACS/employment/data/N_{encode_tag}_aug_{data_tag}_V.npy", allow_pickle=True)

    label = numpy.load(f"../dataset/ACS/employment/data/N_{encode_tag}_{data_tag}_y.npy", allow_pickle=True)
    label = numpy.eye(2)[label]

    # ----------------------- 2. 计算替换 token 的方向 -----------------------
    dir_unit = calculate_replaced_value_each_feature(model)

    # ----------------------- 3. 保存攻击结果的列表 -----------------------
    results = []

    # ----------------------- 4. 运行攻击 -----------------------
    for i in range(int(select_rate*index1.shape[0])):
        # 随机选择一个 augmented 对象
        s_i = random.randint(0, index2.shape[0] - 1)

        clean_xi = index1[i]
        clean_xv = value1[i]
        clean_y = label[i]

        aug_xi = index2[s_i][i]
        aug_xv = value2[s_i][i]
        # 运行 false-bias 攻击
        if encode_tag=="E":
            clean1=[clean_xi, clean_xv]
            clean2=[aug_xi, aug_xv]
            adv1, adv2, y, history = false(model=model, x_i1=clean_xi, x_v1=clean_xv, y=clean_y,
                                           x_i2=aug_xi, x_v2=aug_xv, times=10, dir_unit=dir_unit)
        else:
            clean1=[clean_xi*clean_xv]
            clean2=[aug_xi*aug_xv]
            adv1, adv2, y, history = false(model=model, x_i1=None, x_v1=clean_xi * clean_xv, y=clean_y,
                                           x_i2=None, x_v2=aug_xi*aug_xv, times=10, dir_unit=dir_unit)

        # 保存结果字典
        results.append({"sample_id": i, "selected_aug_id": s_i, "clean1": clean1, "clean2": clean2,"y":y,
                        "adv1": adv1, "adv2": adv2, "history": history})

    # ----------------------- 5. 保存到文件 -----------------------
    save_path=f"../dataset/ACS/employment/adv/False_{data_tag}_{model_name}.npy"
    numpy.save(save_path, results, allow_pickle=True)
    print(f"[✓] false-bias attack finished. Results saved to {save_path}")
    return results


if __name__ == '__main__':
    num_fields=16

    data_tags = ["vali"]
    select_rate = 0.2
    data_tags = ["test"]
    select_rate = 1
    for data_tag in data_tags:
        # MLP
        mlp_path="../dataset/ACS/employment/model/MLP.pt"
        mlp_model = load_pytorch_model(MLP, mlp_path, init_args={"input_dim":num_fields})
        false_attack(mlp_model, "MLP", encode_tag="O", data_tag=data_tag)
        print("MLP")


        # WideDeep
        wd_path = "../dataset/ACS/employment/model/wideDeep.pt"
        wd_model = load_pytorch_model(WideDeep, wd_path,init_args={"num_fields":num_fields})
        false_attack(wd_model, "WideDeep", encode_tag="E", data_tag=data_tag)
        print("WideDeep")


        # DeepFM
        dfm_path = "../dataset/ACS/employment/model/DeepFM.pt"
        dfm_model = load_pytorch_model(DeepFM, dfm_path,init_args={"num_fields": num_fields})
        false_attack(dfm_model, "DeepFM", encode_tag="E", data_tag=data_tag)
        print("DeepFM")


        # AutoInt
        ai_path = "../dataset/ACS/employment/model/AutoInt.pt"
        ai_model = load_pytorch_model(AutoInt, ai_path,init_args={})
        false_attack(ai_model, "AutoInt", encode_tag="E", data_tag=data_tag)
        print("AutoInt")


        # TabTransformer
        tt_path = "../dataset/ACS/employment/model/TabTransformer.pt"
        tt_model = load_pytorch_model(TabTransformer, tt_path, init_args={"num_fields": num_fields})
        false_attack(tt_model, "TabTransformer", encode_tag="O", data_tag=data_tag)
        print("TabTransformer")
