import keras
import numpy
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils.utils_evaluate import check_items_AF, check_item_confusion
from utils.utils_generate import data_augmentation_adult_item, compute_loss_grad_MP, approximate_by_total_derivative, \
    compute_loss_grad, sort_perturbation_direction, data_augmentation_compas_item, data_augmentation_credit_item, \
    data_augmentation_bank_item


def generate_similar_items(item, dataset, protected_attr):
    """
    生成样本的相似样本
    :return:
    """
    # 生成相似样本
    if dataset == "adult":
        similar_items = data_augmentation_adult_item(item, protected_attr)
    elif dataset == "compas":
        similar_items = data_augmentation_compas_item(item, protected_attr)
    elif dataset == "credit":
        similar_items = data_augmentation_credit_item(item, protected_attr)
    else:
        similar_items = data_augmentation_bank_item(item, protected_attr)
    return similar_items


def add_perturbation_to_similar_items(similar_items, perturbation):
    """

    :return:
    """
    perturbed_similar_items = []
    for i in range(similar_items.shape[0]):
        perturbed_similar_items.append(similar_items[i] + perturbation)
    return numpy.squeeze(numpy.array(perturbed_similar_items))


def TB_global(model, test_item, similar_items, search_times, extent, protected_attr):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    generate_similar = []
    # 保存预测结果错误或歧视的位置
    x, y = numpy.split(test_item, [-1, ], axis=1)
    x_i = x.copy().reshape(1, -1)
    y_i = y.copy()
    FB_Tag = False
    Accurate_fairness_confusion = numpy.array([False, False, False])
    for _ in range(search_times):
        # 计算损失函数导数符号
        y_cate = to_categorical(y_i, num_classes=2)
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)
        # 计算相似样本损失函数，及最远相似样本
        grad2, max_similar = compute_loss_grad_MP(similar_items, y_cate, model)
        sign2 = numpy.sign(grad2)
        # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
        direction = numpy.zeros_like(x_i)
        for n in range(x.shape[1]):
            if n not in protected_attr and sign1[n] != sign2[n]:
                # if sign1[n] != sign2[n]:
                if sign1[n] != 0:
                    direction[0, n] = -1 * sign1[n]
                else:
                    direction[0, n] = sign2[n]
        #  扰动所选择属性
        perturbation = extent * direction
        generated_x = x_i + perturbation
        # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
        pert_l, l_p = approximate_by_total_derivative(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
        # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
        x_i = generated_x
        y_i = pert_l
        generate_x.append(x_i)
        generate_y.append(y_i)

        similar_x_i = add_perturbation_to_similar_items(similar_items, perturbation)
        generate_similar.append(similar_x_i)
        AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
                                        numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i, dist=0, K=0)
        if not AF:
            FB_Tag = True
        Accurate_fairness_confusion = numpy.logical_or(Accurate_fairness_confusion, AF_C)

    adversarial_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
    return adversarial_data, generate_similar, FB_Tag, Accurate_fairness_confusion


def TB_local(model, test_items, similar_items, search_times, extent, FB_Tag, AFC, protected_attr):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    generate_similar = []
    x, y = numpy.split(test_items, [-1, ], axis=1)
    for search_id in range((x.shape[0])):
        x_i = x[search_id].copy().reshape(1, -1)
        y_i = y[search_id].copy()
        similar_x_i = similar_items[search_id]
        generate_x.append(x_i)
        generate_y.append(y_i)
        generate_similar.append(similar_x_i)
        # 计算损失函数导数符号
        y_cate = to_categorical(y_i, num_classes=2)  # 计算分类任务损失函数
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)
        # 计算相似样本损失函数，及最远相似样本
        grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
        sign2 = numpy.sign(grad2)
        # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
        dir = numpy.zeros_like(x_i)
        for n in range(x.shape[1]):
            if n not in protected_attr and sign1[n] != sign2[n]:
                # if sign1[n] != sign2[n]:
                if sign1[n] != 0:
                    dir[0, n] = -1 * sign1[n]
                else:
                    dir[0, n] = sign2[n]
        # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
        sort_privilege = sort_perturbation_direction(dir.copy())
        if len(sort_privilege) == 0:
            continue
        elif len(sort_privilege) > search_times:
            sort_privilege = sort_privilege[:search_times]
        for j in sort_privilege:
            perturbation_direction = numpy.zeros_like(x_i)
            perturbation_direction[0, j] = dir[0, j]
            # 结合扰动变量与扰动方向进行扰动，计算生成的item，label
            perturbation = extent * perturbation_direction
            pert_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_total_derivative(model, x_i, max_similar, y_cate, grad1, grad2, extent, dir)
            generate_x.append(pert_x)
            generate_y.append(pert_l)
            # 检查扰动后样本的准确公平性
            similar_x_i = add_perturbation_to_similar_items(similar_x_i, perturbation)
            generate_similar.append(similar_x_i)
            AF, AF_C = check_item_confusion(numpy.array(pert_l), numpy.argmax(model.predict(pert_x), axis=1),
                                            numpy.argmax(model.predict(similar_x_i), axis=1), pert_x, similar_x_i,
                                            dist=0, K=0)
            if not AF:
                FB_Tag = True
            AFC = numpy.logical_or(AFC, AF_C)

    adversarial_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
    return adversarial_data, generate_similar, FB_Tag, AFC


def TB_evaluation(model, test_item, dataset, protected_attr, global_search_times, local_search_times, extent):
    """
    对样本进行TB搜索，返回搜索结果，相似样本，以及该样本的公平混淆矩阵分类结果
    :return:
    """
    similar_items = generate_similar_items(test_item[0, :-1].reshape(1, -1), dataset, protected_attr)
    G_data, G_similar, search_Tag, search_afc = TB_global(model, test_item, similar_items, global_search_times, extent,
                                                          protected_attr)
    return TB_local(model, G_data, G_similar, local_search_times, extent, search_Tag, search_afc, protected_attr)


def FF_global(model, test_item, similar_items, search_times, extent, protected_attr):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    generate_similar = []
    # 保存预测结果错误或歧视的位置
    x, y = numpy.split(test_item, [-1, ], axis=1)
    x_i = x.copy().reshape(1, -1)
    y_i = y.copy()
    FB_Tag = False
    Accurate_fairness_confusion = numpy.array([False, False, False])
    for _ in range(search_times):
        # 计算损失函数导数符号
        y_cate = to_categorical(y_i, num_classes=2)
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)
        # 计算相似样本损失函数，及最远相似样本
        grad2, max_similar = compute_loss_grad_MP(similar_items, y_cate, model)
        sign2 = numpy.sign(grad2)
        # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
        direction = numpy.zeros_like(x_i)
        for n in range(x.shape[1]):
            if n not in protected_attr and sign1[n] == sign2[n]:
                # if sign1[n] == sign2[n]:
                direction[0, n] = sign1[n]
        #  扰动所选择属性
        perturbation = extent * direction
        generated_x = x_i + perturbation
        # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
        pert_l, l_p = approximate_by_total_derivative(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
        # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
        x_i = generated_x
        y_i = pert_l
        generate_x.append(x_i)
        generate_y.append(y_i)

        similar_x_i = add_perturbation_to_similar_items(similar_items, perturbation)
        generate_similar.append(similar_x_i)
        AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
                                        numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i, dist=0, K=0)
        if not AF:
            FB_Tag = True
        Accurate_fairness_confusion = numpy.logical_or(Accurate_fairness_confusion, AF_C)

    adversarial_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
    return adversarial_data, generate_similar, FB_Tag, Accurate_fairness_confusion


def FF_local(model, test_items, similar_items, search_times, extent, FB_Tag, AFC, protected_attr):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    generate_similar = []
    x, y = numpy.split(test_items, [-1, ], axis=1)
    for search_id in range((x.shape[0])):
        x_i = x[search_id].copy().reshape(1, -1)
        y_i = y[search_id].copy()
        similar_x_i = similar_items[search_id]
        generate_x.append(x_i)
        generate_y.append(y_i)
        generate_similar.append(similar_x_i)
        # 计算损失函数导数符号
        y_cate = to_categorical(y_i, num_classes=2)  # 计算分类任务损失函数
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)
        # 计算相似样本损失函数，及最远相似样本
        grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
        sign2 = numpy.sign(grad2)
        # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
        dir = numpy.zeros_like(x_i)
        for n in range(x.shape[1]):
            if n not in protected_attr and sign1[n] == sign2[n]:
                # if sign1[n] == sign2[n]:
                dir[0, n] = sign1[n]
        # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
        sort_privilege = sort_perturbation_direction(dir.copy())
        if len(sort_privilege) == 0:
            continue
        elif len(sort_privilege) > search_times:
            sort_privilege = sort_privilege[:search_times]
        for j in sort_privilege:
            perturbation_direction = numpy.zeros_like(x_i)
            perturbation_direction[0, j] = dir[0, j]
            # 结合扰动变量与扰动方向进行扰动，计算生成的item，label
            perturbation = extent * perturbation_direction
            pert_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_total_derivative(model, x_i, max_similar, y_cate, grad1, grad2, extent, dir)
            generate_x.append(pert_x)
            generate_y.append(pert_l)
            # 检查扰动后样本的准确公平性
            similar_x_i = add_perturbation_to_similar_items(similar_x_i, perturbation)
            generate_similar.append(similar_x_i)
            AF, AF_C = check_item_confusion(numpy.array(pert_l), numpy.argmax(model.predict(pert_x), axis=1),
                                            numpy.argmax(model.predict(similar_x_i), axis=1), pert_x, similar_x_i,
                                            dist=0, K=0)
            if not AF:
                FB_Tag = True
            AFC = numpy.logical_or(AFC, AF_C)

    adversarial_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
    return adversarial_data, generate_similar, FB_Tag, AFC


def FF_evaluation(model, test_item, dataset, protected_attr, global_search_times, local_search_times, extent):
    """
    对样本进行FF搜索，返回搜索结果，相似样本，以及该样本的公平混淆矩阵分类结果
    :return:
    """
    similar_items = generate_similar_items(test_item[0, :-1].reshape(1, -1), dataset, protected_attr)
    G_data, G_similar, search_Tag, search_afc = FF_global(model, test_item, similar_items, global_search_times, extent,
                                                          protected_attr)
    return FF_local(model, G_data, G_similar, local_search_times, extent, search_Tag, search_afc, protected_attr)


def FB_global(model, test_item, similar_items, search_times, extent, protected_attr):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    generate_similar = []
    # 保存预测结果错误或歧视的位置
    x, y = numpy.split(test_item, [-1, ], axis=1)
    x_i = x.copy().reshape(1, -1)
    y_i = y.copy()
    FB_Tag = False
    Accurate_fairness_confusion = numpy.array([False, False, False])
    for _ in range(search_times):
        # 计算损失函数导数符号
        y_cate = to_categorical(y_i, num_classes=2)
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)
        # 计算相似样本损失函数，及最远相似样本
        grad2, max_similar = compute_loss_grad_MP(similar_items, y_cate, model)
        sign2 = numpy.sign(grad2)
        # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
        direction = numpy.zeros_like(x_i)
        for n in range(x.shape[1]):
            if n not in protected_attr and sign1[n] != sign2[n]:
                # if sign1[n] != sign2[n]:
                if sign1[n] != 0:
                    direction[0, n] = sign1[n]
                else:
                    direction[0, n] = -1 * sign2[n]
        #  扰动所选择属性
        perturbation = extent * direction
        generated_x = x_i + perturbation
        # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
        pert_l, l_p = approximate_by_total_derivative(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
        # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
        x_i = generated_x
        y_i = pert_l
        generate_x.append(x_i)
        generate_y.append(y_i)

        similar_x_i = add_perturbation_to_similar_items(similar_items, perturbation)
        generate_similar.append(similar_x_i)
        AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
                                        numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i, dist=0, K=0)
        if not AF:
            FB_Tag = True
        Accurate_fairness_confusion = numpy.logical_or(Accurate_fairness_confusion, AF_C)

    adversarial_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
    return adversarial_data, generate_similar, FB_Tag, Accurate_fairness_confusion


def FB_local(model, test_items, similar_items, search_times, extent, FB_Tag, AFC, protected_attr):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    generate_similar = []
    x, y = numpy.split(test_items, [-1, ], axis=1)
    for search_id in range((x.shape[0])):
        x_i = x[search_id].copy().reshape(1, -1)
        y_i = y[search_id].copy()
        similar_x_i = similar_items[search_id]
        generate_x.append(x_i)
        generate_y.append(y_i)
        generate_similar.append(similar_x_i)
        # 计算损失函数导数符号
        y_cate = to_categorical(y_i, num_classes=2)  # 计算分类任务损失函数
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)
        # 计算相似样本损失函数，及最远相似样本
        grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
        sign2 = numpy.sign(grad2)
        # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
        dir = numpy.zeros_like(x_i)
        for n in range(x.shape[1]):
            if n not in protected_attr and sign1[n] != sign2[n]:
                # if sign1[n] != sign2[n]:
                if sign1[n] != 0:
                    dir[0, n] = sign1[n]
                else:
                    dir[0, n] = -1 * sign2[n]
        # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
        sort_privilege = sort_perturbation_direction(dir.copy())
        if len(sort_privilege) == 0:
            continue
        elif len(sort_privilege) > search_times:
            sort_privilege = sort_privilege[:search_times]
        for j in sort_privilege:
            perturbation_direction = numpy.zeros_like(x_i)
            perturbation_direction[0, j] = dir[0, j]
            # 结合扰动变量与扰动方向进行扰动，计算生成的item，label
            perturbation = extent * perturbation_direction
            pert_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_total_derivative(model, x_i, max_similar, y_cate, grad1, grad2, extent, dir)
            generate_x.append(pert_x)
            generate_y.append(pert_l)
            # 检查扰动后样本的准确公平性
            similar_x_i = add_perturbation_to_similar_items(similar_x_i, perturbation)
            generate_similar.append(similar_x_i)
            AF, AF_C = check_item_confusion(numpy.array(pert_l), numpy.argmax(model.predict(pert_x), axis=1),
                                            numpy.argmax(model.predict(similar_x_i), axis=1), pert_x, similar_x_i,
                                            dist=0, K=0)
            if not AF:
                FB_Tag = True
            AFC = numpy.logical_or(AFC, AF_C)

    adversarial_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
    return adversarial_data, generate_similar, FB_Tag, AFC


def FB_evaluation(model, test_item, dataset, protected_attr, global_search_times, local_search_times, extent):
    """
    对样本进行TB搜索，返回搜索结果，相似样本，以及该样本的公平混淆矩阵分类结果
    :return:
    """
    similar_items = generate_similar_items(test_item[0, :-1].reshape(1, -1), dataset, protected_attr)
    G_data, G_similar, search_Tag, search_afc = FB_global(model, test_item, similar_items, global_search_times, extent,
                                                          protected_attr)
    return FB_local(model, G_data, G_similar, local_search_times, extent, search_Tag, search_afc, protected_attr)


def run_RobustFair_experiment(model_file, test_file, dataset, protected_attr, G_time, L_time, extent):
    """
    进行准确公平性测试
    :return:
    """
    model = keras.models.load_model(model_file)
    test_data = numpy.load(test_file)

    adv_data = []
    adv_similar_data = []
    adv_AF = []
    adv_AFC = []
    for i in range(test_data.shape[0]):
        test_item = test_data[i].copy().reshape(1, -1)
        TB_R, TB_S, TB_AF, TB_AFC = TB_evaluation(model, test_item, dataset, protected_attr, G_time, L_time, extent)
        FF_R, FF_S, FF_AF, FF_AFC = FF_evaluation(model, test_item, dataset, protected_attr, G_time, L_time, extent)
        FB_R, FB_S, FB_AF, FB_AFC = FF_evaluation(model, test_item, dataset, protected_attr, G_time, L_time, extent)

        result = numpy.concatenate((TB_R, FF_R, FB_R), axis=0)
        similar_result = numpy.concatenate((TB_S, FF_S, FB_S), axis=0)
        unique_adv_data, unique_index = numpy.unique(result, return_index=True, axis=0)

        adv_data.append(unique_adv_data)
        adv_similar_data.append(similar_result[unique_index])
        adv_AF.append(numpy.logical_or(TB_AF, numpy.logical_or(FF_AF, FB_AF)))
        adv_AFC.append(numpy.logical_or(TB_AFC, numpy.logical_or(FF_AFC, FB_AFC)))

    return adv_data, adv_similar_data, adv_AF, adv_AFC


def AF_RobustFair_experiment(M_files, S_file, D_tag, P_attr, G_time, L_time, extent, AF_f_0, AF_f_1, AF_f_2, AF_f_3):
    """
    AF 测试样本生成
    :return:
    """
    for i in range(len(M_files)):
        AF_D, AF_S, FB_P, FB_D = run_RobustFair_experiment(M_files[i], S_file, D_tag, P_attr, G_time, L_time, extent)
        numpy.save(AF_f_0[i], AF_D)
        numpy.save(AF_f_1[i], AF_S)
        numpy.save(AF_f_2[i], FB_P)
        numpy.save(AF_f_3[i], FB_D)


def run_feature_RobustFair_experiment(model_file, test_file, dataset, protected_attr, G_time, L_time, extent):
    """
    进行准确公平性测试
    :return:
    """
    model = keras.models.load_model(model_file)
    test_data = numpy.load(test_file, allow_pickle=True)

    R_adv_data = []
    R_adv_similar_data = []
    R_adv_AF = []
    R_adv_AFC = []
    for i in range(test_data.shape[0]):
        test_seed = test_data[i]
        adv_data = []
        adv_similar_data = []
        adv_AF = []
        adv_AFC = []
        for i in range(test_seed.shape[0]):
            test_item = test_seed[i].copy().reshape(1, -1)
            TB_R, TB_S, TB_AF, TB_AFC = TB_evaluation(model, test_item, dataset, protected_attr, G_time, L_time, extent)
            FF_R, FF_S, FF_AF, FF_AFC = FF_evaluation(model, test_item, dataset, protected_attr, G_time, L_time, extent)
            FB_R, FB_S, FB_AF, FB_AFC = FF_evaluation(model, test_item, dataset, protected_attr, G_time, L_time, extent)

            result = numpy.concatenate((TB_R, FF_R, FB_R), axis=0)
            similar_result = numpy.concatenate((TB_S, FF_S, FB_S), axis=0)
            unique_adv_data, unique_index = numpy.unique(result, return_index=True, axis=0)

            adv_data.append(unique_adv_data)
            adv_similar_data.append(similar_result[unique_index])
            adv_AF.append(numpy.logical_or(TB_AF, numpy.logical_or(FF_AF, FB_AF)))
            adv_AFC.append(numpy.logical_or(TB_AFC, numpy.logical_or(FF_AFC, FB_AFC)))
        R_adv_data.append(adv_data)
        R_adv_similar_data.append(adv_similar_data)
        R_adv_AF.append(adv_AF)
        R_adv_AFC.append(adv_AFC)

    return R_adv_data, R_adv_similar_data, R_adv_AF, R_adv_AFC


def feature_AF_RobustFair_experiment(M_files, S_file, D_tag, P_attr, G_time, L_time, extent,
                                     AF_f_0, AF_f_1, AF_f_2, AF_f_3):
    """
    AF 测试样本生成
    :return:
    """
    for i in range(len(M_files)):
        AF_D, AF_S, FB_P, FB_D = run_feature_RobustFair_experiment(M_files[i], S_file, D_tag, P_attr,
                                                                   G_time, L_time, extent)
        numpy.save(AF_f_0[i], AF_D)
        numpy.save(AF_f_1[i], AF_S)
        numpy.save(AF_f_2[i], FB_P)
        numpy.save(AF_f_3[i], FB_D)

# def TB_global(model, test_data, dataset, protected, search_times, extent):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
#     :return:
#     """
#     generate_x = []
#     generate_y = []
#     loss_perturbation = []
#     # 保存预测结果错误或歧视的位置
#     FB_P = []
#     FB_D = []
#     x, y = numpy.split(test_data, [-1, ], axis=1)
#     for i in range(x.shape[0]):
#         # 初始化
#         x_i = x[i].copy().reshape(1, -1)
#         y_i = y[i].copy()
#         FB_P_cond = False
#         FB_D_cond = numpy.array([False, False, False])
#         for _ in range(search_times):
#             if dataset == "adult":
#                 similar_x_i = data_augmentation_adult_item(x_i, protected)
#             elif dataset == "compas":
#                 similar_x_i = data_augmentation_compas_item(x_i, protected)
#             elif dataset == "credit":
#                 similar_x_i = data_augmentation_credit_item(x_i, protected)
#             else:
#                 similar_x_i = data_augmentation_bank_item(x_i, protected)
#             # 计算损失函数导数符号
#             y_cate = to_categorical(y_i, num_classes=2)
#             grad1 = compute_loss_grad(x_i, y_cate, model)
#             sign1 = numpy.sign(grad1)
#             # 计算相似样本损失函数，及最远相似样本
#             grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
#             sign2 = numpy.sign(grad2)
#             # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#             direction = numpy.zeros_like(x_i)
#             for n in range(x.shape[1]):
#                 # if n not in protected and sign1[n] != sign2[n]:
#                 if sign1[n] != sign2[n]:
#                     if sign1[n] != 0:
#                         direction[0, n] = -1 * sign1[n]
#                     else:
#                         direction[0, n] = sign2[n]
#             # if numpy.all(direction == 0):
#             #     break
#             #  扰动所选择属性
#             perturbation = extent * direction
#             generated_x = x_i + perturbation
#             # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
#             pert_l, l_p = approximate_by_total_derivative(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
#             # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
#             x_i = generated_x
#             y_i = pert_l
#             generate_x.append(x_i)
#             generate_y.append(y_i)
#             loss_perturbation.append(l_p)
#
#             # 检查扰动后样本预测结果是否准确且公平
#             if dataset == "adult":
#                 similar_x_i = data_augmentation_adult_item(x_i, protected)
#             elif dataset == "compas":
#                 similar_x_i = data_augmentation_compas_item(x_i, protected)
#             elif dataset == "credit":
#                 similar_x_i = data_augmentation_credit_item(x_i, protected)
#             else:
#                 similar_x_i = data_augmentation_bank_item(x_i, protected)
#
#             AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
#                                             numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i,
#                                             dist=0, K=0)
#             if not AF:
#                 FB_P_cond = True
#             FB_D_cond = numpy.logical_or(FB_D_cond, AF_C)
#         FB_P.append(FB_P_cond)
#         FB_D.append(FB_D_cond.tolist())
#
#     generate_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
#     generate_loss = numpy.squeeze(numpy.array(loss_perturbation))
#     FB_P = numpy.array(FB_P, dtype=bool)
#     FB_D = numpy.array(FB_D, dtype=bool)
#
#     return generate_data, generate_loss, FB_P, FB_D
#
#
# def FF_global(model, test_data, dataset, protected, search_times, extent):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
#     :return:
#     """
#     generate_x = []
#     generate_y = []
#     loss_perturbation = []
#     # 保存预测结果错误或歧视的位置
#     FB_P = []
#     FB_D = []
#     x, y = numpy.split(test_data, [-1, ], axis=1)
#     for i in range(x.shape[0]):
#         # 初始化
#         x_i = x[i].copy().reshape(1, -1)
#         y_i = y[i].copy()
#         FB_P_cond = False
#         FB_D_cond = numpy.array([False, False, False])
#         for _ in range(search_times):
#             # 生成相似样本
#             if dataset == "adult":
#                 similar_x_i = data_augmentation_adult_item(x_i, protected)
#             elif dataset == "compas":
#                 similar_x_i = data_augmentation_compas_item(x_i, protected)
#             elif dataset == "credit":
#                 similar_x_i = data_augmentation_credit_item(x_i, protected)
#             else:
#                 similar_x_i = data_augmentation_bank_item(x_i, protected)
#             # 计算损失函数导数符号
#             y_cate = to_categorical(y_i, num_classes=2)
#             grad1 = compute_loss_grad(x_i, y_cate, model)
#             sign1 = numpy.sign(grad1)
#             # 计算相似样本损失函数，及最远相似样本
#             grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
#             sign2 = numpy.sign(grad2)
#             # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#             direction = numpy.zeros_like(x_i)
#             for n in range(x.shape[1]):
#                 # if n not in protected and sign1[n] == sign2[n]:
#                 if sign1[n] == sign2[n]:
#                     direction[0, n] = sign1[n]
#             # if numpy.all(direction == 0):
#             #     break
#             #  扰动所选择属性
#             perturbation = extent * direction
#             generated_x = x_i + perturbation
#             # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
#             pert_l, l_p = approximate_by_total_derivative(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
#             # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
#             x_i = generated_x
#             y_i = pert_l
#             generate_x.append(x_i)
#             generate_y.append(y_i)
#             loss_perturbation.append(l_p)
#             # 检查扰动后样本预测结果是否准确且公平
#             if dataset == "adult":
#                 similar_x_i = data_augmentation_adult_item(x_i, protected)
#             elif dataset == "compas":
#                 similar_x_i = data_augmentation_compas_item(x_i, protected)
#             elif dataset == "credit":
#                 similar_x_i = data_augmentation_credit_item(x_i, protected)
#             else:
#                 similar_x_i = data_augmentation_bank_item(x_i, protected)
#
#             AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
#                                             numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i,
#                                             dist=0, K=0)
#             if not AF:
#                 FB_P_cond = True
#             FB_D_cond = numpy.logical_or(FB_D_cond, AF_C)
#         FB_P.append(FB_P_cond)
#         FB_D.append(FB_D_cond.tolist())
#
#     generate_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
#     generate_loss = numpy.squeeze(numpy.array(loss_perturbation))
#     FB_P = numpy.array(FB_P, dtype=bool)
#     FB_D = numpy.array(FB_D, dtype=bool)
#
#     return generate_data, generate_loss, FB_P, FB_D
#
#
# def FB_global(model, test_data, dataset, protected, search_times, extent):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
#     :return:
#     """
#     generate_x = []
#     generate_y = []
#     loss_perturbation = []
#     # 保存预测结果错误或歧视的位置
#     FB_P = []
#     FB_D = []
#     x, y = numpy.split(test_data, [-1, ], axis=1)
#     for i in range(x.shape[0]):
#         # 初始化
#         x_i = x[i].copy().reshape(1, -1)
#         y_i = y[i].copy()
#         FB_P_cond = False
#         FB_D_cond = numpy.array([False, False, False])
#         for _ in range(search_times):
#             # 生成相似样本用于确定准确公平混淆矩阵扰动方向
#             if dataset == "adult":
#                 similar_x_i = data_augmentation_adult_item(x_i, protected)
#             elif dataset == "compas":
#                 similar_x_i = data_augmentation_compas_item(x_i, protected)
#             elif dataset == "credit":
#                 similar_x_i = data_augmentation_credit_item(x_i, protected)
#             else:
#                 similar_x_i = data_augmentation_bank_item(x_i, protected)
#             # 计算损失函数导数符号
#             y_cate = to_categorical(y_i, num_classes=2)
#             grad1 = compute_loss_grad(x_i, y_cate, model)
#             sign1 = numpy.sign(grad1)
#             # 计算相似样本损失函数，及最远相似样本
#             grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
#             sign2 = numpy.sign(grad2)
#             # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#             direction = numpy.zeros_like(x_i)
#             for n in range(x.shape[1]):
#                 # if n not in protected and sign1[n] != sign2[n]:
#                 if sign1[n] != sign2[n]:
#                     if sign1[n] != 0:
#                         direction[0, n] = sign1[n]
#                     else:
#                         direction[0, n] = -1 * sign2[n]
#             # if numpy.all(direction == 0):
#             #     break
#             #  扰动所选择属性
#             perturbation = extent * direction
#             generated_x = x_i + perturbation
#             # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
#             pert_l, l_p = approximate_by_total_derivative(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
#             # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
#             x_i = generated_x
#             y_i = pert_l
#             generate_x.append(x_i)
#             generate_y.append(y_i)
#             loss_perturbation.append(l_p)
#             # 检查扰动后样本预测结果是否准确且公平
#             if dataset == "adult":
#                 similar_x_i = data_augmentation_adult_item(x_i, protected)
#             elif dataset == "compas":
#                 similar_x_i = data_augmentation_compas_item(x_i, protected)
#             elif dataset == "credit":
#                 similar_x_i = data_augmentation_credit_item(x_i, protected)
#             else:
#                 similar_x_i = data_augmentation_bank_item(x_i, protected)
#
#             AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
#                                             numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i,
#                                             dist=0, K=0)
#             if not AF:
#                 FB_P_cond = True
#             FB_D_cond = numpy.logical_or(FB_D_cond, AF_C)
#         FB_P.append(FB_P_cond)
#         FB_D.append(FB_D_cond.tolist())
#
#     generate_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
#     generate_loss = numpy.squeeze(numpy.array(loss_perturbation))
#     FB_P = numpy.array(FB_P, dtype=bool)
#     FB_D = numpy.array(FB_D, dtype=bool)
#
#     return generate_data, generate_loss, FB_P, FB_D
#
#
# def global_generation(model, test_data, dataset, protected, search_times, extent):
#     """
#     全局搜索：依次远离 break robustness fairness 以及 both
#     :return:
#     """
#     tb_data, tb_loss, FB_P_0, FB_D_0 = TB_global(model, test_data, dataset, protected, search_times, extent)
#     ff_data, ff_loss, FB_P_1, FB_D_1 = FF_global(model, test_data, dataset, protected, search_times, extent)
#     fb_data, fb_loss, FB_P_2, FB_D_2 = FB_global(model, test_data, dataset, protected, search_times, extent)
#
#     search_data = numpy.concatenate((fb_data, tb_data, ff_data), axis=0)
#     search_loss = numpy.concatenate((fb_loss, tb_loss, ff_loss), axis=0)
#
#     unique_data, unique_index = numpy.unique(search_data, return_index=True, axis=0)
#     unique_loss = search_loss[unique_index]
#
#     FB_P = numpy.logical_or(FB_P_2, numpy.logical_or(FB_P_0, FB_P_1))
#     FB_D = numpy.logical_or(FB_D_2, numpy.logical_or(FB_D_0, FB_D_1))
#
#     return unique_data, unique_loss, FB_P, FB_D
#
#
# def TB_local(model, test_data, dataset, protected, search_times, extent, K=0):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
#     :return:
#     """
#     generate_x = []
#     generate_y = []
#     loss_perturbation = []
#     x, y = numpy.split(test_data, [-1, ], axis=1)
#     for s_id in range((x.shape[0])):
#         # 初始化
#         x_i = x[s_id].copy().reshape(1, -1)
#         y_i = y[s_id].copy()
#         # 生成相似样本
#         if dataset == "adult":
#             similar_x_i = data_augmentation_adult_item(x_i, protected)
#         elif dataset == "compas":
#             similar_x_i = data_augmentation_compas_item(x_i, protected)
#         elif dataset == "credit":
#             similar_x_i = data_augmentation_credit_item(x_i, protected)
#         else:
#             similar_x_i = data_augmentation_bank_item(x_i, protected)
#
#         pre_i = model.predict(x_i)
#         S_pre_i = model.predict(similar_x_i)
#         AF = check_item_AF(y_i, numpy.argmax(pre_i, axis=1), numpy.argmax(S_pre_i, axis=1), x_i, similar_x_i, 0, K)
#         if AF:
#             continue
#         # 计算损失函数导数符号
#         y_cate = to_categorical(y_i, num_classes=2)  # 计算分类任务损失函数
#         grad1 = compute_loss_grad(x_i, y_cate, model)
#         sign1 = numpy.sign(grad1)
#         # 计算相似样本损失函数，及最远相似样本
#         grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
#         sign2 = numpy.sign(grad2)
#         # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#         direction = numpy.zeros_like(x_i)
#         for n in range(x.shape[1]):
#             if n not in protected and sign1[n] != sign2[n]:
#                 if sign1[n] != 0:
#                     direction[0, n] = -1 * sign1[n]
#                 else:
#                     direction[0, n] = sign2[n]
#                     # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
#         sort_privilege = sort_perturbation_direction(direction.copy(), protected)
#         if len(sort_privilege) == 0:
#             continue
#         elif len(sort_privilege) > search_times:
#             sort_privilege = sort_privilege[:search_times]
#
#         for j in sort_privilege:
#             perturbation_direction = numpy.zeros_like(x_i)
#             perturbation_direction[0, j] = direction[0, j]
#             # 结合扰动变量与扰动方向进行扰动，计算生成的item，label
#             perturbation = extent * perturbation_direction
#             pert_x = x_i + perturbation
#             # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
#             pert_l, l_p = approximate_by_total_derivative(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
#             generate_x.append(pert_x)
#             generate_y.append(pert_l)
#             loss_perturbation.append(l_p)
#
#     generate_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
#     generate_loss = numpy.squeeze(numpy.array(loss_perturbation))
#     return generate_data, generate_loss
#
#
# def FB_local(model, test_data, dataset, protected, search_times, extent, K=0):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
#     :return:
#     """
#     generate_x = []
#     generate_y = []
#     loss_perturbation = []
#     x, y = numpy.split(test_data, [-1, ], axis=1)
#     for s_id in range((x.shape[0])):
#         # 初始化
#         x_i = x[s_id].copy().reshape(1, -1)
#         y_i = y[s_id].copy()
#         # 生成相似样本
#         if dataset == "adult":
#             similar_x_i = data_augmentation_adult_item(x_i, protected)
#         elif dataset == "compas":
#             similar_x_i = data_augmentation_compas_item(x_i, protected)
#         elif dataset == "credit":
#             similar_x_i = data_augmentation_credit_item(x_i, protected)
#         else:
#             similar_x_i = data_augmentation_bank_item(x_i, protected)
#         pre_i = model.predict(x_i)
#         S_pre_i = model.predict(similar_x_i)
#         AF = check_item_AF(y_i, numpy.argmax(pre_i, axis=1), numpy.argmax(S_pre_i, axis=1), x_i, similar_x_i, 0, K)
#         if AF:
#             continue
#         # 计算损失函数导数符号
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i, y_cate, model)
#         sign1 = numpy.sign(grad1)
#         # 计算相似样本损失函数，及最远相似样本
#         grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
#         sign2 = numpy.sign(grad2)
#         # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#         direction = numpy.zeros_like(x_i)
#         for n in range(x.shape[1]):
#             if n not in protected and sign1[n] != sign2[n]:
#                 if sign1[n] != 0:
#                     direction[0, n] = sign1[n]
#                 else:
#                     direction[0, n] = -1 * sign2[n]
#         # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
#         sort_privilege = sort_perturbation_direction(direction.copy(), protected)
#         if len(sort_privilege) == 0:
#             continue
#         elif len(sort_privilege) > search_times:
#             sort_privilege = sort_privilege[:search_times]
#         for j in sort_privilege:  # 根据梯度方向进行局部搜索
#             perturbation_direction = numpy.zeros_like(x_i)
#             perturbation_direction[0, j] = direction[0, j]
#             # 结合扰动变量与扰动方向进行扰动，计算生成的item，label
#             perturbation = extent * perturbation_direction
#             pert_x = x_i + perturbation
#             # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
#             pert_l, l_p = approximate_by_total_derivative(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
#             generate_x.append(pert_x)
#             generate_y.append(pert_l)
#             loss_perturbation.append(l_p)
#     generate_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
#     generate_loss = numpy.squeeze(numpy.array(loss_perturbation))
#     return generate_data, generate_loss
#
#
# def FF_local(model, test_data, dataset, protected, search_times, extent, K=0):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
#     :return:
#     """
#     generate_x = []
#     generate_y = []
#     loss_perturbation = []
#     x, y = numpy.split(test_data, [-1, ], axis=1)
#     for s_id in range((x.shape[0])):
#         # 初始化
#         x_i = x[s_id].copy().reshape(1, -1)
#         y_i = y[s_id].copy()
#         # 生成相似样本
#         if dataset == "adult":
#             similar_x_i = data_augmentation_adult_item(x_i, protected)
#         elif dataset == "compas":
#             similar_x_i = data_augmentation_compas_item(x_i, protected)
#         elif dataset == "credit":
#             similar_x_i = data_augmentation_credit_item(x_i, protected)
#         else:
#             similar_x_i = data_augmentation_bank_item(x_i, protected)
#         pre_i = model.predict(x_i)
#         S_pre_i = model.predict(similar_x_i)
#         AF = check_item_AF(y_i, numpy.argmax(pre_i, axis=1), numpy.argmax(S_pre_i, axis=1), x_i, similar_x_i, 0, K)
#         if AF:
#             continue
#         # 计算损失函数导数符号
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i, y_cate, model)
#         sign1 = numpy.sign(grad1)
#         # 计算相似样本损失函数，及最远相似样本
#         grad2, max_similar = compute_loss_grad_MP(similar_x_i, y_cate, model)
#         sign2 = numpy.sign(grad2)
#         direction = numpy.zeros_like(x_i)
#         for n in range(x.shape[1]):
#             if n not in protected and sign1[n] == sign2[n]:
#                 direction[0, n] = sign1[n]
#         # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
#         sort_privilege = sort_perturbation_direction(direction.copy(), protected)
#         if len(sort_privilege) == 0:
#             continue
#         elif len(sort_privilege) > search_times:
#             sort_privilege = sort_privilege[:search_times]
#         for j in sort_privilege:
#             perturbation_direction = numpy.zeros_like(x_i)
#             perturbation_direction[0, j] = direction[0, j]
#             #  扰动所选择属性
#             perturbation = extent * perturbation_direction
#             pert_x = x_i + perturbation
#             # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
#             pert_l, l_p = approximate_by_total_derivative(model, x_i, max_similar, y_cate, grad1, grad2, extent, direction)
#             generate_x.append(pert_x)
#             generate_y.append(pert_l)
#             loss_perturbation.append(l_p)
#
#     generate_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
#     generate_loss = numpy.squeeze(numpy.array(loss_perturbation))
#     return generate_data, generate_loss
#
#
# def local_generation(model, local_seeds, dataset, protected, search_times, extent):
#     """
#     局部搜索：依次远离 break robustness fairness 以及 both
#     :return:
#     """
#     tb_data, tb_loss = TB_local(model, local_seeds, dataset, protected, search_times, extent)
#     ff_data, ff_loss = FF_local(model, local_seeds, dataset, protected, search_times, extent)
#     fb_data, fb_loss = FB_local(model, local_seeds, dataset, protected, search_times, extent)
#
#     search_data = numpy.concatenate((fb_data, tb_data, ff_data), axis=0)
#     search_loss = numpy.concatenate((fb_loss, tb_loss, ff_loss), axis=0)
#
#     unique_result, unique_index = numpy.unique(search_data, return_index=True, axis=0)
#     unique_loss = search_loss[unique_index]
#
#     return unique_result, unique_loss, None, None
#
#
# def run_RobustFair_experiment(model_file, test_file, dataset, protected, G_time, L_time, extent):
#     """
#     进行准确公平性测试
#     :return:
#     """
#     model = keras.models.load_model(model_file)
#     test_data = numpy.load(test_file)
#
#     G_item, G_loss, FB_P, FB_D = global_generation(model, test_data, dataset, protected, G_time, extent)
#     L_item, L_loss, _, _ = local_generation(model, G_item, dataset, protected, L_time, extent)
#     RobustFair_data = numpy.concatenate((G_item, L_item), axis=0)
#     RobustFair_data, unique_index = numpy.unique(RobustFair_data, return_index=True, axis=0)
#     return RobustFair_data, FB_P, FB_D
#
#
# def AF_RobustFair_experiment(M_files, S_file, D_tag, P_attr, G_time, L_time, extent, AF_file0, AF_file1, AF_file2):
#     """
#     AF 测试样本生成
#     :return:
#     """
#     for i in range(len(M_files)):
#         AF_D, FB_P, FB_D = run_RobustFair_experiment(M_files[i], S_file, D_tag, P_attr, G_time, L_time, extent)
#         numpy.save(AF_file0[i], AF_D)
#         numpy.save(AF_file1[i], FB_P)
#         numpy.save(AF_file2[i], FB_D)



# import keras
# import numpy
# from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.utils.np_utils import to_categorical
#
# from utils.utils_draw import draw_lines_loss, draw_lines_num
# from utils.utils_evaluate import calculate_MSE, check_dist
# from utils.utils_generate import data_augmentation_adult_item, compute_loss_grad, sort_perturbation_direction, \
#     data_augmentation_compas_item, data_augmentation_credit_item, data_augmentation_bank_item, get_AF_seed, \
#     compute_loss_grad_AVG, approximate_by_calculus_Avg
#
#
# def check_false_bias_global(model, global_seeds, dataset, protected, search_times, extent):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
#     :return:
#     """
#     generate_x = []
#     loss_perturbation = []
#     x, y = numpy.split(global_seeds, [-1, ], axis=1)
#     for i in range(x.shape[0]):
#         # 初始化
#         x_i = x[i].copy().reshape(1, -1)
#         y_i = y[i].copy()
#
#         pert_x = []
#         pert_y = []
#         pert_loss = []
#
#         for _ in range(search_times):
#             # 生成相似样本
#             if dataset == "adult":
#                 similar_x_i = data_augmentation_adult_item(x_i, protected)
#             elif dataset == "compas":
#                 similar_x_i = data_augmentation_compas_item(x_i, protected)
#             elif dataset == "credit":
#                 similar_x_i = data_augmentation_credit_item(x_i, protected)
#             elif dataset == "bank":
#                 similar_x_i = data_augmentation_bank_item(x_i, protected)
#             # 计算损失函数导数符号
#             y_cate = to_categorical(y_i, num_classes=2)
#             grad1 = compute_loss_grad(x_i, y_cate, model)
#             sign1 = numpy.sign(grad1)
#             # 计算相似样本损失函数，及最远相似样本
#             grad2 = compute_loss_grad_AVG(similar_x_i, y_cate, model)
#             sign2 = numpy.sign(grad2)
#             # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#             direction = numpy.zeros_like(x_i)
#             for n in range(x.shape[1]):
#                 if n not in protected and sign1[n] != sign2[n]:
#                     if sign1[n] != 0:
#                         direction[0, n] = sign1[n]
#                     else:
#                         direction[0, n] = -1 * sign2[n]
#             #  扰动所选择属性
#             perturbation = extent * direction
#             generated_x = x_i + perturbation
#             # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
#             pert_l, l_p = approximate_by_calculus_Avg(model, x_i, similar_x_i, y_cate, grad1, grad2, extent, direction)
#
#             # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
#             x_i = generated_x
#             y_i = pert_l
#             pert_x.append(x_i)
#             pert_y.append(y_i)
#             pert_loss.append(l_p)
#
#         generate_x.append(numpy.concatenate((numpy.squeeze(numpy.array(pert_x)), numpy.array(pert_y)), axis=1))
#         loss_perturbation.append(pert_loss)
#
#     return numpy.squeeze(numpy.array(generate_x)), numpy.squeeze(numpy.array(loss_perturbation))
#
#
# def check_true_bias_global(model, global_seeds, dataset, protected, search_times, extent):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
#     :return:
#     """
#     generate_x = []
#     loss_perturbation = []
#     x, y = numpy.split(global_seeds, [-1, ], axis=1)
#     for i in range(x.shape[0]):
#         # 初始化
#         x_i = x[i].copy().reshape(1, -1)
#         y_i = y[i].copy()
#
#         pert_x = []
#         pert_y = []
#         pert_loss = []
#
#         for _ in range(search_times):
#             # 生成相似样本
#             if dataset == "adult":
#                 similar_x_i = data_augmentation_adult_item(x_i, protected)
#             elif dataset == "compas":
#                 similar_x_i = data_augmentation_compas_item(x_i, protected)
#             elif dataset == "credit":
#                 similar_x_i = data_augmentation_credit_item(x_i, protected)
#             elif dataset == "bank":
#                 similar_x_i = data_augmentation_bank_item(x_i, protected)
#             # 计算损失函数导数符号
#             y_cate = to_categorical(y_i, num_classes=2)
#             grad1 = compute_loss_grad(x_i, y_cate, model)
#             sign1 = numpy.sign(grad1)
#             # 计算相似样本损失函数，及最远相似样本
#             grad2 = compute_loss_grad_AVG(similar_x_i, y_cate, model)
#             sign2 = numpy.sign(grad2)
#             # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#             direction = numpy.zeros_like(x_i)
#             for n in range(x.shape[1]):
#                 if n not in protected and sign1[n] != sign2[n]:
#                     if sign1[n] != 0:
#                         direction[0, n] = -1 * sign1[n]
#                     else:
#                         direction[0, n] = sign2[n]
#
#             #  扰动所选择属性
#             perturbation = extent * direction
#             generated_x = x_i + perturbation
#             # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
#             pert_l, l_p = approximate_by_calculus_Avg(model, x_i, similar_x_i, y_cate, grad1, grad2, extent, direction)
#             # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
#             x_i = generated_x
#             y_i = pert_l
#             pert_x.append(x_i)
#             pert_y.append(y_i)
#             pert_loss.append(l_p)
#
#         generate_x.append(numpy.concatenate((numpy.squeeze(numpy.array(pert_x)), numpy.array(pert_y)), axis=1))
#         loss_perturbation.append(pert_loss)
#
#     return numpy.squeeze(numpy.array(generate_x)), numpy.squeeze(numpy.array(loss_perturbation))
#
#
# def check_false_fair_global(model, global_seeds, dataset, protected, search_times, extent):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
#     :return:
#     """
#     generate_x = []
#     loss_perturbation = []
#     x, y = numpy.split(global_seeds, [-1, ], axis=1)
#     for i in range(x.shape[0]):
#         # 初始化
#         x_i = x[i].copy().reshape(1, -1)
#         y_i = y[i].copy()
#
#         pert_x = []
#         pert_y = []
#         pert_loss = []
#
#         for _ in range(search_times):
#             # 生成相似样本
#             if dataset == "adult":
#                 similar_x_i = data_augmentation_adult_item(x_i, protected)
#             elif dataset == "compas":
#                 similar_x_i = data_augmentation_compas_item(x_i, protected)
#             elif dataset == "credit":
#                 similar_x_i = data_augmentation_credit_item(x_i, protected)
#             elif dataset == "bank":
#                 similar_x_i = data_augmentation_bank_item(x_i, protected)
#             # 计算损失函数导数符号
#             y_cate = to_categorical(y_i, num_classes=2)
#             grad1 = compute_loss_grad(x_i, y_cate, model)
#             sign1 = numpy.sign(grad1)
#             # 计算相似样本损失函数，及最远相似样本
#             grad2 = compute_loss_grad_AVG(similar_x_i, y_cate, model)
#             sign2 = numpy.sign(grad2)
#             # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#             direction = numpy.zeros_like(x_i)
#             for n in range(x.shape[1]):
#                 if n not in protected and sign1[n] == sign2[n]:
#                     direction[0, n] = sign1[n]
#
#             #  扰动所选择属性
#             perturbation = extent * direction
#             generated_x = x_i + perturbation
#             # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
#             pert_l, l_p = approximate_by_calculus_Avg(model, x_i, similar_x_i, y_cate, grad1, grad2, extent, direction)
#             # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
#             x_i = generated_x
#             y_i = pert_l
#             pert_x.append(x_i)
#             pert_y.append(y_i)
#             pert_loss.append(l_p)
#
#         generate_x.append(numpy.concatenate((numpy.squeeze(numpy.array(pert_x)), numpy.array(pert_y)), axis=1))
#         loss_perturbation.append(pert_loss)
#
#     return numpy.squeeze(numpy.array(generate_x)), numpy.squeeze(numpy.array(loss_perturbation))
#
#
# def check_false_bias_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
#     :return:
#     """
#     generate_x = []
#     loss_perturbation = []
#     x, y = numpy.split(local_seeds, [-1, ], axis=1)
#     for s_id in range((x.shape[0])):
#         # 初始化
#         x_i = x[s_id].copy().reshape(1, -1)
#         y_i = y[s_id].copy()
#
#         pert_x = []
#         pert_y = []
#         pert_loss = []
#
#         # 生成相似样本
#         if dataset == "adult":
#             similar_x_i = data_augmentation_adult_item(x_i, protected)
#         elif dataset == "compas":
#             similar_x_i = data_augmentation_compas_item(x_i, protected)
#         elif dataset == "credit":
#             similar_x_i = data_augmentation_credit_item(x_i, protected)
#         elif dataset == "bank":
#             similar_x_i = data_augmentation_bank_item(x_i, protected)
#
#         # 计算损失函数导数符号
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i, y_cate, model)
#         sign1 = numpy.sign(grad1)
#         # 计算相似样本损失函数，及最远相似样本
#         grad2 = compute_loss_grad_AVG(similar_x_i, y_cate, model)
#         sign2 = numpy.sign(grad2)
#         # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#         direction = numpy.zeros_like(x_i)
#         for n in range(x.shape[1]):
#             if n not in protected and sign1[n] != sign2[n]:
#                 if sign1[n] != 0:
#                     direction[0, n] = sign1[n]
#                 else:
#                     direction[0, n] = -1 * sign2[n]
#         # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
#         sort_privilege = sort_perturbation_direction(direction.copy(), protected)
#         if len(sort_privilege) == 0:
#             continue
#         elif len(sort_privilege) > search_times:
#             sort_privilege = sort_privilege[:search_times]
#
#         for j in sort_privilege:  # 根据梯度方向进行局部搜索
#             perturbation_direction = numpy.zeros_like(x_i)
#             perturbation_direction[0, j] = direction[0, j]
#             # 结合扰动变量与扰动方向进行扰动，计算生成的item，label
#             perturbation = extent * perturbation_direction
#             generated_x = x_i + perturbation
#             # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
#             pert_l, l_p = approximate_by_calculus_Avg(model, x_i, similar_x_i, y_cate, grad1, grad2, extent, direction)
#             x_i = generated_x
#             y_i = pert_l
#             pert_x.append(x_i)
#             pert_y.append(y_i)
#             pert_loss.append(l_p)
#
#         generate_x.append(numpy.concatenate((numpy.squeeze(numpy.array(pert_x)), numpy.array(pert_y)), axis=1))
#         loss_perturbation.append(pert_loss)
#
#     return numpy.squeeze(numpy.array(generate_x)), numpy.squeeze(numpy.array(loss_perturbation))
#
#
# def check_true_bias_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
#     :return:
#     """
#     generate_x = []
#     loss_perturbation = []
#     x, y = numpy.split(local_seeds, [-1, ], axis=1)
#     for s_id in range((x.shape[0])):
#         # 初始化
#         x_i = x[s_id].copy().reshape(1, -1)
#         y_i = y[s_id].copy()
#
#         pert_x = []
#         pert_y = []
#         pert_loss = []
#
#         # 生成相似样本
#         if dataset == "adult":
#             similar_x_i = data_augmentation_adult_item(x_i, protected)
#         elif dataset == "compas":
#             similar_x_i = data_augmentation_compas_item(x_i, protected)
#         elif dataset == "credit":
#             similar_x_i = data_augmentation_credit_item(x_i, protected)
#         elif dataset == "bank":
#             similar_x_i = data_augmentation_bank_item(x_i, protected)
#
#         # 计算损失函数导数符号
#         y_cate = to_categorical(y_i, num_classes=2)  # 计算分类任务损失函数
#         grad1 = compute_loss_grad(x_i, y_cate, model)
#         sign1 = numpy.sign(grad1)
#         # 计算相似样本损失函数，及最远相似样本
#         grad2 = compute_loss_grad_AVG(similar_x_i, y_cate, model)
#         sign2 = numpy.sign(grad2)
#         # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#         direction = numpy.zeros_like(x_i)
#         for n in range(x.shape[1]):
#             if n not in protected and sign1[n] != sign2[n]:
#                 if sign1[n] != 0:
#                     direction[0, n] = -1 * sign1[n]
#                 else:
#                     direction[0, n] = sign2[n]
#                     # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
#         sort_privilege = sort_perturbation_direction(direction.copy(), protected)
#         if len(sort_privilege) == 0:
#             continue
#         elif len(sort_privilege) > search_times:
#             sort_privilege = sort_privilege[:search_times]
#
#         for j in sort_privilege:
#             perturbation_direction = numpy.zeros_like(x_i)
#             perturbation_direction[0, j] = direction[0, j]
#             # 结合扰动变量与扰动方向进行扰动，计算生成的item，label
#             perturbation = extent * perturbation_direction
#             generated_x = x_i + perturbation
#             # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
#             pert_l, l_p = approximate_by_calculus_Avg(model, x_i, similar_x_i, y_cate, grad1, grad2, extent, direction)
#             x_i = generated_x
#             y_i = pert_l
#             pert_x.append(x_i)
#             pert_y.append(y_i)
#             pert_loss.append(l_p)
#
#         generate_x.append(numpy.concatenate((numpy.squeeze(numpy.array(pert_x)), numpy.array(pert_y)), axis=1))
#         loss_perturbation.append(pert_loss)
#
#     return numpy.squeeze(numpy.array(generate_x)), numpy.squeeze(numpy.array(loss_perturbation))
#
#
# def check_false_fair_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
#     :return:
#     """
#     generate_x = []
#     loss_perturbation = []
#     x, y = numpy.split(local_seeds, [-1, ], axis=1)
#     for s_id in range((x.shape[0])):
#         # 初始化
#         x_i = x[s_id].copy().reshape(1, -1)
#         y_i = y[s_id].copy()
#
#         pert_x = []
#         pert_y = []
#         pert_loss = []
#
#         # 生成相似样本
#         if dataset == "adult":
#             similar_x_i = data_augmentation_adult_item(x_i, protected)
#         elif dataset == "compas":
#             similar_x_i = data_augmentation_compas_item(x_i, protected)
#         elif dataset == "credit":
#             similar_x_i = data_augmentation_credit_item(x_i, protected)
#         elif dataset == "bank":
#             similar_x_i = data_augmentation_bank_item(x_i, protected)
#
#         # 计算损失函数导数符号
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i, y_cate, model)
#         sign1 = numpy.sign(grad1)
#         # 计算相似样本损失函数，及最远相似样本
#         grad2 = compute_loss_grad_AVG(similar_x_i, y_cate, model)
#         sign2 = numpy.sign(grad2)
#         direction = numpy.zeros_like(x_i)
#         for n in range(x.shape[1]):
#             if n not in protected and sign1[n] == sign2[n]:
#                 direction[0, n] = sign1[n]
#
#         # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
#         sort_privilege = sort_perturbation_direction(direction.copy(), protected)
#         if len(sort_privilege) == 0:
#             continue
#         elif len(sort_privilege) > search_times:
#             sort_privilege = sort_privilege[:search_times]
#
#         for j in sort_privilege:
#             perturbation_direction = numpy.zeros_like(x_i)
#             perturbation_direction[0, j] = direction[0, j]
#             #  扰动所选择属性
#             perturbation = extent * perturbation_direction
#             generated_x = x_i + perturbation
#             # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
#             pert_l, l_p = approximate_by_calculus_Avg(model, x_i, similar_x_i, y_cate, grad1, grad2, extent, direction)
#             x_i = generated_x
#             y_i = pert_l
#             pert_x.append(x_i)
#             pert_y.append(y_i)
#             pert_loss.append(l_p)
#
#         generate_x.append(numpy.concatenate((numpy.squeeze(numpy.array(pert_x)), numpy.array(pert_y)), axis=1))
#         loss_perturbation.append(pert_loss)
#
#     return numpy.squeeze(numpy.array(generate_x)), numpy.squeeze(numpy.array(loss_perturbation))
#
#
# def check_global_generation(model, global_seeds, dataset, protected, search_times, extent, check_tag):
#     """
#     全局搜索：依次远离 break robustness fairness 以及 both
#     :return:
#     """
#     if check_tag == "FB_G":
#         R_item, R_loss = check_false_bias_global(model, global_seeds, dataset, protected, search_times, extent)
#     elif check_tag == "TB_G":
#         R_item, R_loss = check_true_bias_global(model, global_seeds, dataset, protected, search_times, extent)
#     elif check_tag == "FF_G":
#         R_item, R_loss = check_false_fair_global(model, global_seeds, dataset, protected, search_times, extent)
#
#     return R_item, R_loss
#
#
# def check_local_generation(model, local_seeds, dataset, protected, search_times, extent, check_tag):
#     """
#     局部搜索：依次远离 break robustness fairness 以及 both
#     :return:
#     """
#     if check_tag == "FB_L":
#         R_item, R_loss = check_false_bias_local(model, local_seeds, dataset, protected, search_times, extent)
#     elif check_tag == "TB_L":
#         R_item, R_loss = check_true_bias_local(model, local_seeds, dataset, protected, search_times, extent)
#     elif check_tag == "FF_L":
#         R_item, R_loss = check_false_fair_local(model, local_seeds, dataset, protected, search_times, extent)
#
#     return R_item, R_loss
#
#
# def check_FCD_loss(model_file, test_file, similar_file, dataset, protected, t1, t2, extent, AF_tag, check_tag):
#     """
#     检测进行 true bias， false bias， false fair 扰动时， 损失函数的变化情况
#     :return:
#     """
#     model = keras.models.load_model(model_file)
#     search_seeds = get_AF_seed(model_file, test_file, similar_file, AF_tag)
#     if check_tag in {"TB_G", "FB_G", "FF_G"}:
#         item, loss = check_global_generation(model, search_seeds, dataset, protected, t1, extent, check_tag)
#     else:
#         item, loss = check_local_generation(model, search_seeds, dataset, protected, t2, extent, check_tag)
#
#     return numpy.array(item), numpy.array(loss)
#
#
# def check_RobustFair_loss(M_file, T_file1, T_file2, D_tag, P_attr, s_t, extent, AF_tag, C_tag, pic_name, dist=0, K=0):
#     """
#     AF 测试样本生成
#     :return:
#     """
#     while True:
#         RF_R, RF_L = check_FCD_loss(M_file, T_file1, T_file2, D_tag, P_attr, s_t, s_t, extent, AF_tag, C_tag)
#         if numpy.any(numpy.subtract(RF_L[:, 0], RF_L[:, 1]) > 0.001):
#             print("data:{},search:{}".format(AF_tag, pic_name))
#             break
#     model = load_model(M_file)
#     x1, y1 = numpy.split(RF_R, [-1, ], axis=1)
#     pre1 = numpy.argmax(model.predict(x1), axis=1).reshape(-1, 1)
#     MSE = calculate_MSE(pre1, y1)
#     Acc_Cond = check_dist(MSE, dist)
#
#     # 生成相似样本
#     if D_tag == "adult":
#         similar_items = data_augmentation_adult_item(x1, P_attr)
#     elif D_tag == "compas":
#         similar_items = data_augmentation_compas_item(x1, P_attr)
#     elif D_tag == "credit":
#         similar_items = data_augmentation_credit_item(x1, P_attr)
#     elif D_tag == "bank":
#         similar_items = data_augmentation_bank_item(x1, P_attr)
#
#     x2 = []
#     pre2 = []
#     for j in range(similar_items.shape[1]):
#         x2.append(similar_items[:, j, :])
#         pre2.append(numpy.argmax(model.predict(similar_items[:, j, :]), axis=1).reshape(-1, 1))
#
#     AF_cond = numpy.ones(Acc_Cond.shape)
#
#     for h in range(len(pre2)):
#         # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
#         D_distance = calculate_MSE(y1, pre2[h])
#         Kd_distance = K * calculate_MSE(x1, x2[h])
#         AF_cond = numpy.logical_and(AF_cond, check_dist(D_distance - Kd_distance, dist))
#
#     TF_num = compute_cumulative(numpy.logical_and(Acc_Cond, AF_cond))
#     TB_num = compute_cumulative(numpy.logical_and(Acc_Cond, ~AF_cond))
#     FF_num = compute_cumulative(numpy.logical_and(~Acc_Cond, AF_cond))
#     FB_num = compute_cumulative(numpy.logical_and(~Acc_Cond, ~AF_cond))
#
#     x = [a for a in range(RF_R.shape[0])]
#     y = [TF_num, TB_num, FF_num, FB_num]
#     names = ["TF", "TB", "FF", "FB"]
#     s_eval = "../dataset/{}/result/Pic_Avg_{}_D_{}_search_num.pdf".format(D_tag, AF_tag, pic_name)
#     draw_lines_num(x, y, names, x_label="Iterations", y_label="Number", P_title=pic_name, output_file=s_eval)
#
#     loss_change = RF_L[:, 0]
#     loss_change_similar = RF_L[:, 1]
#     y = [loss_change, loss_change_similar]
#     x = [a for a in range(RF_L.shape[0])]
#     names = ["Perturbed Individual", "Perturbed Similar Individual"]
#     s_eval = "../dataset/{}/result/Pic_Avg_{}_D_{}_search_loss.pdf".format(D_tag, AF_tag, pic_name)
#     draw_lines_loss(x, y, names, x_label="Iterations", y_label="Loss", P_title=pic_name, output_file=s_eval)
#
#
# def compute_cumulative(input_data):
#     """
#     计算累计值
#     :return:
#     """
#     output_data = []
#     cum_data = 0
#     for i in range(len(input_data)):
#         if input_data[i]:
#             cum_data += 1
#             output_data.append(cum_data)
#         else:
#             output_data.append(cum_data)
#
#     return output_data
