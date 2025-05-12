import pickle

import numpy
import tensorflow
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.utils.np_utils import to_categorical


def split_model_by_embedding_layer(model, input_size, embed_size):
    """
    计算模型embedding层的输出结果embedding vector
    计算各embedding vector到其余embedding vector的单位方向向量
    :return:
    """
    # 获取embedding模型
    index_inputs = Input(shape=(input_size,))
    value_inputs = Input(shape=(input_size,))
    layer_2_output = model.get_layer(index=2)(index_inputs, value_inputs)
    layer_2_model = Model(inputs=[index_inputs, value_inputs], outputs=layer_2_output)
    # 获取embedding层后的模型
    layer_3_input = Input(shape=(input_size, embed_size))
    layer_3_output = model.get_layer(index=3)(layer_3_input)
    layer_4_output = model.get_layer(index=4)(layer_3_output)
    layer_5_output = model.get_layer(index=5)(layer_4_output)
    layer_6_output = model.get_layer(index=6)(layer_5_output)
    layer_7_output = model.get_layer(index=7)(layer_6_output)
    layer_8_output = model.get_layer(index=8)(layer_7_output)
    layer_9_output = model.get_layer(index=9)(layer_8_output)
    layer_10_output = model.get_layer(index=10)(layer_9_output)
    layer_11_output = model.get_layer(index=11)(layer_10_output)
    layer_12_output = model.get_layer(index=12)(layer_11_output)
    layer_13_output = model.get_layer(index=13)(layer_12_output)
    layer_14_output = model.get_layer(index=14)(layer_13_output)

    layer_3_model = Model(inputs=layer_3_input, outputs=layer_14_output)
    return model, layer_2_model, layer_3_model


def compute_loss_grad(x, y, model, loss_func=mean_squared_error):
    """
    计算损失函数的一阶导数
    :return:
    """
    x = tensorflow.Variable(x, dtype=tensorflow.float32)
    y = tensorflow.Variable(y, dtype=tensorflow.float32)
    with tensorflow.GradientTape() as tape:
        tape.watch(x)
        loss = loss_func(y, model(x))
        dl_dx = tape.gradient(loss, x)  # 一阶导数
    return dl_dx[0]
    # x = tensorflow.Variable(x, dtype=tensorflow.float32)
    # y = tensorflow.Variable(y, dtype=tensorflow.float32)
    # with tensorflow.GradientTape(persistent=True) as tape2:
    #     tape2.watch(x)
    #     with tensorflow.GradientTape() as tape1:
    #         tape1.watch(x)
    #         loss = loss_func(y, model(x))
    #         dl_dx = tape1.gradient(loss, x)  # 一阶导数
    #     dl2_dx2 = tape2.gradient(dl_dx, x)  # 二阶导数
    # # hessian = tape2.jacobian(dl_dx, x)  # Hessian 矩阵
    # return dl_dx[0], dl2_dx2[0]


# def optimize_false_bias(grad1, grad2, delt_unit, x, sen_id, continue_index):
#     """
#     优化false bias，计算扰动与梯度的乘积
#     :return:
#     """
#     dl_dx = grad1 - grad2
#     delt_derivative = []
#     for j in range(len(delt_unit[int(x[sen_id])])):
#         if j in continue_index:
#             delt_derivative.append(+100000)
#         else:
#             delt_derivative.append(numpy.dot(dl_dx[sen_id, :], delt_unit[int(x[sen_id])][j]))
#
#     # 对delt与梯度乘积进行排序，返回最小乘积的下标
#     # sort_data = numpy.flip(numpy.sort(delt_derivative))
#     sort_data = numpy.sort(delt_derivative)
#     adv_id = delt_derivative.index(sort_data[0])
#     return adv_id
#
#
# def optimize_false_fair(grad1, grad2, delt_unit, x, sen_id, continue_index):
#     """
#     优化false fair，计算扰动与梯度的乘积
#     :return:
#     """
#     dl_dx = grad1 + grad2
#     delt_derivative = []
#     for j in range(len(delt_unit[int(x[sen_id])])):
#         if j in continue_index:
#             delt_derivative.append(+100000)
#         else:
#             delt_derivative.append(numpy.dot(dl_dx[sen_id, :], delt_unit[int(x[sen_id])][j]))
#
#     # 对delt与梯度乘积进行排序，返回最小乘积的下标
#     sort_data = numpy.sort(delt_derivative)
#     adv_id = delt_derivative.index(sort_data[0])
#     return adv_id


# def calculate_directional_derivative(grad, direction_unit, points):
#     """
#     计算梯度沿各方向的方向导数
#     :return:
#     """
#     directional_derivative = []
#     for i in range(len(points)):
#         point_directional_derivative = []
#         for j in range(len(direction_unit[int(points[i])])):
#             point_directional_derivative.append(numpy.dot(grad[i, :], direction_unit[int(points[i])][j]))
#         directional_derivative.append(point_directional_derivative)
#     return directional_derivative


def clip_numerical_data(item, low_mini, up_max):
    """
    将连续数据的取值投影到值域内：当取值大于up max时，设为up max；小于low min时，设为low min
    :return:
    """
    for i in range(item.shape[0]):
        if item[i] < low_mini[i]:
            item[i] = low_mini[i]
        if item[i] > up_max[i]:
            item[i] = up_max[i]
    return item


def select_sensitive_feature(gradient, protect):
    """
    按梯度绝对值大小对属性进行排序，选择最敏感的非保护属性进行扰动（梯度绝对值最大）
    :return:
    """
    linalg_norm = numpy.linalg.norm(gradient, axis=1).tolist()
    sort_data = numpy.flip(numpy.sort(linalg_norm))
    for i in range(len(sort_data)):
        sen_id = linalg_norm.index(sort_data[i])
        if sen_id not in protect:
            return sen_id


def optimize_tabular_data(grad1, grad2, delt_unit, x, sen_id, continue_index):
    """
    优化true bias，计算扰动与梯度的乘积
    true bias：min l(f(x+delt),y)+l(f(x'+delt),!y)
    false bias：min l(f(x+delt),!y)+l(f(x'+delt),y)
    false fair：min l(f(x+delt),!y)+l(f(x'+delt),!y)
    :return:
    """
    dl_dx = grad1 + grad2
    delt_derivative = []
    for j in range(len(delt_unit[int(x[sen_id])])):
        if j in continue_index:
            delt_derivative.append(+100000)
        else:
            delt_derivative.append(numpy.dot(dl_dx[sen_id, :], delt_unit[int(x[sen_id])][j]))

    # 对delt与梯度乘积进行排序，返回最小乘积的下标
    sort_data = numpy.sort(delt_derivative)
    adv_id = delt_derivative.index(sort_data[0])
    return adv_id


def true_bias(model, y_i, x_i, x_s, times, dir_unit, protect, low_min, up_max):
    """
    min l(f(x+delt),y)+l(f(x'+delt),!y)
    :return:
    """
    y_cate = to_categorical(y_i, num_classes=2)
    ones = numpy.ones_like(y_cate)
    not_y = ones - y_cate
    # 保存多次搜索结果
    adv_result1 = [x_i]
    adv_result2 = [x_s]
    adv_label = [y_i]
    # 获取连续属性的index取值
    continue_index = []
    for i in range(len(x_i[1])):
        if x_i[1][i] != 1:
            continue_index.append(x_i[0][i])
    for _ in range(times):
        # 计算原样本、相似样本embedding层输出结果，损失函数对embedding结果的导数
        embedding_output1 = model[1].predict([x_i[0].reshape(1, -1), x_i[1].reshape(1, -1)])
        embedding_output2 = model[1].predict([x_s[0].reshape(1, -1), x_s[1].reshape(1, -1)])
        grad1 = compute_loss_grad(embedding_output1, y_cate, model[2])
        grad2 = compute_loss_grad(embedding_output2, not_y, model[2])
        # 选择梯度最大的属性进行扰动
        sen_id = select_sensitive_feature(grad1 + grad2, protect)
        # 保存本次搜索结果
        if x_i[1][sen_id] == 1:  # category feature
            # 计算扰动delt与所选梯度的乘积
            adv_id = optimize_tabular_data(grad1, grad2, dir_unit, x_i[0], sen_id, continue_index)
            adv_x_i = numpy.copy(x_i)
            adv_x_i[0][sen_id] = adv_id
            adv_x_s = numpy.copy(x_s)
            adv_x_s[0][sen_id] = adv_id
        else:  # numberical feature
            sign = numpy.sign(numpy.sum(grad1[sen_id] + grad2[sen_id]))
            adv_x_i = numpy.copy(x_i)
            adv_x_i[1][sen_id] += -1 * sign
            adv_x_s = numpy.copy(x_s)
            adv_x_s[1][sen_id] += -1 * sign
            # 对连续属性进行裁剪，确保扰动后取值仍然在值域内
            adv_x_i[1] = clip_numerical_data(adv_x_i[1], low_min, up_max)
            adv_x_s[1] = clip_numerical_data(adv_x_s[1], low_min, up_max)
        if any(numpy.array_equal(adv_x_i, f_r) for f_r in adv_result1):
            # 当扰动结果重复出现时，为了避免重复搜索，退出本次搜索
            return adv_result1, adv_result2, adv_label, len(adv_label)
        else:
            adv_result1.append(adv_x_i)
            adv_result2.append(adv_x_s)
            adv_label.append(y_i)
            x_i = adv_x_i
            x_s = adv_x_s
    return adv_result1, adv_result2, adv_label, len(adv_label)


def false_bias(model, y_i, x_i, x_s, times, dir_unit, protect, low_min, up_max):
    """
    min l(f(x+delt),!y)+l(f(x'+delt),y)
    :return:
    """
    y_cate = to_categorical(y_i, num_classes=2)
    ones = numpy.ones_like(y_cate)
    not_y = ones - y_cate
    # 保存多次搜索结果
    adv_result1 = [x_i]
    adv_result2 = [x_s]
    adv_label = [y_i]
    # 获取连续属性的index取值
    continue_index = []
    for i in range(len(x_i[1])):
        if x_i[1][i] != 1:
            continue_index.append(x_i[0][i])
    for _ in range(times):
        # 计算原样本、相似样本embedding层输出结果，损失函数对embedding结果的导数
        embedding_output1 = model[1].predict([x_i[0].reshape(1, -1), x_i[1].reshape(1, -1)])
        embedding_output2 = model[1].predict([x_s[0].reshape(1, -1), x_s[1].reshape(1, -1)])
        grad1 = compute_loss_grad(embedding_output1, not_y, model[2])
        grad2 = compute_loss_grad(embedding_output2, y_cate, model[2])
        # 选择梯度最大的属性进行扰动
        sen_id = select_sensitive_feature(grad1 + grad2, protect)
        # 保存本次搜索结果
        if x_i[1][sen_id] == 1:  # category feature
            # 计算扰动delt与所选梯度的乘积
            adv_id = optimize_tabular_data(grad1, grad2, dir_unit, x_i[0], sen_id, continue_index)
            adv_x_i = numpy.copy(x_i)
            adv_x_i[0][sen_id] = adv_id
            adv_x_s = numpy.copy(x_s)
            adv_x_s[0][sen_id] = adv_id
        else:  # numberical feature
            sign = numpy.sign(numpy.sum(grad1[sen_id] + grad2[sen_id]))
            adv_x_i = numpy.copy(x_i)
            adv_x_i[1][sen_id] += -1 * sign
            adv_x_s = numpy.copy(x_s)
            adv_x_s[1][sen_id] += -1 * sign
            # 对连续属性进行裁剪，确保扰动后取值仍然在值域内
            adv_x_i[1] = clip_numerical_data(adv_x_i[1], low_min, up_max)
            adv_x_s[1] = clip_numerical_data(adv_x_s[1], low_min, up_max)
        if any(numpy.array_equal(adv_x_i, f_r) for f_r in adv_result1):
            # 当扰动结果重复出现时，为了避免重复搜索，退出本次搜索
            return adv_result1, adv_result2, adv_label, len(adv_label)
        else:
            adv_result1.append(adv_x_i)
            adv_result2.append(adv_x_s)
            adv_label.append(y_i)
            x_i = adv_x_i
            x_s = adv_x_s
    return adv_result1, adv_result2, adv_label, len(adv_label)
    # for _ in range(times):
    #     # 计算原样本、相似样本embedding层输出结果进行预测时损失函数的导数
    #     embedding_output1 = model[1].predict([x_i[0].reshape(1, -1), x_i[1].reshape(1, -1)])
    #     embedding_output2 = model[1].predict([x_s[0].reshape(1, -1), x_s[1].reshape(1, -1)])
    #     grad1 = compute_loss_grad(embedding_output1, y_cate, model[2])
    #     grad2 = compute_loss_grad(embedding_output2, y_cate, model[2])
    #     # 根据样本损失函数、相似样本损失函数embedding vector导数计算方向导数，选择扰动的离散非敏感属性
    #     dir_derivative1 = calculate_directional_derivative(grad1[0], dir_unit[0], x_i[0])
    #     dir_derivative2 = calculate_directional_derivative(grad2[0], dir_unit[0], x_s[0])
    #     # 选择最敏感的非保护属性进行扰动
    #     sen_id = select_sensitive_feature(grad1[0], protect_index)
    #     print(dir_derivative1[sen_id])
    #     print(dir_derivative2[sen_id])
    #     # 保存本次搜索结果
    #     intermediate_result1 = []
    #     intermediate_result2 = []
    #     if x_i[1][sen_id] == 1:  # category feature
    #         # # 确定属性取值编码起点与终点
    #         # if sen_id == 0:
    #         #     start_index = 0
    #         #     end_index = dir_unit[1][sen_id]
    #         # else:
    #         #     start_index = dir_unit[1][sen_id - 1]
    #         #     end_index = dir_unit[1][sen_id]
    #         # if end_index - start_index > 1:
    #         #     for j in range(start_index, end_index):
    #         #         if dir_derivative1[sen_id][j] > 0 and dir_derivative2[sen_id][j] < 0:
    #         #             adv_x_i = numpy.copy(x_i)
    #         #             adv_x_i[0][sen_id] = j
    #         #             intermediate_result1.append(adv_x_i)
    #         #             adv_max_sim = numpy.copy(x_s)
    #         #             adv_max_sim[0][sen_id] = j
    #         #             intermediate_result2.append(adv_max_sim)
    #
    #         for j in range(len(dir_derivative1[sen_id])):
    #             if j in continue_index:
    #                 continue  # 当index为连续属性编码时，跳过本次循环
    #             # 分析向各个词向量扰动时的方向导数：方向导数大于0，损失函数增加；方向导数小于0，损失函数减小
    #             if dir_derivative1[sen_id][j] > 0 and dir_derivative2[sen_id][j] < 0:
    #                 adv_x_i = numpy.copy(x_i)
    #                 adv_x_i[0][sen_id] = j
    #                 intermediate_result1.append(adv_x_i)
    #                 adv_max_sim = numpy.copy(x_s)
    #                 adv_max_sim[0][sen_id] = j
    #                 intermediate_result2.append(adv_max_sim)
    #     elif x_i[1][sen_id] != 1:  # numberical feature
    #         if numpy.sum(grad1[0][sen_id]) > 0 and numpy.sum(grad2[0][sen_id]) < 0:
    #             adv_x_i = numpy.copy(x_i)
    #             adv_x_i[1][sen_id] += 1
    #             intermediate_result1.append(adv_x_i)
    #             adv_max_sim = numpy.copy(x_s)
    #             adv_max_sim[1][sen_id] += 1
    #             intermediate_result2.append(adv_max_sim)
    #     if len(intermediate_result1) == 0:  # 当没有属性可扰动时，退出搜索
    #         return adv_result1, adv_result2, adv_label, len(adv_label)
    #     else:  # 根据扰动后样本的马氏距离选择恰当的属性进行扰动，以保持样本属性间的相关性
    #         mah_dist = mahalanobis_distance(intermediate_result1, adv_result1[0], covariance)
    #         inter_x_i = intermediate_result1[mah_dist.index(min(mah_dist))]
    #         inter_x_s = intermediate_result2[mah_dist.index(min(mah_dist))]
    #         # 对连续属性进行裁剪，确保扰动后取值仍然在值域内
    #         inter_x_i[1] = clip_numerical_data(inter_x_i[1], low_min, up_max)
    #         inter_x_s[1] = clip_numerical_data(inter_x_s[1], low_min, up_max)
    #         if any(numpy.array_equal(inter_x_i, f_r) for f_r in adv_result1):
    #             # 当扰动结果重复出现时，为了避免重复搜索，退出本次搜索
    #             return adv_result1, adv_result2, adv_label, len(adv_label)
    #         else:
    #             adv_result1.append(inter_x_i)
    #             adv_result2.append(inter_x_s)
    #             adv_label.append(y_i)
    #             x_i = inter_x_i
    #             x_s = inter_x_s
    #             # 检查扰动结果的准确公平性，若扰动结果FF，扰动成功，退出搜索
    #             AF = check_item_AF(model[0], x_i, x_s, y_i)
    #             if AF[3]:
    #                 return adv_result1, adv_result2, adv_label, len(adv_label)
    # return adv_result1, adv_result2, adv_label, len(adv_label)


def false_fair(model, y_i, x_i, x_s, times, dir_unit, protect, low_min, up_max):
    """
    min l(f(x+delt),!y)+l(f(x'+delt),!y)
    :return:
    """
    y_cate = to_categorical(y_i, num_classes=2)
    ones = numpy.ones_like(y_cate)
    not_y = ones - y_cate
    # 保存多次搜索结果
    adv_result1 = [x_i]
    adv_result2 = [x_s]
    adv_label = [y_i]
    # 获取连续属性的index取值
    continue_index = []
    for i in range(len(x_i[1])):
        if x_i[1][i] != 1:
            continue_index.append(x_i[0][i])
    for _ in range(times):
        # 计算原样本、相似样本embedding层输出结果，损失函数对embedding结果的导数
        embedding_output1 = model[1].predict([x_i[0].reshape(1, -1), x_i[1].reshape(1, -1)])
        embedding_output2 = model[1].predict([x_s[0].reshape(1, -1), x_s[1].reshape(1, -1)])
        grad1 = compute_loss_grad(embedding_output1, not_y, model[2])
        grad2 = compute_loss_grad(embedding_output2, not_y, model[2])
        # 选择梯度最大的属性进行扰动
        sen_id = select_sensitive_feature(grad1 + grad2, protect)
        # 保存本次搜索结果
        if x_i[1][sen_id] == 1:  # category feature
            # 计算扰动delt与所选梯度的乘积
            adv_id = optimize_tabular_data(grad1, grad2, dir_unit, x_i[0], sen_id, continue_index)
            adv_x_i = numpy.copy(x_i)
            adv_x_i[0][sen_id] = adv_id
            adv_x_s = numpy.copy(x_s)
            adv_x_s[0][sen_id] = adv_id
        else:  # numberical feature
            sign = numpy.sign(numpy.sum(grad1[sen_id] + grad2[sen_id]))
            adv_x_i = numpy.copy(x_i)
            adv_x_i[1][sen_id] += -1 * sign
            adv_x_s = numpy.copy(x_s)
            adv_x_s[1][sen_id] += -1 * sign
            # 对连续属性进行裁剪，确保扰动后取值仍然在值域内
            adv_x_i[1] = clip_numerical_data(adv_x_i[1], low_min, up_max)
            adv_x_s[1] = clip_numerical_data(adv_x_s[1], low_min, up_max)
        if any(numpy.array_equal(adv_x_i, f_r) for f_r in adv_result1):
            # 当扰动结果重复出现时，为了避免重复搜索，退出本次搜索
            return adv_result1, adv_result2, adv_label, len(adv_label)
        else:
            adv_result1.append(adv_x_i)
            adv_result2.append(adv_x_s)
            adv_label.append(y_i)
            x_i = adv_x_i
            x_s = adv_x_s
    return adv_result1, adv_result2, adv_label, len(adv_label)


# NLP
def transform_word_to_token(word_dic, token_dic):
    """
    将单词转换为token
    :return:
    """
    result_dic = {}
    for w in word_dic:
        if w in token_dic.keys():
            token_w = token_dic[w]
            w_keys = word_dic[w]
            token_w_keys = [token_w]
            for w_k in w_keys:
                if w_k in token_dic.keys():
                    token_w_keys.append(token_dic[w_k])
            for t_w_k in token_w_keys:
                result_dic[t_w_k] = token_w_keys
    return result_dic


def get_number_token(token_dic):
    """
    获取token列表中的数字token
    :param token_dic:
    :return:
    """
    number_tokens = []
    keys = list(token_dic.keys())
    for k in keys:
        if k.isdigit():
            number_tokens.append(token_dic[k])
    return number_tokens


def split_NLP_model_by_embedding_layer(model, input_size, embed_size):
    """
    计算模型embedding层的输出结果embedding vector
    计算各embedding vector到其余embedding vector的单位方向向量
    :return:
    """
    # 获取embedding模型
    model_inputs = Input(shape=(input_size,))
    layer_1_output = model.get_layer(index=1)(model_inputs)
    layer_1_model = Model(inputs=model_inputs, outputs=layer_1_output)
    # 获取embedding层后的模型
    layer_2_input = Input(shape=(input_size, embed_size))
    layer_2_output = model.get_layer(index=2)(layer_2_input)
    layer_3_output = model.get_layer(index=3)(layer_2_output)
    layer_4_output = model.get_layer(index=4)(layer_3_output)
    layer_5_output = model.get_layer(index=5)(layer_4_output)
    layer_6_output = model.get_layer(index=6)(layer_5_output)
    layer_7_output = model.get_layer(index=7)(layer_6_output)
    layer_8_output = model.get_layer(index=8)(layer_7_output)
    layer_9_output = model.get_layer(index=9)(layer_8_output)
    layer_10_output = model.get_layer(index=10)(layer_9_output)
    layer_11_output = model.get_layer(index=11)(layer_10_output)
    layer_12_output = model.get_layer(index=12)(layer_11_output)
    layer_13_output = model.get_layer(index=13)(layer_12_output)

    layer_2_model = Model(inputs=layer_2_input, outputs=layer_13_output)
    return model, layer_1_model, layer_2_model


# def select_NLP_sensitive_feature(gradient, protect, input_data, adv_token):
#     """
#     按梯度绝对值大小对属性进行排序，选择最敏感的非保护属性进行扰动（梯度绝对值最大）
#     :return:
#     """
#     linalg_norm = numpy.linalg.norm(gradient, axis=1).tolist()
#     sort_data = numpy.flip(numpy.sort(linalg_norm))
#     for i in range(len(sort_data)):
#         sen_id = linalg_norm.index(sort_data[i])
#         if not protect[sen_id] and input_data[sen_id] in adv_token:
#             return sen_id


def get_adult_embedding_vector_direction_unit(model, input_size):
    """
    计算模型embedding层的输出结果embedding vector
    计算各embedding vector到其余embedding vector的单位方向向量
    :return:
    """
    embedding_vector = []
    with open("../dataset/NLP/adult/data/vocab_dic.pkl", 'rb') as f:
        word_index_dic = pickle.load(f)
    with open("../dataset/NLP/adult/data/adv_synonyms_dic.pkl", 'rb') as f:
        adv_synonyms_dic = pickle.load(f)
    adv_synonyms_token = transform_word_to_token(adv_synonyms_dic, word_index_dic)
    number_token = get_number_token(word_index_dic)

    for i in range(1500):  # 根据编码信息及embedding模型获取编码后的编码向量vector,共1500个单词
        raw_data = numpy.array([i] * input_size)
        embedding_vector.append(model(raw_data))

    # 根据同义词字token，以及数字token 为各个属性确定扰动范围,仅保留同义词的编码
    direction_unit = {}
    direction_index = {}
    for i in range(len(embedding_vector)):
        if i not in adv_synonyms_token.keys() and i not in number_token:
            continue
        elif i in adv_synonyms_token.keys():
            perturbation_delt = []
            perturbation_index = []
            for j in adv_synonyms_token[i]:  # 为各个同义词创建词典，key为扰动，value为扰动后同义词的索引
                perturbation_delt.append(embedding_vector[j] - embedding_vector[i])
                perturbation_index.append(j)
            direction_unit[i] = perturbation_delt
            direction_index[i] = perturbation_index
        elif i in number_token:
            perturbation_delt = []
            perturbation_index = []
            for j in number_token:
                perturbation_delt.append(embedding_vector[j] - embedding_vector[i])
                perturbation_index.append(j)
            direction_unit[i] = perturbation_delt
            direction_index[i] = perturbation_index

    return direction_unit, direction_index


def get_bank_embedding_vector_direction_unit(model, input_size):
    """
    计算模型embedding层的输出结果embedding vector
    计算各embedding vector到其余embedding vector的单位方向向量
    :return:
    """
    embedding_vector = []
    with open("../dataset/NLP/bank/data/vocab_dic.pkl", 'rb') as f:
        word_index_dic = pickle.load(f)
    with open("../dataset/NLP/bank/data/adv_synonyms_dic.pkl", 'rb') as f:
        adv_synonyms_dic = pickle.load(f)
    adv_synonyms_token = transform_word_to_token(adv_synonyms_dic, word_index_dic)
    number_token = get_number_token(word_index_dic)
    for i in range(1500):  # 根据编码信息及embedding模型获取编码后的编码向量vector,共1500个单词
        raw_data = numpy.array([i] * input_size)
        embedding_vector.append(model(raw_data))

    # 根据同义词字token，以及数字token 为各个属性确定扰动范围,仅保留同义词的编码
    direction_unit = {}
    direction_index = {}
    for i in range(len(embedding_vector)):
        if i not in adv_synonyms_token.keys() and i not in number_token:
            continue
        elif i in adv_synonyms_token.keys():
            perturbation_delt = []
            perturbation_index = []
            for j in adv_synonyms_token[i]:  # 为各个同义词创建词典，key为扰动，value为扰动后同义词的索引
                if j < 1500:
                    perturbation_delt.append(embedding_vector[j] - embedding_vector[i])
                    perturbation_index.append(j)
            direction_unit[i] = perturbation_delt
            direction_index[i] = perturbation_index
        elif i in number_token:
            perturbation_delt = []
            perturbation_index = []
            for j in number_token:
                if j < 1500:
                    perturbation_delt.append(embedding_vector[j] - embedding_vector[i])
                    perturbation_index.append(j)
            direction_unit[i] = perturbation_delt
            direction_index[i] = perturbation_index

    return direction_unit, direction_index


def get_compas_embedding_vector_direction_unit(model, input_size):
    """
    计算模型embedding层的输出结果embedding vector
    计算各embedding vector到其余embedding vector的单位方向向量
    :return:
    """
    embedding_vector = []
    with open("../dataset/NLP/compas/data/vocab_dic.pkl", 'rb') as f:
        word_index_dic = pickle.load(f)
    with open("../dataset/NLP/compas/data/adv_synonyms_dic.pkl", 'rb') as f:
        adv_synonyms_dic = pickle.load(f)
    adv_synonyms_token = transform_word_to_token(adv_synonyms_dic, word_index_dic)
    number_token = get_number_token(word_index_dic)
    for i in range(1500):  # 根据编码信息及embedding模型获取编码后的编码向量vector,共1500个单词
        raw_data = numpy.array([i] * input_size)
        embedding_vector.append(model(raw_data))

    # 根据同义词字token，以及数字token 为各个属性确定扰动范围,仅保留同义词的编码
    direction_unit = {}
    direction_index = {}
    for i in range(len(embedding_vector)):
        if i not in adv_synonyms_token.keys() and i not in number_token:
            continue
        elif i in adv_synonyms_token.keys():
            perturbation_delt = []
            perturbation_index = []
            for j in adv_synonyms_token[i]:  # 为各个同义词创建词典，key为扰动，value为扰动后同义词的索引
                if j < 1500:
                    perturbation_delt.append(embedding_vector[j] - embedding_vector[i])
                    perturbation_index.append(j)
            direction_unit[i] = perturbation_delt
            direction_index[i] = perturbation_index
        elif i in number_token:
            perturbation_delt = []
            perturbation_index = []
            for j in number_token:
                if j < 1500:
                    perturbation_delt.append(embedding_vector[j] - embedding_vector[i])
                    perturbation_index.append(j)
            direction_unit[i] = perturbation_delt
            direction_index[i] = perturbation_index

    return direction_unit, direction_index


def get_ACSEmployment_embedding_vector_direction_unit(model, input_size):
    """
    计算模型embedding层的输出结果embedding vector
    计算各embedding vector到其余embedding vector的单位方向向量
    :return:
    """
    embedding_vector = []
    with open("../dataset/ACS/NLP/employment/data/vocab_dic.pkl", 'rb') as f:
        word_index_dic = pickle.load(f)
    with open("../dataset/ACS/NLP/employment/data/adv_synonyms_dic.pkl", 'rb') as f:
        adv_synonyms_dic = pickle.load(f)
    adv_synonyms_token = transform_word_to_token(adv_synonyms_dic, word_index_dic)
    number_token = get_number_token(word_index_dic)
    for i in range(1500):  # 根据编码信息及embedding模型获取编码后的编码向量vector,共1500个单词
        raw_data = numpy.array([i] * input_size)
        embedding_vector.append(model(raw_data))

    # 根据同义词字token，以及数字token 为各个属性确定扰动范围,仅保留同义词的编码
    direction_unit = {}
    direction_index = {}
    for i in range(len(embedding_vector)):
        if i not in adv_synonyms_token.keys() and i not in number_token:
            continue
        elif i in adv_synonyms_token.keys():
            perturbation_delt = []
            perturbation_index = []
            for j in adv_synonyms_token[i]:  # 为各个同义词创建词典，key为扰动，value为扰动后同义词的索引
                if j < 1500:
                    perturbation_delt.append(embedding_vector[j] - embedding_vector[i])
                    perturbation_index.append(j)
            direction_unit[i] = perturbation_delt
            direction_index[i] = perturbation_index
        elif i in number_token:
            perturbation_delt = []
            perturbation_index = []
            for j in number_token:
                if j < 1500:
                    perturbation_delt.append(embedding_vector[j] - embedding_vector[i])
                    perturbation_index.append(j)
            direction_unit[i] = perturbation_delt
            direction_index[i] = perturbation_index

    return direction_unit, direction_index


def get_ACSIncome_embedding_vector_direction_unit(model, input_size):
    """
    计算模型embedding层的输出结果embedding vector
    计算各embedding vector到其余embedding vector的单位方向向量
    :return:
    """
    embedding_vector = []
    with open("../dataset/ACS/NLP/income/data/vocab_dic.pkl", 'rb') as f:
        word_index_dic = pickle.load(f)
    with open("../dataset/ACS/NLP/income/data/adv_synonyms_dic.pkl", 'rb') as f:
        adv_synonyms_dic = pickle.load(f)
    adv_synonyms_token = transform_word_to_token(adv_synonyms_dic, word_index_dic)
    number_token = get_number_token(word_index_dic)
    for i in range(1500):  # 根据编码信息及embedding模型获取编码后的编码向量vector,共1500个单词
        raw_data = numpy.array([i] * input_size)
        embedding_vector.append(model(raw_data))

    # 根据同义词字token，以及数字token 为各个属性确定扰动范围,仅保留同义词的编码
    direction_unit = {}
    direction_index = {}
    for i in range(len(embedding_vector)):
        if i not in adv_synonyms_token.keys() and i not in number_token:
            continue
        elif i in adv_synonyms_token.keys():
            perturbation_delt = []
            perturbation_index = []
            for j in adv_synonyms_token[i]:  # 为各个同义词创建词典，key为扰动，value为扰动后同义词的索引
                if j < 1500:
                    perturbation_delt.append(embedding_vector[j] - embedding_vector[i])
                    perturbation_index.append(j)
            direction_unit[i] = perturbation_delt
            direction_index[i] = perturbation_index
        elif i in number_token:
            perturbation_delt = []
            perturbation_index = []
            for j in number_token:
                if j < 1500:
                    perturbation_delt.append(embedding_vector[j] - embedding_vector[i])
                    perturbation_index.append(j)
            direction_unit[i] = perturbation_delt
            direction_index[i] = perturbation_index

    return direction_unit, direction_index


def get_ACSCoverage_embedding_vector_direction_unit(model, input_size):
    """
    计算模型embedding层的输出结果embedding vector
    计算各embedding vector到其余embedding vector的单位方向向量
    :return:
    """
    embedding_vector = []
    with open("../dataset/ACS/NLP/coverage/data/vocab_dic.pkl", 'rb') as f:
        word_index_dic = pickle.load(f)
    with open("../dataset/ACS/NLP/coverage/data/adv_synonyms_dic.pkl", 'rb') as f:
        adv_synonyms_dic = pickle.load(f)
    adv_synonyms_token = transform_word_to_token(adv_synonyms_dic, word_index_dic)
    number_token = get_number_token(word_index_dic)
    for i in range(1500):  # 根据编码信息及embedding模型获取编码后的编码向量vector,共1500个单词
        raw_data = numpy.array([i] * input_size)
        embedding_vector.append(model(raw_data))

    # 根据同义词字token，以及数字token 为各个属性确定扰动范围,仅保留同义词的编码
    direction_unit = {}
    direction_index = {}
    for i in range(len(embedding_vector)):
        if i not in adv_synonyms_token.keys() and i not in number_token:
            continue
        elif i in adv_synonyms_token.keys():
            perturbation_delt = []
            perturbation_index = []
            for j in adv_synonyms_token[i]:  # 为各个同义词创建词典，key为扰动，value为扰动后同义词的索引
                if j < 1500:
                    perturbation_delt.append(embedding_vector[j] - embedding_vector[i])
                    perturbation_index.append(j)
            direction_unit[i] = perturbation_delt
            direction_index[i] = perturbation_index
        elif i in number_token:
            perturbation_delt = []
            perturbation_index = []
            for j in number_token:
                if j < 1500:
                    perturbation_delt.append(embedding_vector[j] - embedding_vector[i])
                    perturbation_index.append(j)
            direction_unit[i] = perturbation_delt
            direction_index[i] = perturbation_index

    return direction_unit, direction_index


def get_ACSTravel_embedding_vector_direction_unit(model, input_size):
    """
    计算模型embedding层的输出结果embedding vector
    计算各embedding vector到其余embedding vector的单位方向向量
    :return:
    """
    embedding_vector = []
    with open("../dataset/ACS/NLP/travel/data/vocab_dic.pkl", 'rb') as f:
        word_index_dic = pickle.load(f)
    with open("../dataset/ACS/NLP/travel/data/adv_synonyms_dic.pkl", 'rb') as f:
        adv_synonyms_dic = pickle.load(f)
    adv_synonyms_token = transform_word_to_token(adv_synonyms_dic, word_index_dic)
    number_token = get_number_token(word_index_dic)
    for i in range(1500):  # 根据编码信息及embedding模型获取编码后的编码向量vector,共1500个单词
        raw_data = numpy.array([i] * input_size)
        embedding_vector.append(model(raw_data))

    # 根据同义词字token，以及数字token 为各个属性确定扰动范围,仅保留同义词的编码
    direction_unit = {}
    direction_index = {}
    for i in range(len(embedding_vector)):
        if i not in adv_synonyms_token.keys() and i not in number_token:
            continue
        elif i in adv_synonyms_token.keys():
            perturbation_delt = []
            perturbation_index = []
            for j in adv_synonyms_token[i]:  # 为各个同义词创建词典，key为扰动，value为扰动后同义词的索引
                if j < 1500:
                    perturbation_delt.append(embedding_vector[j] - embedding_vector[i])
                    perturbation_index.append(j)
            direction_unit[i] = perturbation_delt
            direction_index[i] = perturbation_index
        elif i in number_token:
            perturbation_delt = []
            perturbation_index = []
            for j in number_token:
                if j < 1500:
                    perturbation_delt.append(embedding_vector[j] - embedding_vector[i])
                    perturbation_index.append(j)
            direction_unit[i] = perturbation_delt
            direction_index[i] = perturbation_index

    return direction_unit, direction_index


def optimize_NLP_data(grad1, grad2, delt_unit, input_data, protect):
    """
    优化false fair，计算扰动与梯度的乘积
    true bias：min l(f(x+delt),y)+l(f(x'+delt),!y)
    false bias：min l(f(x+delt),!y)+l(f(x'+delt),y)
    false fair：min l(f(x+delt),!y)+l(f(x'+delt),!y)
    :return:
    """
    dl_dx = grad1 + grad2
    delt_derivative = []
    for position in range(len(input_data)):
        dl_dx_position = dl_dx[position]
        code_position = input_data[position]
        delt_derivative_j = []

        if code_position in delt_unit[0].keys():  # 该属性存在同义词
            for synonyms_token in delt_unit[0][code_position]:
                delt_derivative_j.append(numpy.dot(dl_dx_position, synonyms_token[position, :]))
        delt_derivative.append(delt_derivative_j)

    min_delt_derivative = []
    delt_derivative_index = []
    for position in range(len(delt_derivative)):
        # 各个属性delt与梯度乘积进行排序，返回最小乘积、及其的下标
        if len(delt_derivative[position]) > 0:
            sort_data = numpy.sort(delt_derivative[position])
            adv_id = delt_derivative[position].index(sort_data[0])
            min_delt_derivative.append(sort_data[0])
            delt_derivative_index.append(delt_unit[1][input_data[position]][adv_id])
        else:
            min_delt_derivative.append(None)
            delt_derivative_index.append(None)

    min_derivative = 100000
    min_sen_id = None
    min_adv_index = None
    # 选择delt与gradient乘积最小的值（非保护属性），返回对应位置与扰动后的属性编码
    for i in range(len(min_delt_derivative)):
        if min_delt_derivative[i] is not None and min_delt_derivative[i] < min_derivative and protect[i]:
            min_derivative = min_delt_derivative[i]
            min_sen_id = i
            min_adv_index = delt_derivative_index[i]
    return min_sen_id, min_adv_index


def NLP_true_bias(model, y_i, x_d, x_s, times, dir_unit, protect):
    """
    min l(f(x+delt),y)+l(f(x'+delt),!y)
    :return:
    """
    y_cate = to_categorical(y_i, num_classes=2)
    ones = numpy.ones_like(y_cate)
    not_y = ones - y_cate
    # 保存多次搜索结果
    adv_result1 = [x_d]
    adv_result2 = [x_s]
    adv_label = [y_i]
    for _ in range(times):
        # 计算原样本、相似样本embedding层输出结果，损失函数对embedding结果的导数
        embedding_output1 = model[1].predict(x_d.reshape(1, -1))
        embedding_output2 = model[1].predict(x_s.reshape(1, -1))
        grad1 = compute_loss_grad(embedding_output1, y_cate, model[2])
        grad2 = compute_loss_grad(embedding_output2, not_y, model[2])
        # 根据扰动delt与梯度的乘积，选择扰动位置及属性值
        sen_id, adv_id = optimize_NLP_data(grad1, grad2, dir_unit, x_d, protect)
        if sen_id is None:
            return adv_result1, adv_result2, adv_label, len(adv_label)
        adv_x_i = numpy.copy(x_d)
        adv_x_i[sen_id] = adv_id
        adv_x_s = numpy.copy(x_s)
        adv_x_s[sen_id] = adv_id
        if any(numpy.array_equal(adv_x_i, f_r) for f_r in adv_result1):
            # 当扰动结果重复出现时，为了避免重复搜索，退出本次搜索
            return adv_result1, adv_result2, adv_label, len(adv_label)
        else:  # 保存本次搜索结果
            adv_result1.append(adv_x_i)
            adv_result2.append(adv_x_s)
            adv_label.append(y_i)
            x_d = adv_x_i
            x_s = adv_x_s
    return adv_result1, adv_result2, adv_label, len(adv_label)


def NLP_false_bias(model, y_i, x_d, x_s, times, dir_unit, protect):
    """
    min l(f(x+delt),!y)+l(f(x'+delt),y)
    :return:
    """
    y_cate = to_categorical(y_i, num_classes=2)
    ones = numpy.ones_like(y_cate)
    not_y = ones - y_cate
    # 保存多次搜索结果
    adv_result1 = [x_d]
    adv_result2 = [x_s]
    adv_label = [y_i]
    for _ in range(times):
        # 计算原样本、相似样本embedding层输出结果，损失函数对embedding结果的导数
        embedding_output1 = model[1].predict(x_d.reshape(1, -1))
        embedding_output2 = model[1].predict(x_s.reshape(1, -1))
        grad1 = compute_loss_grad(embedding_output1, not_y, model[2])
        grad2 = compute_loss_grad(embedding_output2, y_cate, model[2])
        # 根据扰动delt与梯度的乘积，选择扰动位置及属性值
        sen_id, adv_id = optimize_NLP_data(grad1, grad2, dir_unit, x_d, protect)
        if sen_id is None:
            return adv_result1, adv_result2, adv_label, len(adv_label)
        adv_x_i = numpy.copy(x_d)
        adv_x_i[sen_id] = adv_id
        adv_x_s = numpy.copy(x_s)
        adv_x_s[sen_id] = adv_id
        if any(numpy.array_equal(adv_x_i, f_r) for f_r in adv_result1):
            # 当扰动结果重复出现时，为了避免重复搜索，退出本次搜索
            return adv_result1, adv_result2, adv_label, len(adv_label)
        else:  # 保存本次搜索结果
            adv_result1.append(adv_x_i)
            adv_result2.append(adv_x_s)
            adv_label.append(y_i)
            x_d = adv_x_i
            x_s = adv_x_s
    return adv_result1, adv_result2, adv_label, len(adv_label)


def NLP_false_fair(model, y_i, x_d, x_s, times, dir_unit, protect):
    """
    min l(f(x+delt),!y)+l(f(x'+delt),!y)
    :return:
    """
    y_cate = to_categorical(y_i, num_classes=2)
    ones = numpy.ones_like(y_cate)
    not_y = ones - y_cate
    # 保存多次搜索结果
    adv_result1 = [x_d]
    adv_result2 = [x_s]
    adv_label = [y_i]
    for _ in range(times):
        # 计算原样本、相似样本embedding层输出结果，损失函数对embedding结果的导数
        embedding_output1 = model[1].predict(x_d.reshape(1, -1))
        embedding_output2 = model[1].predict(x_s.reshape(1, -1))
        grad1 = compute_loss_grad(embedding_output1, not_y, model[2])
        grad2 = compute_loss_grad(embedding_output2, not_y, model[2])
        # 根据扰动delt与梯度的乘积，选择扰动位置及属性值
        sen_id, adv_id = optimize_NLP_data(grad1, grad2, dir_unit, x_d, protect)
        if sen_id is None:
            return adv_result1, adv_result2, adv_label, len(adv_label)
        adv_x_i = numpy.copy(x_d)
        adv_x_i[sen_id] = adv_id
        adv_x_s = numpy.copy(x_s)
        adv_x_s[sen_id] = adv_id
        if any(numpy.array_equal(adv_x_i, f_r) for f_r in adv_result1):
            # 当扰动结果重复出现时，为了避免重复搜索，退出本次搜索
            return adv_result1, adv_result2, adv_label, len(adv_label)
        else:  # 保存本次搜索结果
            adv_result1.append(adv_x_i)
            adv_result2.append(adv_x_s)
            adv_label.append(y_i)
            x_d = adv_x_i
            x_s = adv_x_s
    return adv_result1, adv_result2, adv_label, len(adv_label)

# def add_perturbation_to_similar(similar_items, replace_data, perturbation, P_index, low_min, up_max):
#     """
#     向所用相似样本添加相同扰动
#     :return:
#     """
#     perturbed_index = []
#     perturbed_value = []
#     for i in range(similar_items[0].shape[0]):
#         # 离散属性进行替换
#         for r in range(len(replace_data)):
#             if replace_data[r] != 0 and r not in P_index:
#                 similar_items[0][i][r] = replace_data[r]
#         perturbed_index.append(similar_items[0][i])
#         # 连续属性进行修改(同时裁剪)
#         perturbed_value.append(clip_numerical_data(similar_items[1][i] + numpy.array(perturbation), low_min, up_max))
#     return [numpy.array(perturbed_index), numpy.array(perturbed_value)]


# def TrueBiasedGeneration(model_file, raw_file, similar_file, search_time, result_file, feature_size, protect_index,
#                          shuffle_tag=False):
#     """
#     进行 true bias 准确公平性测试
#     :return:
#     """
#     custom_layers = {'AdultEmbedding': AdultEmbedding,
#                      'AutoIntTransformerBlock': AutoIntTransformerBlock}
#     model = keras.models.load_model(model_file, custom_objects=custom_layers)
#
#     if shuffle_tag:
#         label = numpy.load(raw_file[2]).reshape(-1, 1)
#         shuffle_index = numpy.arange(label.shape[0])
#         numpy.random.shuffle(shuffle_index)
#         split_index = int(label.shape[0] * 0.20)
#
#         index = numpy.load(raw_file[0])[shuffle_index][:split_index]
#         value = numpy.load(raw_file[1])[shuffle_index][:split_index]
#         label = label[shuffle_index][:split_index]
#         cov = [numpy.cov(index, rowvar=False), numpy.cov(value, rowvar=False)]
#
#         s_index = numpy.load(similar_file[0])[:, shuffle_index, :][:, : split_index, :]
#         s_value = numpy.load(similar_file[1])[:, shuffle_index, :][:, : split_index, :]
#         s_label = numpy.load(similar_file[2])[:, shuffle_index][:, : split_index]
#
#     else:
#         index = numpy.load(raw_file[0])
#         value = numpy.load(raw_file[1])
#         label = numpy.load(raw_file[2]).reshape(-1, 1)
#         cov = [numpy.cov(index, rowvar=False), numpy.cov(value, rowvar=False)]
#
#         s_index = numpy.load(similar_file[0])
#         s_value = numpy.load(similar_file[1])
#         s_label = numpy.load(similar_file[2])
#
#     dir_unit = get_embedding_vector_direction_unit(model_file, feature_size)
#
#     adv_index = []
#     adv_value = []
#     adv_label = []
#     adv_search_time = []
#
#     for i in range(label.shape[0]):
#         print("check {}th item".format(i))
#         false_bias = true_bias_generation(model, label[i], [index[i], value[i]], [s_index[:, i, :], s_value[:, i, :]],
#                                           search_time, dir_unit, protect_index, cov)
#         adv_index.extend(false_bias[0])
#         adv_value.extend(false_bias[1])
#         adv_label.extend(false_bias[2])
#         adv_search_time.append(false_bias[3])
#
#     numpy.save(result_file[0], adv_index)
#     numpy.save(result_file[1], adv_value)
#     numpy.save(result_file[2], adv_label)
#     numpy.save(result_file[3], adv_search_time)
#
#
# def true_bias_generation(model, label, item, similar_items, search_times, dir_unit, protect_index, covariance):
#     """
#     离散非敏感属性:根据样本损失函数、相似样本损失函数embedding vector导数计算方向导数.
#         true bias： 沿着原样本负方向导数，相似样本正方向导数方向扰动
#     连续非敏感属性:根据样本损失函数、相似样本损失函数符号。
#         true bias： 选择梯度符号相反的属性扰动，沿着原样本负梯度方向扰动
#     :return:
#     """
#     low_min = [0] * item[0].shape[0]
#     up_max = [1000000] * item[0].shape[0]
#     x_i = item.copy()
#     y_i = label.copy()
#     # 生成结果
#     data_x_index = []
#     data_x_value = []
#     data_y = []
#     for _ in range(search_times):
#         # 计算y原样本损失函数embedding vector导数、连续输入值导数
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i[0], x_i[1], y_cate, model)
#         # 计算相似样本损失函数embedding vector导数、连续输入值导数
#         grad2, max_similar = compute_loss_grad_MP(similar_items[0], similar_items[1], y_cate, model)
#         # 根据样本损失函数、相似样本损失函数embedding vector导数计算方向导数，选择扰动的离散非敏感属性
#         replace_data = numpy.zeros_like(x_i[1])
#         dir_derivative1 = calculate_directional_derivative(grad1[0], dir_unit[0], x_i[0])
#         dir_derivative2 = calculate_directional_derivative(grad2[0], dir_unit[0], max_similar[0])
#         delta = numpy.zeros_like(x_i[1])
#         # 离散属性
#         for m in range(x_i[0].shape[0]):
#             if x_i[1][m] == 1 and m not in protect_index:  # 离散属性，且不是保护属性
#                 if m == 0:
#                     start_index = 0
#                     end_index = dir_unit[1][m]
#                 else:
#                     start_index = dir_unit[1][m - 1]
#                     end_index = dir_unit[1][m]
#                 # 截取恰当的单位方向向量（该属性内的所有取值方向向量）
#                 feat_change1 = dir_derivative1[m][start_index:end_index]
#                 feat_change2 = dir_derivative2[m][start_index:end_index]
#                 feat_sign1 = numpy.sign(feat_change1)
#                 feat_sign2 = numpy.sign(feat_change2)
#                 # true bias： 沿着原样本负方向导数，相似样本正方向导数方向扰动
#                 Dis_M = 10000000
#                 for h in range(len(feat_sign1)):
#                     if feat_sign1[h] < 0 and feat_sign2[h] > 0:
#                         # 选取扰动后马氏距离最小的属性
#                         replace_index = h + 1 + start_index
#                         pre_data = delta[m]
#                         delta[m] = replace_index - x_i[0][m]
#                         dist_m = mahalanobis_distance(delta, covariance[0])
#                         if dist_m < Dis_M:
#                             Dis_M = dist_m
#                             replace_data[m] = replace_index
#                         else:
#                             delta[m] = pre_data
#         # 对原样本添加扰动 离散属性
#         for r in range(len(replace_data)):
#             if replace_data[r] != 0 and r not in protect_index:
#                 x_i[0][r] = replace_data[r]
#         # 根据样本损失函数、相似样本损失函数符号，选择扰动的连续非敏感属性。
#         # true bias： 选择梯度符号相反的属性扰动，沿着原样本负梯度方向扰动
#         sign1 = numpy.sign(grad1[1])
#         sign2 = numpy.sign(grad2[1])
#         direction = numpy.zeros_like(x_i[1])
#         for n in range(x_i[1].shape[0]):
#             if sign1[0, n] != sign2[0, n] and x_i[1][n] != 1 and n not in protect_index:  # 连续属性，且不是保护属性
#                 if sign1[0, n] != 0:
#                     direction[n] = -1 * sign1[0, n]
#                 else:
#                     direction[n] = sign2[0, n]
#         # 连续属性扰动为0，或无离散属性扰动，返回 已搜索结果
#         if numpy.sum(direction) == 0 and numpy.sum(replace_data) == 0:
#             return data_x_index, data_x_value, data_y, len(data_y)
#         # 对原样本添加扰动 连续属性
#         perturbation = 1 * direction
#         x_i[1] = clip_numerical_data(x_i[1] + numpy.array(perturbation), low_min, up_max)
#         # 对相似样本添加扰动
#         similar_items = add_perturbation_to_similar(similar_items, replace_data, perturbation,
#                                                     protect_index, low_min, up_max)
#         # 将扰动结果添加到结果list中
#         data_x_index.append(x_i[0].tolist())
#         data_x_value.append(x_i[1].tolist())
#         data_y.append(label.tolist())
#
#     return data_x_index, data_x_value, data_y, len(data_y)
#
#
# def FalseFairGeneration(model_file, raw_file, similar_file, search_time, result_file, feature_size, protect_index,
#                         shuffle_tag=False):
#     """
#     进行准确公平性测试
#     :return:
#     """
#     custom_layers = {'AdultEmbedding': AdultEmbedding,
#                      'AutoIntTransformerBlock': AutoIntTransformerBlock}
#     model = keras.models.load_model(model_file, custom_objects=custom_layers)
#
#     if shuffle_tag:
#         label = numpy.load(raw_file[2]).reshape(-1, 1)
#         shuffle_index = numpy.arange(label.shape[0])
#         numpy.random.shuffle(shuffle_index)
#         split_index = int(label.shape[0] * 0.20)
#
#         index = numpy.load(raw_file[0])[shuffle_index][:split_index]
#         value = numpy.load(raw_file[1])[shuffle_index][:split_index]
#         label = label[shuffle_index][:split_index]
#         cov = [numpy.cov(index, rowvar=False), numpy.cov(value, rowvar=False)]
#
#         s_index = numpy.load(similar_file[0])[:, shuffle_index, :][:, : split_index, :]
#         s_value = numpy.load(similar_file[1])[:, shuffle_index, :][:, : split_index, :]
#         s_label = numpy.load(similar_file[2])[:, shuffle_index][:, : split_index]
#
#     else:
#         index = numpy.load(raw_file[0])
#         value = numpy.load(raw_file[1])
#         label = numpy.load(raw_file[2]).reshape(-1, 1)
#         cov = [numpy.cov(index, rowvar=False), numpy.cov(value, rowvar=False)]
#
#         s_index = numpy.load(similar_file[0])
#         s_value = numpy.load(similar_file[1])
#         s_label = numpy.load(similar_file[2])
#
#     dir_unit = get_embedding_vector_direction_unit(model_file, feature_size)
#
#     adv_index = []
#     adv_value = []
#     adv_label = []
#     adv_search_time = []
#
#     for i in range(label.shape[0]):
#         print("check {}th item".format(i))
#         false_bias = false_fair_generation(model, label[i], [index[i], value[i]], [s_index[:, i, :], s_value[:, i, :]],
#                                            search_time, dir_unit, protect_index, cov)
#         adv_index.extend(false_bias[0])
#         adv_value.extend(false_bias[1])
#         adv_label.extend(false_bias[2])
#         adv_search_time.append(false_bias[3])
#
#     numpy.save(result_file[0], adv_index)
#     numpy.save(result_file[1], adv_value)
#     numpy.save(result_file[2], adv_label)
#     numpy.save(result_file[3], adv_search_time)
#
#
# def false_fair_generation(model, label, item, similar_items, search_times, direction_unit, protect_index, covariance):
#     """
#     离散非敏感属性:根据样本损失函数、相似样本损失函数embedding vector导数计算方向导数.
#         false fair： 沿着原样本正方向导数，相似样本正方向导数方向扰动
#     连续非敏感属性:根据样本损失函数、相似样本损失函数符号。
#         false fair： 选择梯度符号相同的属性扰动，沿着原样本正梯度方向扰动
#     :return:
#     """
#     low_min = [0] * item[0].shape[0]
#     up_max = [1000000] * item[0].shape[0]
#     x_i = item.copy()
#     y_i = label.copy()
#     # 生成结果
#     data_x_index = []
#     data_x_value = []
#     data_y = []
#     for _ in range(search_times):
#         # 计算y原样本损失函数embedding vector导数、连续输入值导数
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i[0], x_i[1], y_cate, model)
#         # 计算相似样本损失函数embedding vector导数、连续输入值导数
#         grad2, max_similar = compute_loss_grad_MP(similar_items[0], similar_items[1], y_cate, model)
#         # 根据样本损失函数、相似样本损失函数embedding vector导数计算方向导数，选择扰动的离散非敏感属性
#         replace_data = numpy.zeros_like(x_i[1])
#         dir_derivative1 = calculate_directional_derivative(grad1[0], direction_unit[0], x_i[0])
#         dir_derivative2 = calculate_directional_derivative(grad2[0], direction_unit[0], max_similar[0])
#         delta = numpy.zeros_like(x_i[1])
#         for m in range(x_i[0].shape[0]):
#             if x_i[1][m] == 1 and m not in protect_index:  # 离散属性，且不是保护属性
#                 if m == 0:
#                     start_index = 0
#                     end_index = direction_unit[1][m]
#                 else:
#                     start_index = direction_unit[1][m - 1]
#                     end_index = direction_unit[1][m]
#                 feat_change1 = dir_derivative1[m][start_index:end_index]
#                 feat_change2 = dir_derivative2[m][start_index:end_index]
#                 feat_sign1 = numpy.sign(feat_change1)
#                 feat_sign2 = numpy.sign(feat_change2)
#                 # false fair： 沿着原样本正方向导数，相似样本正方向导数方向扰动
#                 Dis_M = 10000000
#                 for h in range(len(feat_sign1)):
#                     if feat_sign1[h] > 0 and feat_sign2[h] > 0:
#                         # 选取扰动后马氏距离最小的属性
#                         replace_index = h + 1 + start_index
#                         pre_data = delta[m]
#                         delta[m] = replace_index - x_i[0][m]
#                         dist_m = mahalanobis_distance(delta, covariance[0])
#                         if dist_m < Dis_M:
#                             Dis_M = dist_m
#                             replace_data[m] = replace_index
#                         else:
#                             delta[m] = pre_data
#         # 对原样本添加扰动 离散属性
#         for r in range(len(replace_data)):
#             if replace_data[r] != 0 and r not in protect_index:
#                 x_i[0][r] = replace_data[r]
#         # 根据样本损失函数、相似样本损失函数符号，选择扰动的连续非敏感属性。
#         # false fair： 选择梯度符号相同的属性扰动，沿着原样本正梯度方向扰动
#         sign1 = numpy.sign(grad1[1])
#         sign2 = numpy.sign(grad2[1])
#         direction = numpy.zeros_like(x_i[1])
#         for n in range(x_i[1].shape[0]):
#             if sign1[0, n] == sign2[0, n] and x_i[1][n] != 1 and n not in protect_index:
#                 if sign1[0, n] != 0:
#                     direction[n] = sign1[0, n]
#                 else:
#                     direction[n] = -1 * sign2[0, n]
#         # 连续属性扰动为0，或无离散属性扰动，返回 已搜索结果
#         if numpy.sum(direction) == 0 and numpy.sum(replace_data) == 0:
#             return data_x_index, data_x_value, data_y, len(data_y)
#         # 对原样本添加扰动 连续属性
#         perturbation = 1 * direction
#         x_i[1] = clip_numerical_data(x_i[1] + numpy.array(perturbation), low_min, up_max)
#         # 对相似样本添加扰动
#         similar_items = add_perturbation_to_similar(similar_items, replace_data, perturbation,
#                                                     protect_index, low_min, up_max)
#         # 将扰动结果添加到结果list中
#         data_x_index.append(x_i[0].tolist())
#         data_x_value.append(x_i[1].tolist())
#         data_y.append(label.tolist())
#
#     return data_x_index, data_x_value, data_y, len(data_y)
#
#
# def TrueFairGeneration(model_file, raw_file, aug_file, S_time, R_raw_file, R_aug_file, F_size, P_index):
#     """
#     进行准确公平性测试
#     :return:
#     """
#     custom_layers = {'AdultEmbedding': AdultEmbedding,
#                      'AutoIntTransformerBlock': AutoIntTransformerBlock}
#     model = keras.models.load_model(model_file, custom_objects=custom_layers)
#
#     raw_index = numpy.squeeze(numpy.load(raw_file[0]))
#     raw_value = numpy.squeeze(numpy.load(raw_file[1]))
#     raw_label = numpy.squeeze(numpy.load(raw_file[2])).reshape(-1, 1)
#     D_statistic = [numpy.cov(raw_index, rowvar=False), numpy.cov(raw_value, rowvar=False)]
#
#     aug_index = numpy.squeeze(numpy.load(aug_file[0]))
#     aug_value = numpy.squeeze(numpy.load(aug_file[1]))
#     aug_label = numpy.squeeze(numpy.load(aug_file[2]))
#
#     dir_unit = get_embedding_vector_direction_unit(model_file, F_size)
#
#     adv_index = []
#     adv_value = []
#     adv_label = []
#     similar_adv_index = []
#     similar_adv_value = []
#     adv_search_time = []
#     for i in range(raw_label.shape[0]):
#         label = raw_label[i]
#         item = [raw_index[i], raw_value[i]]
#         similar_items = [aug_index[:, i, :], aug_value[:, i, :]]
#         true_fair = true_fair_generation(model, label, item, similar_items, S_time, dir_unit, P_index, D_statistic)
#         adv_index.extend(true_fair[0])
#         adv_value.extend(true_fair[1])
#         adv_label.extend(true_fair[2])
#         similar_adv_index.extend(true_fair[3])
#         similar_adv_value.extend(true_fair[4])
#         adv_search_time.append(true_fair[5])
#
#     numpy.save(R_raw_file[0], adv_index)
#     numpy.save(R_raw_file[1], adv_value)
#     numpy.save(R_raw_file[2], adv_label)
#     numpy.save(R_aug_file[0], similar_adv_index)
#     numpy.save(R_aug_file[1], similar_adv_value)
#     numpy.save(R_aug_file[2], adv_search_time)
#
#
# def true_fair_generation(model, label, item, similar_items, S_times, dir_unit, P_index, D_statistic):
#     """
#     离散非敏感属性:根据样本损失函数、相似样本损失函数embedding vector导数计算方向导数.
#         true fair： 沿着原样本负方向导数，相似样本负方向导数方向扰动
#     连续非敏感属性:根据样本损失函数、相似样本损失函数符号。
#         true fair： 选择梯度符号相同的属性扰动，沿着原样本负梯度方向扰动
#     :return:
#     """
#     low_min = [0] * item[0].shape[0]
#     up_max = [1000000] * item[0].shape[0]
#     x_i = item.copy()
#     y_i = label.copy()
#     # 生成结果
#     data_x_index = []
#     data_x_value = []
#     data_y = []
#     similar_data_x_index = []
#     similar_data_x_value = []
#     for _ in range(S_times):
#         # 计算y原样本损失函数embedding vector导数、连续输入值导数
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i[0], x_i[1], y_cate, model)
#         # 计算相似样本损失函数embedding vector导数、连续输入值导数
#         grad2, max_similar = compute_loss_grad_MP(similar_items[0], similar_items[1], y_cate, model)
#         # 根据样本损失函数、相似样本损失函数embedding vector导数计算方向导数，选择扰动的离散非敏感属性
#         replace_data = numpy.zeros_like(x_i[1])
#         dir_derivative1 = calculate_directional_derivative(grad1[0], dir_unit[0], x_i[0])
#         dir_derivative2 = calculate_directional_derivative(grad2[0], dir_unit[0], max_similar[0])
#         delta = numpy.zeros_like(x_i[1])
#         for m in range(x_i[0].shape[0]):
#             if x_i[1][m] == 1 and m not in P_index:  # 离散属性，且不是保护属性
#                 if m == 0:
#                     start_index = 0
#                     end_index = dir_unit[1][m]
#                 else:
#                     start_index = dir_unit[1][m - 1]
#                     end_index = dir_unit[1][m]
#                 feat_change1 = dir_derivative1[m][start_index:end_index]
#                 feat_change2 = dir_derivative2[m][start_index:end_index]
#                 feat_sign1 = numpy.sign(feat_change1)
#                 feat_sign2 = numpy.sign(feat_change2)
#                 # true fair： 沿着原样本负方向导数，相似样本负方向导数方向扰动
#                 Dis_M = 10000000
#                 replace_value = 100000
#                 for h in range(len(feat_sign1)):
#                     if feat_sign1[h] < 0 and feat_sign2[h] < 0:
#                         # 选取扰动后马氏距离最小的属性
#                         replace_index = h + 1 + start_index
#                         pre_data = delta[m]
#                         delta[m] = replace_index - x_i[0][m]
#                         dist_m = mahalanobis_distance(delta, D_statistic[0])
#                         if dist_m < Dis_M:
#                             Dis_M = dist_m
#                             replace_data[m] = replace_index
#                         else:
#                             delta[m] = pre_data
#                         # dir_derivative_value = abs(feat_change1[h]) + abs(feat_change2[h])
#                         # if dir_derivative_value < replace_value:
#                         #     replace_index = h+ 1 + start_index
#                         #     replace_value = dir_derivative_value
#                         #     replace_data[m] = replace_index
#         # adjust_attributes(D_statistic[0], D_statistic[1], replace_data)
#         # 对原样本添加扰动 离散属性
#         for r in range(len(replace_data)):
#             if replace_data[r] != 0 and r not in P_index:
#                 x_i[0][r] = replace_data[r]
#         # 根据样本损失函数、相似样本损失函数符号，选择扰动的连续非敏感属性。
#         # true fair： 选择梯度符号相同的属性扰动，沿着原样本负梯度方向扰动
#         sign1 = numpy.sign(grad1[1])
#         sign2 = numpy.sign(grad2[1])
#         direction = numpy.zeros_like(x_i[1])
#         for n in range(x_i[1].shape[0]):
#             if sign1[0, n] == sign2[0, n] and x_i[1][n] != 1 and n not in P_index:
#                 if sign1[0, n] != 0:
#                     direction[n] = -1 * sign1[0, n]
#                 else:
#                     direction[n] = sign2[0, n]
#             if sign1[0, n] == 0:
#                 direction[n] = sign2[0, n]
#         # 对原样本添加扰动 连续属性
#         perturbation = 1 * direction
#         x_i[1] = clip_numerical_data(x_i[1] + numpy.array(perturbation), low_min, up_max)
#         # 对相似样本添加扰动
#         similar_items = add_perturbation_to_similar(similar_items, replace_data, perturbation, P_index, low_min, up_max)
#         # 连续属性扰动为0，或无离散属性扰动，返回 已搜索结果
#         if numpy.sum(direction) == 0 and numpy.sum(replace_data) == 0:
#             return data_x_index, data_x_value, data_y, similar_data_x_index, similar_data_x_value, len(data_y)
#         # 将扰动结果添加到结果list中
#         data_x_index.append(x_i[0].tolist())
#         data_x_value.append(x_i[1].tolist())
#         data_y.append(label.tolist())
#         similar_data_x_index.append(similar_items[0].tolist())
#         similar_data_x_value.append(similar_items[1].tolist())
#
#     return data_x_index, data_x_value, data_y, similar_data_x_index, similar_data_x_value, len(data_y)

# def adjust_attributes(mean_vector, cov_matrix, replace_data):
#     """
#     调整属性值以维持相关性并最小化与原样本的距离
#     参数：
#     - statistic_mean: numpy.ndarray，均值向量
#     - statistic_covariance: numpy.ndarray，协方差矩阵
#     - replace_data: 扰动值
#     - x_replace: numpy.ndarray，属性 A 的新值
#     - indices_replace: list，属性 A 的索引
#     - indices_none_replace: list，属性 B 的索引
#     返回：
#     - x_B_new: numpy.ndarray，调整后的属性 B 的值
#     """
#     non_zero = numpy.any(cov_matrix, axis=1)
#     cov_matrix = cov_matrix[non_zero]
#     cov_matrix = cov_matrix[:, non_zero]
#     replace_data = replace_data[non_zero]
#     mean_vector = mean_vector[non_zero]
#
#     x_replace = []
#     indices_replace = []
#     indices_none_replace = []
#     for r in range(len(replace_data)):
#         if replace_data[r] != 0:
#             indices_replace.append(r)
#             x_replace.append(replace_data[r])
#         else:
#             indices_none_replace.append(r)
#
#     # 分块均值向量和协方差矩阵
#     mean_A = mean_vector[indices_replace]
#     mean_B = mean_vector[indices_none_replace]
#
#     covariance_AA = cov_matrix[numpy.ix_(indices_replace, indices_replace)]
#     covariance_AB = cov_matrix[numpy.ix_(indices_replace, indices_none_replace)]
#     covariance_BA = cov_matrix[numpy.ix_(indices_none_replace, indices_replace)]
#     covariance_BB = cov_matrix[numpy.ix_(indices_none_replace, indices_none_replace)]
#
#     # 计算条件均值
#     covariance_AA_inv = numpy.linalg.inv(covariance_AA)
#     mean_B_given_A = mean_B + covariance_BA @ covariance_AA_inv @ (x_replace - mean_A)
#
#     result_data = numpy.zeros_like(replace_data)
#
#     for r in range(len(indices_replace)):
#         result_data[indices_replace[r]] = x_replace[r]
#     for r in range(len(indices_none_replace)):
#         result_data[indices_none_replace[r]] = mean_B_given_A[r]
#
#     total_result_data = []
#     h = 0
#     for t in non_zero:
#         if t:
#             total_result_data.append(result_data[h])
#             h += 1
#         else:
#             total_result_data.append(0)
#
#     return total_result_data
#
#
# def TrueBiasedGenerationProb(M_file, raw_file, aug_file, S_time, R_raw_file, R_aug_file, F_size, P_index):
#     """
#     进行 true bias 准确公平性测试
#     :return:
#     """
#     custom_layers = {'AdultEmbedding': AdultEmbedding,
#                      'AutoIntTransformerBlock': AutoIntTransformerBlock}
#     model = keras.models.load_model(M_file, custom_objects=custom_layers)
#
#     raw_index = numpy.squeeze(numpy.load(raw_file[0]))
#     raw_value = numpy.squeeze(numpy.load(raw_file[1]))
#     raw_label = numpy.squeeze(numpy.load(raw_file[2])).reshape(-1, 1)
#     D_statistic = [numpy.mean(raw_index, axis=0), numpy.cov(raw_index, rowvar=False)]
#
#     aug_index = numpy.squeeze(numpy.load(aug_file[0]))
#     aug_value = numpy.squeeze(numpy.load(aug_file[1]))
#     aug_label = numpy.squeeze(numpy.load(aug_file[2]))
#     # 各embedding vector到其余vector的单位方向向量
#     dir_unit = get_embedding_vector_direction_unit(M_file, F_size)
#
#     adv_index = []
#     adv_value = []
#     adv_label = []
#     similar_adv_index = []
#     similar_adv_value = []
#     adv_search_time = []
#     for i in range(5000):
#         # 原样本标签
#         label = raw_label[i]
#         # 原样本
#         item = [raw_index[i], raw_value[i]]
#         # 相似样本
#         similar_items = [aug_index[:, i, :], aug_value[:, i, :]]
#         # 运行 true bias adversarial attack
#         true_bias = true_bias_generation_prob(model, label, item, similar_items, S_time, dir_unit, P_index, D_statistic)
#         adv_index.extend(true_bias[0])
#         adv_value.extend(true_bias[1])
#         adv_label.extend(true_bias[2])
#         similar_adv_index.extend(true_bias[3])
#         similar_adv_value.extend(true_bias[4])
#         adv_search_time.append(true_bias[5])
#
#     numpy.save(R_raw_file[0], adv_index)
#     numpy.save(R_raw_file[1], adv_value)
#     numpy.save(R_raw_file[2], adv_label)
#     numpy.save(R_aug_file[0], similar_adv_index)
#     numpy.save(R_aug_file[1], similar_adv_value)
#     numpy.save(R_aug_file[2], adv_search_time)
#
#
# def true_bias_generation_prob(model, label, item, similar_items, S_times, dir_unit, P_index, D_statistic):
#     """
#     离散非敏感属性:根据样本损失函数、相似样本损失函数embedding vector导数计算方向导数.
#         true bias： 沿着原样本负方向导数，相似样本正方向导数方向扰动
#     连续非敏感属性:根据样本损失函数、相似样本损失函数符号。
#         true bias： 选择梯度符号相反的属性扰动，沿着原样本负梯度方向扰动
#     :return:
#     """
#     low_min = [0] * item[0].shape[0]
#     up_max = [1000000] * item[0].shape[0]
#     x_i = item.copy()
#     y_i = label.copy()
#     # 生成结果
#     data_x_index = []
#     data_x_value = []
#     data_y = []
#     similar_data_x_index = []
#     similar_data_x_value = []
#     for _ in range(S_times):
#         # 计算y原样本损失函数embedding vector导数、连续输入值导数
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i[0], x_i[1], y_cate, model)
#         # 计算相似样本损失函数embedding vector导数、连续输入值导数
#         grad2, max_similar = compute_loss_grad_MP(similar_items[0], similar_items[1], y_cate, model)
#         # 根据样本损失函数、相似样本损失函数embedding vector导数计算方向导数，选择扰动的离散非敏感属性
#         replace_data = numpy.zeros_like(x_i[1])
#         dir_derivative1 = calculate_directional_derivative(grad1[0], dir_unit[0], x_i[0])
#         dir_derivative2 = calculate_directional_derivative(grad2[0], dir_unit[0], max_similar[0])
#         for m in range(x_i[0].shape[0]):
#             if x_i[1][m] == 1 and m not in P_index:  # 离散属性，且不是保护属性
#                 if m == 0:
#                     start_index = 0
#                     end_index = dir_unit[1][m]
#                 else:
#                     start_index = dir_unit[1][m - 1]
#                     end_index = dir_unit[1][m]
#                 # 截取恰当的单位方向向量（该属性内的所有取值方向向量）
#                 feat_change1 = dir_derivative1[m][start_index:end_index]
#                 feat_change2 = dir_derivative2[m][start_index:end_index]
#                 feat_sign1 = numpy.sign(feat_change1)
#                 feat_sign2 = numpy.sign(feat_change2)
#                 # true bias： 沿着原样本负方向导数，相似样本正方向导数方向扰动
#                 replace_value = 100000
#                 for h in range(len(feat_sign1)):
#                     if feat_sign1[h] < 0 and feat_sign2[h] > 0:
#                         dir_derivative_value = abs(feat_change1[h]) + abs(feat_change2[h])
#                         if dir_derivative_value < replace_value:
#                             replace_index = h + 1 + start_index
#                             replace_value = dir_derivative_value
#                             replace_data[m] = replace_index
#         # 对原样本添加扰动 离散属性
#         replace_data = adjust_attributes(D_statistic[0], D_statistic[1], replace_data)
#         for r in range(len(replace_data)):
#             if replace_data[r] != 0 and r not in P_index:
#                 x_i[0][r] = replace_data[r]
#         # 根据样本损失函数、相似样本损失函数符号，选择扰动的连续非敏感属性。
#         # true bias： 选择梯度符号相反的属性扰动，沿着原样本负梯度方向扰动
#         sign1 = numpy.sign(grad1[1])
#         sign2 = numpy.sign(grad2[1])
#         direction = numpy.zeros_like(x_i[1])
#         for n in range(x_i[1].shape[0]):
#             if sign1[0, n] != sign2[0, n] and x_i[1][n] != 1 and n not in P_index:  # 连续属性，且不是保护属性
#                 if sign1[0, n] != 0:
#                     direction[n] = -1 * sign1[0, n]
#                 else:
#                     direction[n] = sign2[0, n]
#             if sign1[0, n] == 0:
#                 direction[n] = sign2[0, n]
#         # 对原样本添加扰动 连续属性
#         perturbation = 1 * direction
#         x_i[1] = clip_numerical_data(x_i[1] + numpy.array(perturbation), low_min, up_max)
#         # 对相似样本添加扰动
#         similar_items = add_perturbation_to_similar(similar_items, replace_data, perturbation, P_index, low_min, up_max)
#         # 连续属性扰动为0，或无离散属性扰动，返回 已搜索结果
#         if numpy.sum(direction) == 0 and numpy.sum(replace_data) == 0:
#             return data_x_index, data_x_value, data_y, similar_data_x_index, similar_data_x_value, len(data_y)
#         # 将扰动结果添加到结果list中
#         data_x_index.append(x_i[0].tolist())
#         data_x_value.append(x_i[1].tolist())
#         data_y.append(label.tolist())
#         similar_data_x_index.append(similar_items[0].tolist())
#         similar_data_x_value.append(similar_items[1].tolist())
#
#     return data_x_index, data_x_value, data_y, similar_data_x_index, similar_data_x_value, len(data_y)
#
#
# def FalseFairGenerationProb(model_file, raw_file, aug_file, S_time, R_raw_file, R_aug_file, F_size, P_index):
#     """
#     进行准确公平性测试
#     :return:
#     """
#     custom_layers = {'AdultEmbedding': AdultEmbedding,
#                      'AutoIntTransformerBlock': AutoIntTransformerBlock}
#     model = keras.models.load_model(model_file, custom_objects=custom_layers)
#
#     raw_index = numpy.squeeze(numpy.load(raw_file[0]))
#     raw_value = numpy.squeeze(numpy.load(raw_file[1]))
#     raw_label = numpy.squeeze(numpy.load(raw_file[2])).reshape(-1, 1)
#     # raw_input = [raw_index, raw_value]
#     D_sta = [numpy.mean(raw_index, axis=0), numpy.cov(raw_index, rowvar=False)]
#
#     aug_index = numpy.squeeze(numpy.load(aug_file[0]))
#     aug_value = numpy.squeeze(numpy.load(aug_file[1]))
#     aug_label = numpy.squeeze(numpy.load(aug_file[2]))
#
#     dir_unit = get_embedding_vector_direction_unit(model_file, F_size)
#
#     adv_index = []
#     adv_value = []
#     adv_label = []
#     similar_adv_index = []
#     similar_adv_value = []
#     adv_search_time = []
#     for i in range(5000):
#         label = raw_label[i]
#         item = [raw_index[i], raw_value[i]]
#         similar_items = [aug_index[:, i, :], aug_value[:, i, :]]
#         false_fair = false_fair_generation_prob(model, label, item, similar_items, S_time, dir_unit, P_index, D_sta)
#         adv_index.extend(false_fair[0])
#         adv_value.extend(false_fair[1])
#         adv_label.extend(false_fair[2])
#         similar_adv_index.extend(false_fair[3])
#         similar_adv_value.extend(false_fair[4])
#         adv_search_time.append(false_fair[5])
#
#     numpy.save(R_raw_file[0], adv_index)
#     numpy.save(R_raw_file[1], adv_value)
#     numpy.save(R_raw_file[2], adv_label)
#     numpy.save(R_aug_file[0], similar_adv_index)
#     numpy.save(R_aug_file[1], similar_adv_value)
#     numpy.save(R_aug_file[2], adv_search_time)
#
#
# def false_fair_generation_prob(model, label, item, similar_items, S_times, direction_unit, P_index, D_statistic):
#     """
#     离散非敏感属性:根据样本损失函数、相似样本损失函数embedding vector导数计算方向导数.
#         false fair： 沿着原样本正方向导数，相似样本正方向导数方向扰动
#     连续非敏感属性:根据样本损失函数、相似样本损失函数符号。
#         false fair： 选择梯度符号相同的属性扰动，沿着原样本正梯度方向扰动
#     :return:
#     """
#     low_min = [0] * item[0].shape[0]
#     up_max = [1000000] * item[0].shape[0]
#     x_i = item.copy()
#     y_i = label.copy()
#     # 生成结果
#     data_x_index = []
#     data_x_value = []
#     data_y = []
#     similar_data_x_index = []
#     similar_data_x_value = []
#     for _ in range(S_times):
#         # 计算y原样本损失函数embedding vector导数、连续输入值导数
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i[0], x_i[1], y_cate, model)
#         # 计算相似样本损失函数embedding vector导数、连续输入值导数
#         grad2, max_similar = compute_loss_grad_MP(similar_items[0], similar_items[1], y_cate, model)
#         # 根据样本损失函数、相似样本损失函数embedding vector导数计算方向导数，选择扰动的离散非敏感属性
#         replace_data = numpy.zeros_like(x_i[1])
#         dir_derivative1 = calculate_directional_derivative(grad1[0], direction_unit[0], x_i[0])
#         dir_derivative2 = calculate_directional_derivative(grad2[0], direction_unit[0], max_similar[0])
#         for m in range(x_i[0].shape[0]):
#             if x_i[1][m] == 1 and m not in P_index:  # 离散属性，且不是保护属性
#                 if m == 0:
#                     start_index = 0
#                     end_index = direction_unit[1][m]
#                 else:
#                     start_index = direction_unit[1][m - 1]
#                     end_index = direction_unit[1][m]
#                 feat_change1 = dir_derivative1[m][start_index:end_index]
#                 feat_change2 = dir_derivative2[m][start_index:end_index]
#                 feat_sign1 = numpy.sign(feat_change1)
#                 feat_sign2 = numpy.sign(feat_change2)
#                 # false fair： 沿着原样本正方向导数，相似样本正方向导数方向扰动
#                 replace_value = 100000
#                 for h in range(len(feat_sign1)):
#                     if feat_sign1[h] > 0 and feat_sign2[h] > 0:
#                         dir_derivative_value = abs(feat_change1[h]) + abs(feat_change2[h])
#                         if dir_derivative_value < replace_value:
#                             replace_index = h + 1 + start_index
#                             replace_value = dir_derivative_value
#                             replace_data[m] = replace_index
#         # 对原样本添加扰动 离散属性
#         replace_data = adjust_attributes(D_statistic[0], D_statistic[1], replace_data)
#         for r in range(len(replace_data)):
#             if replace_data[r] != 0 and r not in P_index:
#                 x_i[0][r] = replace_data[r]
#         # 根据样本损失函数、相似样本损失函数符号，选择扰动的连续非敏感属性。
#         # false fair： 选择梯度符号相同的属性扰动，沿着原样本正梯度方向扰动
#         sign1 = numpy.sign(grad1[1])
#         sign2 = numpy.sign(grad2[1])
#         direction = numpy.zeros_like(x_i[1])
#         for n in range(x_i[1].shape[0]):
#             if sign1[0, n] == sign2[0, n] and x_i[1][n] != 1 and n not in P_index:
#                 if sign1[0, n] != 0:
#                     direction[n] = sign1[0, n]
#                 else:
#                     direction[n] = -1 * sign2[0, n]
#             if sign1[0, n] == 0:
#                 direction[n] = sign2[0, n]
#         # 对原样本添加扰动 连续属性
#         perturbation = 1 * direction
#         x_i[1] = clip_numerical_data(x_i[1] + numpy.array(perturbation), low_min, up_max)
#         # 对相似样本添加扰动
#         similar_items = add_perturbation_to_similar(similar_items, replace_data, perturbation, P_index, low_min, up_max)
#         # 连续属性扰动为0，或无离散属性扰动，返回 已搜索结果
#         if numpy.sum(direction) == 0 and numpy.sum(replace_data) == 0:
#             return data_x_index, data_x_value, data_y, similar_data_x_index, similar_data_x_value, len(data_y)
#         # 将扰动结果添加到结果list中
#         data_x_index.append(x_i[0].tolist())
#         data_x_value.append(x_i[1].tolist())
#         data_y.append(label.tolist())
#         similar_data_x_index.append(similar_items[0].tolist())
#         similar_data_x_value.append(similar_items[1].tolist())
#
#     return data_x_index, data_x_value, data_y, similar_data_x_index, similar_data_x_value, len(data_y)
#
#
# def FalseBiasedGenerationProb(model_file, raw_file, aug_file, S_time, R_raw_file, R_aug_file, F_size, P_index):
#     """
#     进行准确公平性测试
#     :return:
#     """
#     custom_layers = {'AdultEmbedding': AdultEmbedding,
#                      'AutoIntTransformerBlock': AutoIntTransformerBlock}
#     model = keras.models.load_model(model_file, custom_objects=custom_layers)
#
#     raw_index = numpy.squeeze(numpy.load(raw_file[0]))
#     raw_value = numpy.squeeze(numpy.load(raw_file[1]))
#     raw_label = numpy.squeeze(numpy.load(raw_file[2])).reshape(-1, 1)
#     D_sta = [numpy.mean(raw_index, axis=0), numpy.cov(raw_index, rowvar=False)]
#
#     aug_index = numpy.squeeze(numpy.load(aug_file[0]))
#     aug_value = numpy.squeeze(numpy.load(aug_file[1]))
#     aug_label = numpy.squeeze(numpy.load(aug_file[2]))
#
#     dir_unit = get_embedding_vector_direction_unit(model_file, F_size)
#
#     adv_index = []
#     adv_value = []
#     adv_label = []
#     similar_adv_index = []
#     similar_adv_value = []
#     adv_search_time = []
#     for i in range(5000):
#         label = raw_label[i]
#         item = [raw_index[i], raw_value[i]]
#         similar_items = [aug_index[:, i, :], aug_value[:, i, :]]
#         false_bias = false_bias_generation_prob(model, label, item, similar_items, S_time, dir_unit, P_index, D_sta)
#         adv_index.extend(false_bias[0])
#         adv_value.extend(false_bias[1])
#         adv_label.extend(false_bias[2])
#         similar_adv_index.extend(false_bias[3])
#         similar_adv_value.extend(false_bias[4])
#         adv_search_time.append(false_bias[5])
#
#     numpy.save(R_raw_file[0], adv_index)
#     numpy.save(R_raw_file[1], adv_value)
#     numpy.save(R_raw_file[2], adv_label)
#     numpy.save(R_aug_file[0], similar_adv_index)
#     numpy.save(R_aug_file[1], similar_adv_value)
#     numpy.save(R_aug_file[2], adv_search_time)
#
#
# def false_bias_generation_prob(model, label, item, similar_items, S_times, dir_unit, P_index, D_statistic):
#     """
#     离散非敏感属性:根据样本损失函数、相似样本损失函数embedding vector导数计算方向导数.
#         false bias： 沿着原样本正方向导数，相似样本负方向导数方向扰动
#     连续非敏感属性:根据样本损失函数、相似样本损失函数符号。
#         false bias： 选择梯度符号相反的属性扰动，沿着原样本正梯度方向扰动
#     :return:
#     """
#     low_min = [0] * item[0].shape[0]
#     up_max = [1000000] * item[0].shape[0]
#     x_i = item.copy()
#     y_i = label.copy()
#     # 生成结果
#     data_x_index = []
#     data_x_value = []
#     data_y = []
#     similar_data_x_index = []
#     similar_data_x_value = []
#     for _ in range(S_times):
#         # 计算y原样本损失函数embedding vector导数、连续输入值导数
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i[0], x_i[1], y_cate, model)
#         # 计算相似样本损失函数embedding vector导数、连续输入值导数
#         grad2, max_similar = compute_loss_grad_MP(similar_items[0], similar_items[1], y_cate, model)
#         # 根据样本损失函数、相似样本损失函数embedding vector导数计算方向导数，选择扰动的离散非敏感属性
#         replace_data = numpy.zeros_like(x_i[1])
#         dir_derivative1 = calculate_directional_derivative(grad1[0], dir_unit[0], x_i[0])
#         dir_derivative2 = calculate_directional_derivative(grad2[0], dir_unit[0], max_similar[0])
#         for m in range(x_i[0].shape[0]):
#             if x_i[1][m] == 1 and m not in P_index:  # 离散属性，且不是保护属性
#                 if m == 0:
#                     start_index = 0
#                     end_index = dir_unit[1][m]
#                 else:
#                     start_index = dir_unit[1][m - 1]
#                     end_index = dir_unit[1][m]
#                 feat_change1 = dir_derivative1[m][start_index:end_index]
#                 feat_change2 = dir_derivative2[m][start_index:end_index]
#                 feat_sign1 = numpy.sign(feat_change1)
#                 feat_sign2 = numpy.sign(feat_change2)
#                 # false bias： 沿着原样本正方向导数，相似样本负方向导数方向扰动
#                 replace_value = 100000
#                 for h in range(len(feat_sign1)):
#                     if feat_sign1[h] > 0 and feat_sign2[h] < 0:
#                         dir_derivative_value = abs(feat_change1[h]) + abs(feat_change2[h])
#                         if dir_derivative_value < replace_value:
#                             replace_index = h + 1 + start_index
#                             replace_value = dir_derivative_value
#                             replace_data[m] = replace_index
#         # 对原样本添加扰动 离散属性
#         replace_data = adjust_attributes(D_statistic[0], D_statistic[1], replace_data)
#         for r in range(len(replace_data)):
#             if replace_data[r] != 0 and r not in P_index:
#                 x_i[0][r] = replace_data[r]
#         # 根据样本损失函数、相似样本损失函数符号，选择扰动的连续非敏感属性。
#         # false bias： 选择梯度符号相反的属性扰动，沿着原样本正梯度方向扰动
#         sign1 = numpy.sign(grad1[1])
#         sign2 = numpy.sign(grad2[1])
#         direction = numpy.zeros_like(x_i[1])
#         for n in range(x_i[1].shape[0]):
#             if sign1[0, n] != sign2[0, n] and x_i[1][n] != 1 and n not in P_index:
#                 if sign1[0, n] != 0:
#                     direction[n] = sign1[0, n]
#                 else:
#                     direction[n] = -1 * sign2[0, n]
#             if sign1[0, n] == 0:
#                 direction[n] = sign2[0, n]
#         # 对原样本添加扰动 连续属性
#         perturbation = 1 * direction
#         x_i[1] = clip_numerical_data(x_i[1] + numpy.array(perturbation), low_min, up_max)
#         # 对相似样本添加扰动
#         similar_items = add_perturbation_to_similar(similar_items, replace_data, perturbation, P_index, low_min, up_max)
#         # 连续属性扰动为0，或无离散属性扰动，返回 已搜索结果
#         if numpy.sum(direction) == 0 and numpy.sum(replace_data) == 0:
#             return data_x_index, data_x_value, data_y, similar_data_x_index, similar_data_x_value, len(data_y)
#         # 将扰动结果添加到结果list中
#         data_x_index.append(x_i[0].tolist())
#         data_x_value.append(x_i[1].tolist())
#         data_y.append(label.tolist())
#         similar_data_x_index.append(similar_items[0].tolist())
#         similar_data_x_value.append(similar_items[1].tolist())
#
#     return data_x_index, data_x_value, data_y, similar_data_x_index, similar_data_x_value, len(data_y)

# def get_NLP_data(x, similar_x, label, pre, similar_pre):
#     """
#     获取NLP结果
#     :return:
#     """
#     if label[0] == pre[0]:
#         for p in similar_pre:
#             if p != pre[0]:
#                 raw_data = restore_NLP_Adult_data(x[0], x[1], label, pre)
#                 similar_data = []
#                 for i in range(similar_x[0].shape[0]):
#                     similar_data.append(restore_NLP_Adult_data(similar_x[0][i], similar_x[1][i], label, similar_pre[i]))
#                 return "True Bias", raw_data, similar_data
#         raw_data = restore_NLP_Adult_data(x[0], x[1], label, pre)
#         similar_data = []
#         for i in range(similar_x[0].shape[0]):
#             similar_data.append(restore_NLP_Adult_data(similar_x[0][i], similar_x[1][i], label, similar_pre[i]))
#         return "True Fair", raw_data, similar_data
#     else:
#         for p in similar_pre:
#             if p != pre[0]:
#                 raw_data = restore_NLP_Adult_data(x[0], x[1], label, pre)
#                 similar_data = []
#                 for i in range(similar_x[0].shape[0]):
#                     similar_data.append(restore_NLP_Adult_data(similar_x[0][i], similar_x[1][i], label, similar_pre[i]))
#                 return "False Bias", raw_data, similar_data
#         raw_data = restore_NLP_Adult_data(x[0], x[1], label, pre)
#         similar_data = []
#         for i in range(similar_x[0].shape[0]):
#             similar_data.append(restore_NLP_Adult_data(similar_x[0][i], similar_x[1][i], label, similar_pre[i]))
#         return "False Fair", raw_data, similar_data

# def generate_similar_items(item, dataset, protected_attr):
#     """
#     生成样本的相似样本
#     :return:
#     """
#     # 生成相似样本
#     if dataset == "adult":
#         similar_items = data_augmentation_adult_item(item, protected_attr)
#     elif dataset == "compas":
#         similar_items = data_augmentation_compas_item(item, protected_attr)
#     elif dataset == "credit":
#         similar_items = data_augmentation_credit_item(item, protected_attr)
#     else:
#         similar_items = data_augmentation_bank_item(item, protected_attr)
#     return similar_items
#
#
# def project_adversarial_data(test_file, adv_file, project_radius, projected_file):
#     """
#     对生成的对抗样本进行投影
#     :return:
#     """
#     test_data = numpy.load(test_file)
#     adv_data = numpy.load(adv_file, allow_pickle=True)
#     adv_result = []
#     for i in range(test_data.shape[0]):
#         adv_items = adv_data[i]
#         projected_data = []
#         for j in range(adv_items.shape[0]):
#             perturbations = test_data[i] - adv_items[j]
#             projected_perturbation = tensorflow.sign(perturbations) * tensorflow.minimum(
#                 tensorflow.math.abs(perturbations), project_radius)
#             projected_adv = test_data[i] + numpy.array(projected_perturbation)
#             projected_data.append(projected_adv.tolist())
#         adv_result.append(projected_adv)
#
#     unique_bias_data, unique_index = numpy.unique(adv_result, return_index=True, axis=0)
#     numpy.save(projected_file, unique_bias_data)
#
#
# def TF(model, test_item, similar_items, search_times, extent):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
#     :return:
#     """
#     # 保存预测结果错误或歧视的位置
#     x, y = numpy.split(test_item, [-1, ], axis=1)
#     x_i = x.copy().reshape(1, -1)
#     y_i = y.copy()
#     for _ in range(search_times):
#         # 计算损失函数导数符号
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i, y_cate, model)
#         sign1 = numpy.sign(grad1)
#         # 计算相似样本损失函数，及最远相似样本
#         grad2, max_similar = compute_loss_grad_MP(similar_items, y_cate, model)
#         sign2 = numpy.sign(grad2)
#         # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#         direction = numpy.zeros_like(x_i)
#         for n in range(x.shape[1]):
#             if sign1[n] == sign2[n]:
#                 direction[0, n] = -sign1[n]
#             if sign1[n] == 0:
#                 direction[0, n] = -sign2[n]
#         #  扰动所选择属性
#         perturbation = extent * direction
#         perturbation = tensorflow.sign(perturbation) * tensorflow.minimum(tensorflow.math.abs(perturbation), 0.05)
#         perturbation = numpy.array(perturbation)
#         generated_x = x_i + perturbation
#         x_i = generated_x
#         similar_x_i = add_perturbation_to_similar_items(similar_items, perturbation)
#         AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
#                                         numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i, dist=0, K=0)
#         if AF:
#             return numpy.concatenate((x_i, y_i.reshape(1, -1)), axis=1)
#     return numpy.concatenate((x_i, y_i.reshape(1, -1)), axis=1)
#
#
# def TF_attack(model, test_item, dataset, protected_attr, search_times, extent):
#     """
#     对样本进行FF搜索，返回搜索结果，相似样本，以及该样本的公平混淆矩阵分类结果
#     :return:
#     """
#     similar_items = generate_similar_items(test_item[0, :-1].reshape(1, -1), dataset, protected_attr)
#     return TF(model, test_item, similar_items, search_times, extent)


# def TB_attack(model, test_item, dataset, protected_attr, search_times, extent):
#     """
#     对样本进行TB搜索，返回搜索结果，相似样本，以及该样本的公平混淆矩阵分类结果
#     :return:
#     """
#     similar_items = generate_similar_items(test_item[0, :-1].reshape(1, -1), dataset, protected_attr)
#     return true_bias_adversarial(model, test_item, similar_items, search_times, extent)


# def FF(model, test_item, similar_items, search_times, extent):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
#     :return:
#     """
#     # 保存预测结果错误或歧视的位置
#     x, y = numpy.split(test_item, [-1, ], axis=1)
#     x_i = x.copy().reshape(1, -1)
#     y_i = y.copy()
#     for _ in range(search_times):
#         # 计算损失函数导数符号
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i, y_cate, model)
#         sign1 = numpy.sign(grad1)
#         # 计算相似样本损失函数，及最远相似样本
#         grad2, max_similar = compute_loss_grad_MP(similar_items, y_cate, model)
#         sign2 = numpy.sign(grad2)
#         # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#         direction = numpy.zeros_like(x_i)
#         for n in range(x.shape[1]):
#             # if n not in [10, 11, 12] and sign1[n] == sign2[n]:
#             if sign1[n] == sign2[n]:
#                 direction[0, n] = sign1[n]
#
#         #  扰动所选择属性
#         perturbation = extent * direction
#         perturbation = tensorflow.sign(perturbation) * tensorflow.minimum(tensorflow.math.abs(perturbation), 0.05)
#         perturbation = numpy.array(perturbation)
#         generated_x = x_i + perturbation
#         x_i = generated_x
#         similar_x_i = add_perturbation_to_similar_items(similar_items, perturbation)
#         AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
#                                         numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i, dist=0, K=0)
#         if AF_C[1]:
#             return numpy.concatenate((x_i, y_i.reshape(1, -1)), axis=1)
#     return numpy.concatenate((x_i, y_i.reshape(1, -1)), axis=1)
#
#
# def FF_attack(model, test_item, dataset, protected_attr, search_times, extent):
#     """
#     对样本进行FF搜索，返回搜索结果，相似样本，以及该样本的公平混淆矩阵分类结果
#     :return:
#     """
#     similar_items = generate_similar_items(test_item[0, :-1].reshape(1, -1), dataset, protected_attr)
#     return FF(model, test_item, similar_items, search_times, extent)
#
#
# def FB(model, test_item, similar_items, search_times, extent):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
#     :return:
#     """
#     # 保存预测结果错误或歧视的位置
#     x, y = numpy.split(test_item, [-1, ], axis=1)
#     x_i = x.copy().reshape(1, -1)
#     y_i = y.copy()
#     for _ in range(search_times):
#         # 计算损失函数导数符号
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i, y_cate, model)
#         sign1 = numpy.sign(grad1)
#         # 计算相似样本损失函数，及最远相似样本
#         grad2, max_similar = compute_loss_grad_MP(similar_items, y_cate, model)
#         sign2 = numpy.sign(grad2)
#         # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#         direction = numpy.zeros_like(x_i)
#         for n in range(x.shape[1]):
#             # if n not in [10, 11, 12] and sign1[n] != sign2[n]:
#             if sign1[n] != sign2[n]:
#                 if sign1[n] != 0:
#                     direction[0, n] = sign1[n]
#                 else:
#                     direction[0, n] = -1 * sign2[n]
#             if sign1[n] == 0:
#                 direction[0, n] = -1 * sign2[n]
#         #  扰动所选择属性
#         perturbation = extent * direction
#         perturbation = tensorflow.sign(perturbation) * tensorflow.minimum(tensorflow.math.abs(perturbation), 0.05)
#         perturbation = numpy.array(perturbation)
#         generated_x = x_i + perturbation
#         x_i = generated_x
#         similar_x_i = add_perturbation_to_similar_items(similar_items, perturbation)
#         AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
#                                         numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i, dist=0, K=0)
#         if AF_C[2]:
#             return numpy.concatenate((x_i, y_i.reshape(1, -1)), axis=1)
#     return numpy.concatenate((x_i, y_i.reshape(1, -1)), axis=1)
#
#
# def FB_attack(model, test_item, dataset, protected_attr, search_times, extent):
#     """
#     对样本进行TB搜索，返回搜索结果，相似样本，以及该样本的公平混淆矩阵分类结果
#     :return:
#     """
#     similar_items = generate_similar_items(test_item[0, :-1].reshape(1, -1), dataset, protected_attr)
#     return FB(model, test_item, similar_items, search_times, extent)
#
#
# def run_TrueFair_Repair(model_file, test_file, dataset, protected_attr, search_time, extent, result_file):
#     """
#     进行准确公平性测试
#     :return:
#     """
#     model = keras.models.load_model(model_file)
#     test_data = numpy.load(test_file)
#
#     result = []
#     for i in range(test_data.shape[0]):
#         test_item = test_data[i].copy().reshape(1, -1)
#         R = TF_attack(model, test_item, dataset, protected_attr, search_time, extent)
#         result.append(R)
#     numpy.save(result_file, numpy.squeeze(numpy.array(result)))

# def run_FalseBiasedAttack(model_file, test_file, dataset, protected_attr, search_time, extent, result_file):
#     """
#     进行准确公平性测试
#     :return:
#     """
#     model = keras.models.load_model(model_file)
#     test_data = numpy.load(test_file)
#
#     result = []
#     for i in range(test_data.shape[0]):
#         test_item = test_data[i].copy().reshape(1, -1)
#         R = FB_attack(model, test_item, dataset, protected_attr, search_time, extent)
#         result.append(R)
#     numpy.save(result_file, numpy.squeeze(numpy.array(result)))


# def run_FalseFairAttack(model_file, test_file, dataset, protected_attr, search_time, extent, result_file):
#     """
#     进行准确公平性测试
#     :return:
#     """
#     model = keras.models.load_model(model_file)
#     test_data = numpy.load(test_file)
#
#     result = []
#     for i in range(test_data.shape[0]):
#         test_item = test_data[i].copy().reshape(1, -1)
#         R = FF_attack(model, test_item, dataset, protected_attr, search_time, extent)
#         result.append(R)
#     numpy.save(result_file, numpy.squeeze(numpy.array(result)))


#
# def TB_global(model, test_item, similar_items, search_times, extent):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
#     :return:
#     """
#     generate_data = []
#     generate_similar = []
#     # 保存预测结果错误或歧视的位置
#     x, y = numpy.split(test_item, [-1, ], axis=1)
#     x_i = x.copy().reshape(1, -1)
#     y_i = y.copy()
#     FB_Tag = False
#     Accurate_fairness_confusion = numpy.array([False, False, False])
#     for _ in range(search_times):
#         # 计算损失函数导数符号
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i, y_cate, model)
#         sign1 = numpy.sign(grad1)
#         # 计算相似样本损失函数，及最远相似样本
#         grad2, max_similar = compute_loss_grad_MP(similar_items, y_cate, model)
#         sign2 = numpy.sign(grad2)
#         # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#         direction = numpy.zeros_like(x_i)
#         for n in range(x.shape[1]):
#             # if n not in [10,11,12] and sign1[n] != sign2[n]:
#             if sign1[n] != sign2[n]:
#                 if sign1[n] != 0:
#                     direction[0, n] = -1 * sign1[n]
#                 else:
#                     direction[0, n] = sign2[n]
#         #  扰动所选择属性
#         perturbation = extent * direction
#         perturbation = tensorflow.sign(perturbation) * tensorflow.minimum(tensorflow.math.abs(perturbation), 0.05)
#         perturbation = numpy.array(perturbation)
#         generated_x = x_i + perturbation
#         x_i = generated_x
#         generate_data.append(numpy.concatenate((x_i, y_i.reshape(1, -1)), axis=1))
#
#         similar_x_i = add_perturbation_to_similar_items(similar_items, perturbation)
#         generate_similar.append(similar_x_i)
#         AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
#                                         numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i, dist=0, K=0)
#         if not AF:
#             FB_Tag = True
#         Accurate_fairness_confusion = numpy.logical_or(Accurate_fairness_confusion, AF_C)
#
#     adversarial_data = numpy.squeeze(numpy.array(generate_data))
#     return adversarial_data, generate_similar, FB_Tag, Accurate_fairness_confusion
#
#
# def TB_attack(model, test_item, dataset, protected_attr, search_times, extent):
#     """
#     对样本进行TB搜索，返回搜索结果，相似样本，以及该样本的公平混淆矩阵分类结果
#     :return:
#     """
#     similar_items = generate_similar_items(test_item[0, :-1].reshape(1, -1), dataset, protected_attr)
#     return TB_global(model, test_item, similar_items, search_times, extent)
#
#
# def FF_global(model, test_item, similar_items, search_times, extent):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
#     :return:
#     """
#     generate_data = []
#     generate_similar = []
#     # 保存预测结果错误或歧视的位置
#     x, y = numpy.split(test_item, [-1, ], axis=1)
#     x_i = x.copy().reshape(1, -1)
#     y_i = y.copy()
#     FB_Tag = False
#     Accurate_fairness_confusion = numpy.array([False, False, False])
#     for _ in range(search_times):
#         # 计算损失函数导数符号
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i, y_cate, model)
#         sign1 = numpy.sign(grad1)
#         # 计算相似样本损失函数，及最远相似样本
#         grad2, max_similar = compute_loss_grad_MP(similar_items, y_cate, model)
#         sign2 = numpy.sign(grad2)
#         # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#         direction = numpy.zeros_like(x_i)
#         for n in range(x.shape[1]):
#             # if n not in [10, 11, 12] and sign1[n] == sign2[n]:
#             if sign1[n] == sign2[n]:
#                 direction[0, n] = sign1[n]
#         #  扰动所选择属性
#         perturbation = extent * direction
#         perturbation = tensorflow.sign(perturbation) * tensorflow.minimum(tensorflow.math.abs(perturbation), 0.05)
#         perturbation = numpy.array(perturbation)
#         generated_x = x_i + perturbation
#         x_i = generated_x
#         generate_data.append(numpy.concatenate((x_i, y_i.reshape(1, -1)), axis=1))
#
#         similar_x_i = add_perturbation_to_similar_items(similar_items, perturbation)
#         generate_similar.append(similar_x_i)
#         AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
#                                         numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i, dist=0, K=0)
#         if not AF:
#             FB_Tag = True
#         Accurate_fairness_confusion = numpy.logical_or(Accurate_fairness_confusion, AF_C)
#
#     adversarial_data = numpy.squeeze(numpy.array(generate_data))
#     return adversarial_data, generate_similar, FB_Tag, Accurate_fairness_confusion
#
#
# def FF_attack(model, test_item, dataset, protected_attr, search_times, extent):
#     """
#     对样本进行FF搜索，返回搜索结果，相似样本，以及该样本的公平混淆矩阵分类结果
#     :return:
#     """
#     similar_items = generate_similar_items(test_item[0, :-1].reshape(1, -1), dataset, protected_attr)
#     return FF_global(model, test_item, similar_items, search_times, extent)
#
#
# def FB_global(model, test_item, similar_items, search_times, extent):
#     """
#     计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
#     确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
#     :return:
#     """
#     generate_data = []
#     generate_similar = []
#     # 保存预测结果错误或歧视的位置
#     x, y = numpy.split(test_item, [-1, ], axis=1)
#     x_i = x.copy().reshape(1, -1)
#     y_i = y.copy()
#     FB_Tag = False
#     Accurate_fairness_confusion = numpy.array([False, False, False])
#     for _ in range(search_times):
#         # 计算损失函数导数符号
#         y_cate = to_categorical(y_i, num_classes=2)
#         grad1 = compute_loss_grad(x_i, y_cate, model)
#         sign1 = numpy.sign(grad1)
#         # 计算相似样本损失函数，及最远相似样本
#         grad2, max_similar = compute_loss_grad_MP(similar_items, y_cate, model)
#         sign2 = numpy.sign(grad2)
#         # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
#         direction = numpy.zeros_like(x_i)
#         for n in range(x.shape[1]):
#             # if n not in [10, 11, 12] and sign1[n] != sign2[n]:
#             if sign1[n] != sign2[n]:
#                 if sign1[n] != 0:
#                     direction[0, n] = sign1[n]
#                 else:
#                     direction[0, n] = -1 * sign2[n]
#         #  扰动所选择属性
#         perturbation = extent * direction
#         perturbation = tensorflow.sign(perturbation) * tensorflow.minimum(tensorflow.math.abs(perturbation), 0.05)
#         perturbation = numpy.array(perturbation)
#         generated_x = x_i + perturbation
#         x_i = generated_x
#         generate_data.append(x_i)
#         generate_data.append(numpy.concatenate((x_i, y_i.reshape(1, -1)), axis=1))
#         similar_x_i = add_perturbation_to_similar_items(similar_items, perturbation)
#         generate_similar.append(similar_x_i)
#         AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
#                                         numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i, dist=0, K=0)
#         if not AF:
#             FB_Tag = True
#         Accurate_fairness_confusion = numpy.logical_or(Accurate_fairness_confusion, AF_C)
#
#     adversarial_data = numpy.squeeze(numpy.array(generate_data))
#     return adversarial_data, generate_similar, FB_Tag, Accurate_fairness_confusion
#
#
# def FB_attack(model, test_item, dataset, protected_attr, search_times, extent):
#     """
#     对样本进行TB搜索，返回搜索结果，相似样本，以及该样本的公平混淆矩阵分类结果
#     :return:
#     """
#     similar_items = generate_similar_items(test_item[0, :-1].reshape(1, -1), dataset, protected_attr)
#     return FB_global(model, test_item, similar_items, search_times, extent)
#
#
# def AF_RobustFair_attack(model_file, test_file, dataset, protected_attr, search_time, extent, result_file):
#     """
#     进行准确公平性测试
#     :return:
#     """
#     model = keras.models.load_model(model_file)
#     test_data = numpy.load(test_file)
#
#     Result = []
#     for i in range(test_data.shape[0]):
#         test_item = test_data[i].copy().reshape(1, -1)
#         TB_R = TB_attack(model, test_item, dataset, protected_attr, search_time, extent)
#         FF_R = FF_attack(model, test_item, dataset, protected_attr, search_time, extent)
#         FB_R = FF_attack(model, test_item, dataset, protected_attr, search_time, extent)
#
#         result = numpy.concatenate((TB_R, FF_R, FB_R), axis=0)
#         Result.append(result)
#
#     numpy.save(result_file, numpy.squeeze(numpy.array(Result)))

# def AF_RobustFair_attack(M_file, S_file, D_tag, P_attr, search_time, extent, result_file):
#     """
#     AF 测试样本生成
#     :return:
#     """
#     R = run_RobustFair_experiment(M_file, S_file, D_tag, P_attr, search_time, extent, result_file)
#     numpy.save(result_file, R)
