import time
import keras
import numpy
import xlsxwriter as xlsxwriter
from tensorflow.python.keras.utils.np_utils import to_categorical
from utils.utils_evaluate import robustness_result_evaluation, check_items_AF, check_item_confusion
from utils.utils_input_output import write_worksheet_header, write_worksheet_2d_data
from utils.utils_generate import data_augmentation_adult_item, compute_loss_grad_AVG, approximate_by_total_derivative, \
    compute_loss_grad, sort_perturbation_direction, get_search_seeds, data_augmentation_adult, \
    data_augmentation_compas_item, data_augmentation_compas, data_augmentation_credit_item, data_augmentation_credit, \
    data_augmentation_bank_item, data_augmentation_bank, approximate_by_calculus_Avg


def FB_global(model, global_seeds, dataset, protected, search_times, extent):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    loss_perturbation = []
    false_or_bias_position = []
    TB_FF_FB_position = []
    x, y = numpy.split(global_seeds, [-1, ], axis=1)
    for i in range(x.shape[0]):
        # 初始化
        x_i = x[i].copy().reshape(1, -1)
        y_i = y[i].copy()
        F_or_B = False
        TB_FF_FB = numpy.array([False, False, False])
        for _ in range(search_times):
            # 生成相似样本
            if dataset == "adult":
                similar_x_i = data_augmentation_adult_item(x_i, protected)
            elif dataset == "compas":
                similar_x_i = data_augmentation_compas_item(x_i, protected)
            elif dataset == "credit":
                similar_x_i = data_augmentation_credit_item(x_i, protected)
            elif dataset == "bank":
                similar_x_i = data_augmentation_bank_item(x_i, protected)
            # 计算损失函数导数符号
            y_cate = to_categorical(y_i, num_classes=2)
            grad1 = compute_loss_grad(x_i, y_cate, model)
            sign1 = numpy.sign(grad1)
            # 计算相似样本损失函数，及最远相似样本
            grad2 = compute_loss_grad_AVG(similar_x_i, y_cate, model)
            sign2 = numpy.sign(grad2)
            # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
            direction = numpy.zeros_like(x_i)
            for n in range(x.shape[1]):
                if n not in protected and sign1[n] != sign2[n]:
                    if sign1[n] != 0:
                        direction[0, n] = sign1[n]
                    else:
                        direction[0, n] = -1 * sign2[n]
            if numpy.all(direction == 0):
                break
            #  扰动所选择属性
            perturbation = extent * direction
            generated_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Avg(model, x_i, similar_x_i, y_cate, grad1, grad2, extent, direction)
            # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
            x_i = generated_x
            y_i = pert_l
            generate_x.append(x_i)
            generate_y.append(y_i)
            loss_perturbation.append(l_p)

            # 检查扰动后样本预测结果是否准确且公平
            if dataset == "adult":
                similar_x_i = data_augmentation_adult_item(x_i, protected)
            elif dataset == "compas":
                similar_x_i = data_augmentation_compas_item(x_i, protected)
            elif dataset == "credit":
                similar_x_i = data_augmentation_credit_item(x_i, protected)
            elif dataset == "bank":
                similar_x_i = data_augmentation_bank_item(x_i, protected)
            AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
                                            numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i,
                                            dist=0, K=0)
            if not AF:
                F_or_B = True
            TB_FF_FB = numpy.logical_or(TB_FF_FB, AF_C)

        false_or_bias_position.append(F_or_B)
        TB_FF_FB_position.append(TB_FF_FB.tolist())

    generate_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
    generate_loss = numpy.squeeze(numpy.array(loss_perturbation))
    false_or_bias_position = numpy.array(false_or_bias_position, dtype=bool)
    TB_FF_FB_position = numpy.array(TB_FF_FB_position, dtype=bool)

    return generate_data, generate_loss, false_or_bias_position, TB_FF_FB_position


def TB_global(model, global_seeds, dataset, protected, search_times, extent):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    loss_perturbation = []
    false_or_bias_position = []
    TB_FF_FB_position = []
    x, y = numpy.split(global_seeds, [-1, ], axis=1)
    for i in range(x.shape[0]):
        # 初始化
        x_i = x[i].copy().reshape(1, -1)
        y_i = y[i].copy()
        F_or_B = False
        TB_FF_FB = numpy.array([False, False, False])
        for _ in range(search_times):
            # 生成相似样本
            if dataset == "adult":
                similar_x_i = data_augmentation_adult_item(x_i, protected)
            elif dataset == "compas":
                similar_x_i = data_augmentation_compas_item(x_i, protected)
            elif dataset == "credit":
                similar_x_i = data_augmentation_credit_item(x_i, protected)
            elif dataset == "bank":
                similar_x_i = data_augmentation_bank_item(x_i, protected)
            # 计算损失函数导数符号
            y_cate = to_categorical(y_i, num_classes=2)
            grad1 = compute_loss_grad(x_i, y_cate, model)
            sign1 = numpy.sign(grad1)
            # 计算相似样本损失函数，及最远相似样本
            grad2 = compute_loss_grad_AVG(similar_x_i, y_cate, model)
            sign2 = numpy.sign(grad2)
            # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
            direction = numpy.zeros_like(x_i)
            for n in range(x.shape[1]):
                if n not in protected and sign1[n] != sign2[n]:
                    if sign1[n] != 0:
                        direction[0, n] = -1 * sign1[n]
                    else:
                        direction[0, n] = sign2[n]
            if numpy.all(direction == 0):
                break
            #  扰动所选择属性
            perturbation = extent * direction
            generated_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Avg(model, x_i, similar_x_i, y_cate, grad1, grad2, extent, direction)
            # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
            x_i = generated_x
            y_i = pert_l
            generate_x.append(x_i)
            generate_y.append(y_i)
            loss_perturbation.append(l_p)
            # 检查扰动后样本预测结果是否准确且公平
            if dataset == "adult":
                similar_x_i = data_augmentation_adult_item(x_i, protected)
            elif dataset == "compas":
                similar_x_i = data_augmentation_compas_item(x_i, protected)
            elif dataset == "credit":
                similar_x_i = data_augmentation_credit_item(x_i, protected)
            elif dataset == "bank":
                similar_x_i = data_augmentation_bank_item(x_i, protected)
            AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
                                            numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i,
                                            dist=0, K=0)
            if not AF:
                F_or_B = True
            TB_FF_FB = numpy.logical_or(TB_FF_FB, AF_C)

        false_or_bias_position.append(F_or_B)
        TB_FF_FB_position.append(TB_FF_FB.tolist())

    generate_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
    generate_loss = numpy.squeeze(numpy.array(loss_perturbation))
    false_or_bias_position = numpy.array(false_or_bias_position, dtype=bool)
    TB_FF_FB_position = numpy.array(TB_FF_FB_position, dtype=bool)

    return generate_data, generate_loss, false_or_bias_position, TB_FF_FB_position


def FF_global(model, global_seeds, dataset, protected, search_times, extent):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    loss_perturbation = []
    false_or_bias_position = []
    TB_FF_FB_position = []
    x, y = numpy.split(global_seeds, [-1, ], axis=1)
    for i in range(x.shape[0]):
        # 初始化
        x_i = x[i].copy().reshape(1, -1)
        y_i = y[i].copy()
        F_or_B = False
        TB_FF_FB = numpy.array([False, False, False])
        for _ in range(search_times):
            # 生成相似样本
            if dataset == "adult":
                similar_x_i = data_augmentation_adult_item(x_i, protected)
            elif dataset == "compas":
                similar_x_i = data_augmentation_compas_item(x_i, protected)
            elif dataset == "credit":
                similar_x_i = data_augmentation_credit_item(x_i, protected)
            elif dataset == "bank":
                similar_x_i = data_augmentation_bank_item(x_i, protected)
            # 计算损失函数导数符号
            y_cate = to_categorical(y_i, num_classes=2)
            grad1 = compute_loss_grad(x_i, y_cate, model)
            sign1 = numpy.sign(grad1)
            # 计算相似样本损失函数，及最远相似样本
            grad2 = compute_loss_grad_AVG(similar_x_i, y_cate, model)
            sign2 = numpy.sign(grad2)
            # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
            direction = numpy.zeros_like(x_i)
            for n in range(x.shape[1]):
                if n not in protected and sign1[n] == sign2[n]:
                    direction[0, n] = sign1[n]
            if numpy.all(direction == 0):
                break
            #  扰动所选择属性
            perturbation = extent * direction
            generated_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Avg(model, x_i, similar_x_i, y_cate, grad1, grad2, extent, direction)
            # 保存扰动后样本，相似样本，标签，扰动后损失函数的变化情况
            x_i = generated_x
            y_i = pert_l
            generate_x.append(x_i)
            generate_y.append(y_i)
            loss_perturbation.append(l_p)

            # 检查扰动后样本预测结果是否准确且公平
            if dataset == "adult":
                similar_x_i = data_augmentation_adult_item(x_i, protected)
            elif dataset == "compas":
                similar_x_i = data_augmentation_compas_item(x_i, protected)
            elif dataset == "credit":
                similar_x_i = data_augmentation_credit_item(x_i, protected)
            elif dataset == "bank":
                similar_x_i = data_augmentation_bank_item(x_i, protected)
            AF, AF_C = check_item_confusion(numpy.array(y_i), numpy.argmax(model.predict(x_i), axis=1),
                                            numpy.argmax(model.predict(similar_x_i), axis=1), x_i, similar_x_i,
                                            dist=0, K=0)
            if not AF:
                F_or_B = True
            TB_FF_FB = numpy.logical_or(TB_FF_FB, AF_C)

        false_or_bias_position.append(F_or_B)
        TB_FF_FB_position.append(TB_FF_FB.tolist())

    generate_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
    generate_loss = numpy.squeeze(numpy.array(loss_perturbation))
    false_or_bias_position = numpy.array(false_or_bias_position, dtype=bool)
    TB_FF_FB_position = numpy.array(TB_FF_FB_position, dtype=bool)

    return generate_data, generate_loss, false_or_bias_position, TB_FF_FB_position


def global_generation(model, global_seeds, dataset, protected, search_times, extent):
    """
    全局搜索：依次远离 break robustness fairness 以及 both
    :return:
    """
    fb, fb_loss, fb_1, fb_position = FB_global(model, global_seeds, dataset, protected, search_times, extent)
    tb, tb_loss, fb_2, tb_position = TB_global(model, global_seeds, dataset, protected, search_times, extent)
    ff, ff_loss, fb_3, ff_position = FF_global(model, global_seeds, dataset, protected, search_times, extent)

    search_result = numpy.concatenate((fb, tb, ff), axis=0)
    search_loss = numpy.concatenate((fb_loss, tb_loss, ff_loss), axis=0)

    unique_result, unique_index = numpy.unique(search_result, return_index=True, axis=0)
    unique_result_search_loss = search_loss[unique_index]

    false_or_bias_position = numpy.logical_or(fb_1, numpy.logical_or(fb_2, fb_3))
    false_or_bias_detail = numpy.logical_or(fb_position, numpy.logical_or(tb_position, ff_position))

    return unique_result, unique_result_search_loss, false_or_bias_position, false_or_bias_detail


def FB_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果靠近、保持真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    loss_perturbation = []
    x, y = numpy.split(local_seeds, [-1, ], axis=1)
    for s_id in range((x.shape[0])):
        # 初始化
        x_i = x[s_id].copy().reshape(1, -1)
        y_i = y[s_id].copy()
        # 生成相似样本
        if dataset == "adult":
            similar_x_i = data_augmentation_adult_item(x_i, protected)
        elif dataset == "compas":
            similar_x_i = data_augmentation_compas_item(x_i, protected)
        elif dataset == "credit":
            similar_x_i = data_augmentation_credit_item(x_i, protected)
        elif dataset == "bank":
            similar_x_i = data_augmentation_bank_item(x_i, protected)
        pre_i = model.predict(x_i)
        S_pre_i = model.predict(similar_x_i)
        AF = check_items_AF(y_i, numpy.argmax(pre_i, axis=1), numpy.argmax(S_pre_i, axis=1), x_i, similar_x_i, 0, K)
        if AF:
            continue
        # 计算损失函数导数符号
        y_cate = to_categorical(y_i, num_classes=2)
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)
        # 计算相似样本损失函数，及最远相似样本
        grad2 = compute_loss_grad_AVG(similar_x_i, y_cate, model)
        sign2 = numpy.sign(grad2)
        # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
        direction = numpy.zeros_like(x_i)
        for n in range(x.shape[1]):
            if n not in protected and sign1[n] != sign2[n]:
                if sign1[n] != 0:
                    direction[0, n] = sign1[n]
                else:
                    direction[0, n] = -1 * sign2[n]
        # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
        sort_privilege = sort_perturbation_direction(direction.copy(), protected)
        if len(sort_privilege) == 0:
            continue
        elif len(sort_privilege) > search_times:
            sort_privilege = sort_privilege[:search_times]

        for j in sort_privilege:  # 根据梯度方向进行局部搜索
            perturbation_direction = numpy.zeros_like(x_i)
            perturbation_direction[0, j] = direction[0, j]
            # 结合扰动变量与扰动方向进行扰动，计算生成的item，label
            perturbation = extent * perturbation_direction
            pert_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Avg(model, x_i, similar_x_i, y_cate, grad1, grad2, extent, direction)
            generate_x.append(pert_x)
            generate_y.append(pert_l)
            loss_perturbation.append(l_p)
    generate_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
    generate_loss = numpy.squeeze(numpy.array(loss_perturbation))
    return generate_data, generate_loss


def TB_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果靠近真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    loss_perturbation = []
    x, y = numpy.split(local_seeds, [-1, ], axis=1)
    for s_id in range((x.shape[0])):
        # 初始化
        x_i = x[s_id].copy().reshape(1, -1)
        y_i = y[s_id].copy()
        # 生成相似样本
        if dataset == "adult":
            similar_x_i = data_augmentation_adult_item(x_i, protected)
        elif dataset == "compas":
            similar_x_i = data_augmentation_compas_item(x_i, protected)
        elif dataset == "credit":
            similar_x_i = data_augmentation_credit_item(x_i, protected)
        elif dataset == "bank":
            similar_x_i = data_augmentation_bank_item(x_i, protected)

        pre_i = model.predict(x_i)
        S_pre_i = model.predict(similar_x_i)
        AF = check_items_AF(y_i, numpy.argmax(pre_i, axis=1), numpy.argmax(S_pre_i, axis=1), x_i, similar_x_i, 0, K)
        if AF:
            continue
        # 计算损失函数导数符号
        y_cate = to_categorical(y_i, num_classes=2)  # 计算分类任务损失函数
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)
        # 计算相似样本损失函数，及最远相似样本
        grad2 = compute_loss_grad_AVG(similar_x_i, y_cate, model)
        sign2 = numpy.sign(grad2)
        # 根据样本损失函数、相似样本损失函数符号 选择扰动的非敏感属性
        direction = numpy.zeros_like(x_i)
        for n in range(x.shape[1]):
            if n not in protected and sign1[n] != sign2[n]:
                if sign1[n] != 0:
                    direction[0, n] = -1 * sign1[n]
                else:
                    direction[0, n] = sign2[n]
                    # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
        sort_privilege = sort_perturbation_direction(direction.copy(), protected)
        if len(sort_privilege) == 0:
            continue
        elif len(sort_privilege) > search_times:
            sort_privilege = sort_privilege[:search_times]

        for j in sort_privilege:
            perturbation_direction = numpy.zeros_like(x_i)
            perturbation_direction[0, j] = direction[0, j]
            # 结合扰动变量与扰动方向进行扰动，计算生成的item，label
            perturbation = extent * perturbation_direction
            pert_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Avg(model, x_i, similar_x_i, y_cate, grad1, grad2, extent, direction)
            generate_x.append(pert_x)
            generate_y.append(pert_l)
            loss_perturbation.append(l_p)

    generate_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
    generate_loss = numpy.squeeze(numpy.array(loss_perturbation))
    return generate_data, generate_loss


def FF_local(model, local_seeds, dataset, protected, search_times, extent, K=0):
    """
    计算样本的损失函数导数, 计算相似样本预测距离函数梯度 z=∑(D(f(x,a'),y) dz/dx
    确定扰动方向，原样本预测结果远离真实标价，相似样本预测结果远离真实标记
    :return:
    """
    generate_x = []
    generate_y = []
    loss_perturbation = []
    x, y = numpy.split(local_seeds, [-1, ], axis=1)
    for s_id in range((x.shape[0])):
        # 初始化
        x_i = x[s_id].copy().reshape(1, -1)
        y_i = y[s_id].copy()
        # 生成相似样本
        if dataset == "adult":
            similar_x_i = data_augmentation_adult_item(x_i, protected)
        elif dataset == "compas":
            similar_x_i = data_augmentation_compas_item(x_i, protected)
        elif dataset == "credit":
            similar_x_i = data_augmentation_credit_item(x_i, protected)
        elif dataset == "bank":
            similar_x_i = data_augmentation_bank_item(x_i, protected)

        pre_i = model.predict(x_i)
        S_pre_i = model.predict(similar_x_i)
        AF = check_items_AF(y_i, numpy.argmax(pre_i, axis=1), numpy.argmax(S_pre_i, axis=1), x_i, similar_x_i, 0, K)
        if AF:
            continue
        # 计算损失函数导数符号
        y_cate = to_categorical(y_i, num_classes=2)
        grad1 = compute_loss_grad(x_i, y_cate, model)
        sign1 = numpy.sign(grad1)
        # 计算相似样本损失函数，及最远相似样本
        grad2 = compute_loss_grad_AVG(similar_x_i, y_cate, model)
        sign2 = numpy.sign(grad2)
        direction = numpy.zeros_like(x_i)

        for n in range(x.shape[1]):
            if n not in protected and sign1[n] == sign2[n]:
                direction[0, n] = sign1[n]

        # 对各属性的梯度大小进行排序，局部搜索时，按属性梯度的大小依次扰动
        sort_privilege = sort_perturbation_direction(direction.copy(), protected)
        if len(sort_privilege) == 0:
            continue
        elif len(sort_privilege) > search_times:
            sort_privilege = sort_privilege[:search_times]

        for j in sort_privilege:
            perturbation_direction = numpy.zeros_like(x_i)
            perturbation_direction[0, j] = direction[0, j]
            #  扰动所选择属性
            perturbation = extent * perturbation_direction
            pert_x = x_i + perturbation
            # 根据全微分公式近似计算扰动后样本的真实标记，以及扰动后样本损失函数，相似样本损失函数的变化情况
            pert_l, l_p = approximate_by_calculus_Avg(model, x_i, similar_x_i, y_cate, grad1, grad2, extent, direction)
            generate_x.append(pert_x)
            generate_y.append(pert_l)
            loss_perturbation.append(l_p)

    generate_data = numpy.concatenate((numpy.squeeze(numpy.array(generate_x)), numpy.array(generate_y)), axis=1)
    generate_loss = numpy.squeeze(numpy.array(loss_perturbation))
    return generate_data, generate_loss


def local_generation(model, local_seeds, dataset, protected, search_times, extent):
    """
    局部搜索：依次远离 break robustness fairness 以及 both
    :return:
    """
    fb, fb_loss = FB_local(model, local_seeds, dataset, protected, search_times, extent)
    tb, tb_loss = TB_local(model, local_seeds, dataset, protected, search_times, extent)
    ff, ff_loss = FF_local(model, local_seeds, dataset, protected, search_times, extent)

    search_result = numpy.concatenate((fb, tb, ff), axis=0)
    search_loss = numpy.concatenate((fb_loss, tb_loss, ff_loss), axis=0)

    unique_result, unique_index = numpy.unique(search_result, return_index=True, axis=0)
    unique_result_search_loss = search_loss[unique_index]

    return unique_result, unique_result_search_loss, None


def run_RobustFair_experiment(model_file, test_file, dataset, protected, G_time, L_time, extent):
    """
    进行准确公平性测试
    :return:
    """
    model = keras.models.load_model(model_file)
    test_data = numpy.load(test_file)

    G_item, G_loss, F_or_B_P, F_or_B_D = global_generation(model, test_data, dataset, protected, G_time, extent)
    L_item, L_loss, _ = local_generation(model, G_item, dataset, protected, L_time, extent)
    search_item = numpy.concatenate((G_item, L_item), axis=0)
    RobustFair_result, unique_index = numpy.unique(search_item, return_index=True, axis=0)
    return RobustFair_result, F_or_B_P, F_or_B_D


def AF_RobustFair_experiment(M_files, S_file, D_tag, P_attr, G_time, L_time, extent,
                             AF_data_file, AF_position_file, AF_detail_file):
    """
    AF 测试样本生成
    :return:
    """
    for i in range(len(M_files)):
        AF_data, AF_P, AF_D = run_RobustFair_experiment(M_files[i], S_file, D_tag, P_attr, G_time, L_time, extent)
        numpy.save(AF_data_file[i], AF_data)
        numpy.save(AF_position_file[i], AF_P)
        numpy.save(AF_detail_file[i], AF_D)
