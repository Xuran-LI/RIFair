import random
from random import uniform
import keras
from keras import backend
from sklearn import cluster
import numpy
from numpy import sqrt
import tensorflow
from tensorflow.python.keras.losses import mean_squared_error, categorical_crossentropy, binary_crossentropy
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical


# def compute_loss_grad(x, y, model, loss_func=binary_crossentropy):
#     """计算模型损失函数相对于输入features的导数 dloss(f(x),y)/d(x)"""
#     x = tensorflow.Variable(x, dtype=tensorflow.float32)
#     y = tensorflow.Variable(y, dtype=tensorflow.float32)
#     with tensorflow.GradientTape() as tape:
#         tape.watch(x)
#         loss = loss_func(y, model(x))
#         gradient = tape.gradient(loss, x)
#     return gradient


def compute_loss_grad(index, value, label, model, loss_func=categorical_crossentropy):
    """计算模型损失函数相对于输入features的导数 dloss(f(x),y)/d(x)"""
    index = tensorflow.Variable(index.reshape(1, -1), dtype=tensorflow.float32)
    value = tensorflow.Variable(value.reshape(1, -1), dtype=tensorflow.float32)
    label = tensorflow.Variable(label, dtype=tensorflow.float32)
    with tensorflow.GradientTape() as tape:
        tape.watch([model.trainable_variables, value])
        loss = loss_func(label, model([index, value]))
        gradient = tape.gradient(loss, [model.trainable_variables, value])
    return gradient[0][0].values, gradient[1]


def compute_loss_grad_AVG(x_list, label, model, loss_func=mean_squared_error):
    """计算相似样本距离函数对非保护属性X的平均导数 D(f(x,a'),y)"""
    label = tensorflow.constant([label], dtype=tensorflow.float32)
    gradients = []
    for x in x_list:
        x = tensorflow.constant(x.reshape(1, -1), dtype=tensorflow.float32)
        with tensorflow.GradientTape() as tape:
            tape.watch(x)
            loss = loss_func(label, model(x))
        gradient = tape.gradient(loss, x)
        gradients.append(gradient[0].numpy())
    result = numpy.mean(numpy.array(gradients), axis=0)
    return result


def compute_loss_grad_MP(index_list, value_list, label, model, loss_func=categorical_crossentropy):
    """计算相似样本距离函数对非保护属性X的最大导数 D(f(x,a'),y)"""
    label = tensorflow.constant(label, dtype=tensorflow.float32)
    gradients = []
    for i in range(index_list.shape[0]):
        index = tensorflow.constant(index_list[i].reshape(1, -1), dtype=tensorflow.float32)
        value = tensorflow.Variable(value_list[i].reshape(1, -1), dtype=tensorflow.float32)
        with tensorflow.GradientTape() as tape:
            tape.watch([model.trainable_variables, value])
            loss = loss_func(label, model([index, value]))
            gradient = tape.gradient(loss, [model.trainable_variables, value])
        gradients.append([gradient[0][0].values.numpy(), gradient[1].numpy()])
    max_id = compute_Max_grad(gradients)
    return gradients[max_id], [index_list[max_id].reshape(-1), value_list[max_id].reshape(-1)]


def compute_Max_grad(gradients):
    """
    计算绝对值最大的grad
    :return:
    """
    max_id = 0
    max_grad = 0
    for i in range(len(gradients)):
        grad_sum = numpy.sum(numpy.abs(gradients[i][0])) + numpy.sum(numpy.abs(gradients[i][1]))
        if grad_sum > max_grad:
            max_id = i
    return max_id


# 计算预测距离最远的相似样本
def far_similar_R(pre_list, pre):
    """返回相似样本集中预测结果与真实标记差别最大的索引,回归任务"""
    initial_id = 0
    initial_gap = abs(pre_list[0] - pre[0])[0]
    for i in range(pre_list.shape[0]):
        if abs(pre_list[i] - pre[0])[0] > initial_gap:
            initial_id = i
            initial_gap = abs(pre_list[i] - pre[0])[0]
    return initial_id


def far_similar_C(pre_list, pre):
    """返回相似样本集中预测结果与真实标记差别最大的索引，分类任务"""
    initial_id = 0
    initial_gap = numpy.sum(abs(pre_list[0] - pre[0]))
    for i in range(pre_list.shape[0]):
        if numpy.sum(abs(pre_list[i] - pre[0])) > initial_gap:
            initial_id = i
            initial_gap = abs(pre_list[i] - pre[0])[0]
    return initial_id


# 计算预测结果的梯度
def compute_grad_EIDIG(x, model):
    """计算预测结果的梯度 EIDIG"""
    x = tensorflow.constant(x, dtype=tensorflow.float32)
    with tensorflow.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x)
    gradient = tape.gradient(y_pred, x)
    return gradient[0].numpy()


def compute_grad_loss(x, model, loss_func=mean_squared_error):
    """计算预测结果的梯度 ADF"""
    x = tensorflow.constant(x, dtype=tensorflow.float32)
    y_pred = tensorflow.cast(model(x) > 0.5, dtype=tensorflow.float32)
    with tensorflow.GradientTape() as tape:
        tape.watch(x)
        loss = loss_func(y_pred, model(x))
    gradient = tape.gradient(loss, x)
    return gradient[0].numpy()


# 聚类生成种子
def clustering(data_x, data_y, c_num):
    # standard KMeans algorithm
    kmeans = cluster.KMeans(n_clusters=c_num)
    y_pred = kmeans.fit_predict(data_x)
    return [data_x[y_pred == n] for n in range(c_num)], [data_y[y_pred == n] for n in range(c_num)]


def random_pick(probability):
    # randomly pick an element from a probability distribution
    random_number = numpy.random.rand()
    current_proba = 0
    for i in range(len(probability)):
        current_proba += probability[i]
        if current_proba > random_number:
            return i


def get_seed(data_x, data_y, data_len, c_num):
    pick_probability = [len(data_x[i]) / data_len for i in range(c_num)]
    cluster_i = random_pick(pick_probability)
    seed_x = data_x[cluster_i]
    seed_y = data_y[cluster_i]
    index = numpy.random.randint(0, len(seed_x))
    return seed_x[index], seed_y[index]


def get_AF_seed(file1, file2, file3, AF_Tag):
    """
    获取全局搜索种子
    :return:
    """
    model = load_model(file1)
    data1 = numpy.load(file2)
    x1, y1 = numpy.split(data1, [-1, ], axis=1)
    pre = numpy.argmax(model.predict(x1), axis=1).reshape(-1, 1)
    data2 = numpy.squeeze(numpy.load(file3))
    x2 = []
    pre2 = []
    for j in range(data2.shape[1]):
        x2.append(data2[:, j, :])
        pre2.append(numpy.argmax(model.predict(data2[:, j, :]), axis=1).reshape(-1, 1))
    AF_cond = get_AF_condition(y1, pre, pre2, x1, x2, dist=0, K=0)
    if AF_Tag == "TF":
        seeds = data1[AF_cond[0]]
    if AF_Tag == "TB":
        seeds = data1[AF_cond[1]]
    if AF_Tag == "FF":
        seeds = data1[AF_cond[2]]
    if AF_Tag == "FB":
        seeds = data1[AF_cond[3]]

    # numpy.random.shuffle(seeds)
    i = random.randint(0, seeds.shape[0] - 1)
    return seeds[i, :].reshape(1, -1)


def get_AF_seeds(file1, file2, file3, seeds_num, AF_Tag):
    """
    获取全局搜索种子
    :return:
    """
    model = load_model(file1)
    data1 = numpy.load(file2)
    x1, y1 = numpy.split(data1, [-1, ], axis=1)
    pre = numpy.argmax(model.predict(x1), axis=1).reshape(-1, 1)
    data2 = numpy.squeeze(numpy.load(file3))
    x2 = []
    pre2 = []
    for j in range(data2.shape[1]):
        x2.append(data2[:, j, :])
        pre2.append(numpy.argmax(model.predict(data2[:, j, :]), axis=1).reshape(-1, 1))
    AF_cond = get_AF_condition(y1, pre, pre2, x1, x2, dist=0, K=0)
    if AF_Tag == "TF":
        seeds = data1[AF_cond[0]]
    if AF_Tag == "TB":
        seeds = data1[AF_cond[1]]
    if AF_Tag == "FF":
        seeds = data1[AF_cond[2]]
    if AF_Tag == "FB":
        seeds = data1[AF_cond[3]]

    numpy.random.shuffle(seeds)
    return seeds[:seeds_num, :]


def get_search_seeds(test_file, cluster_num, sample_num):
    """
    获取全局搜索种子
    :return:
    """
    seeds = []
    test_data = numpy.load(test_file)
    test_x, test_y = numpy.split(test_data, [-1, ], axis=1)
    cluster_x, cluster_y = clustering(test_x, test_y, cluster_num)
    for i in range(sample_num):
        s_x, s_y = get_seed(cluster_x, cluster_y, test_x.shape[0], cluster_num)
        seeds.append(numpy.concatenate((s_x, s_y), axis=0))
    return numpy.squeeze(numpy.array(seeds))


def prepare_search_seeds(test_file, cluster_num, sample_num):
    """
    获取全局搜索种子
    :return:
    """
    seeds = []
    test_data = numpy.load(test_file)
    test_x, test_y = numpy.split(test_data, [-1, ], axis=1)
    cluster_x, cluster_y = clustering(test_x, test_y, cluster_num)
    for i in range(sample_num):
        s_x, s_y = get_seed(cluster_x, cluster_y, test_x.shape[0], cluster_num)
        seeds.append(numpy.concatenate((s_x, s_y), axis=0))

    avg_cluster_x = []
    avg_cluster_y = []
    for j in range(cluster_num):
        avg_cluster_x.append(numpy.mean(cluster_x[j], axis=0))
        avg_cluster_y.append(numpy.mean(cluster_y[j], axis=0))

    return numpy.squeeze(numpy.array(seeds)), \
           numpy.concatenate((numpy.array(avg_cluster_x), numpy.array(avg_cluster_y)), axis=1)


# 基于泰勒公式计算生成样本的标记
def multiple_matrix(m1, m2):
    result = 0
    for i in range(m1.shape[1]):
        result += m1[0, i] * m2[i, 0]
    return result


def compute_label_Taylor_R(x1, label1, gradient1, perturbation, x2, model, loss_func=mean_squared_error):
    "根据泰勒公式，近似计算生成样本的真实标记，回归任务"
    label1 = tensorflow.constant([label1], dtype=tensorflow.float32)
    x1 = tensorflow.constant(x1, dtype=tensorflow.float32)
    x2 = tensorflow.constant(x2, dtype=tensorflow.float32)
    with tensorflow.GradientTape() as tape:
        tape.watch(x1)
        pre2 = model(x2)[0].numpy()[0]
        loss = loss_func(label1, model(x1))[0].numpy()
    x2_label_1 = pre2 + sqrt(abs(multiple_matrix(perturbation.reshape(1, -1), gradient1.reshape(-1, 1)) + loss))
    x2_label_2 = pre2 - sqrt(abs(multiple_matrix(perturbation.reshape(1, -1), gradient1.reshape(-1, 1)) + loss))
    if abs(x2_label_1 - label1) <= abs(x2_label_2 - label1):
        return numpy.array([x2_label_1]).astype(float)
    else:
        return numpy.array([x2_label_2]).astype(float)


def compute_loss(model, x1, similar_x1, label1, loss_func=mean_squared_error):
    "根据泰勒公式，近似计算生成样本的真实标记，分类任务"
    label1 = tensorflow.constant([label1], dtype=tensorflow.float32)
    x1 = tensorflow.constant(x1, dtype=tensorflow.float32)
    similar_x1 = tensorflow.constant(similar_x1, dtype=tensorflow.float32)
    with tensorflow.GradientTape() as tape:
        tape.watch(x1)
        loss1 = loss_func(label1, model(x1)).numpy()
        similar_loss1 = loss_func(label1, model(similar_x1)).numpy()
    return numpy.squeeze(numpy.array([loss1, similar_loss1]))


def approximate_by_total_derivative(model, x1, similar_x1, label1, gradient1, gradient2, extent, direction,
                                    loss_func=mean_squared_error):
    "根据泰勒公式，近似计算生成样本的真实标记，分类任务"
    label1 = tensorflow.constant([label1], dtype=tensorflow.float32)
    x1 = tensorflow.constant(x1, dtype=tensorflow.float32)
    similar_x1 = tensorflow.constant(similar_x1, dtype=tensorflow.float32)
    with tensorflow.GradientTape() as tape:
        tape.watch(x1)
        loss1 = loss_func(label1, model(x1)).numpy()
        similar_loss1 = loss_func(label1, model(similar_x1)).numpy()

    # approximated loss and similar_loss
    loss2 = multiple_matrix(extent * direction.reshape(1, -1), gradient1.reshape(-1, 1)) + loss1
    similar_loss2 = multiple_matrix(extent * direction.reshape(1, -1), gradient2.reshape(-1, 1)) + similar_loss1

    # perturbed x1
    generated_x1 = x1 + extent * direction.reshape(1, -1)
    generated_similar_x1 = similar_x1 + extent * direction.reshape(1, -1)

    # 判断 [1,0]与【0，1】的损失与loss2的接近程度
    loss2_0 = loss_func(to_categorical(0, num_classes=2), model(generated_x1))
    loss2_1 = loss_func(to_categorical(1, num_classes=2), model(generated_x1))

    similar_loss2_0 = loss_func(to_categorical(0, num_classes=2), model(generated_similar_x1))
    similar_loss2_1 = loss_func(to_categorical(1, num_classes=2), model(generated_similar_x1))

    if abs(loss2_0 - loss2) <= abs(loss2_1 - loss2):
        return [0], numpy.squeeze(numpy.array([loss2_0, similar_loss2_0]))
    else:
        return [1], numpy.squeeze(numpy.array([loss2_1, similar_loss2_1]))


def approximate_by_calculus_Avg(model, x1, similar_x1, label1, gradient1, gradient2, extent, direction,
                                loss_func=mean_squared_error):
    "根据泰勒公式，近似计算生成样本的真实标记，分类任务"
    loss_0 = []
    loss_1 = []
    label1 = tensorflow.constant([label1], dtype=tensorflow.float32)
    x1 = tensorflow.constant(x1, dtype=tensorflow.float32)
    similar_x1 = tensorflow.constant(similar_x1, dtype=tensorflow.float32)
    with tensorflow.GradientTape() as tape:
        tape.watch(x1)
        loss1 = loss_func(label1, model(x1)).numpy()
        # similar_loss1 = loss_func(label1, model(similar_x1)).numpy()
        for j in range(similar_x1.shape[0]):
            generated_similar_xj = similar_x1[j] + extent * direction.reshape(1, -1)
            loss_0.append(loss_func(to_categorical(0, num_classes=2), model(generated_similar_xj)))
            loss_1.append(loss_func(to_categorical(1, num_classes=2), model(generated_similar_xj)))

    # approximated loss and similar_loss
    loss2 = multiple_matrix(extent * direction.reshape(1, -1), gradient1.reshape(-1, 1)) + loss1

    # perturbed x1
    generated_x1 = x1 + extent * direction.reshape(1, -1)

    # 判断 [1,0]与【0，1】的损失与loss2的接近程度
    loss2_0 = loss_func(to_categorical(0, num_classes=2), model(generated_x1))
    loss2_1 = loss_func(to_categorical(1, num_classes=2), model(generated_x1))

    similar_loss2_0 = numpy.mean(numpy.array(loss_0), axis=0)
    similar_loss2_1 = numpy.mean(numpy.array(loss_1), axis=0)

    if abs(loss2_0 - loss2) <= abs(loss2_1 - loss2):
        return [0], numpy.squeeze(numpy.array([loss2_0, similar_loss2_0]))
    else:
        return [1], numpy.squeeze(numpy.array([loss2_1, similar_loss2_1]))


# def compute_label_Taylor_C(x1, label1, gradient1, perturbation, x2, model, loss_func=mean_squared_error):
#     "根据泰勒公式，近似计算生成样本的真实标记，分类任务"
#     label1 = tensorflow.constant([label1], dtype=tensorflow.float32)
#     x1 = tensorflow.constant(x1, dtype=tensorflow.float32)
#     x2 = tensorflow.constant(x2, dtype=tensorflow.float32)
#     with tensorflow.GradientTape() as tape:
#         tape.watch(x1)
#         pre2 = model(x2)
#         loss1 = loss_func(label1, model(x1)).numpy()
#     # loss2=loss1+loss'*perturbation
#     loss2 = multiple_matrix(perturbation.reshape(1, -1), gradient1.reshape(-1, 1)) + loss1
#     # 判断 [1,0]与【0，1】的损失与loss2的接近程度
#     loss2_0 = loss_func(to_categorical(0, num_classes=2), pre2)
#     loss2_1 = loss_func(to_categorical(1, num_classes=2), pre2)
#     if abs(loss2_0 - loss2) <= abs(loss2_1 - loss2):
#         return [0]
#     else:
#         return [1]


def compute_label_vote_R(x, models):
    "根据投票计算生成样本的真实标记，回归任务"
    pres = []
    for m in models:
        pres.append(m.predict(x))
    return numpy.average(pres)


def compute_dataset_vote_label_R(data_file, vote_files):
    """
    计算数据集的 vote label
    :param data_file:
    :param vote_files:
    :return:
    """
    data = numpy.load(data_file)
    data_x, data_y = numpy.split(data, [-1, ], axis=1)

    vote_models = []
    for v_f in vote_files:
        vote_models.append(keras.models.load_model(v_f))

    vote_y = []
    for i in range(data_x.shape[0]):
        vote_y.append(compute_label_vote_R(data_x[i].reshape(1, -1), vote_models))

    return numpy.concatenate((data_x, numpy.array(vote_y).reshape(-1, 1)), axis=1)


def compute_label_vote_C(x, models):
    "根据投票计算生成样本的真实标记，分类任务"
    pres = []
    for m in models:
        pres.append(numpy.argmax(m.predict(x), axis=1)[0])
    return numpy.array([numpy.argmax(numpy.bincount(pres))])


def compute_dataset_vote_label_C(data_x, vote_files):
    """
    计算数据集的 vote label
    :param data:
    :param vote_files:
    :return:
    """
    vote_models = []
    for v_f in vote_files:
        vote_models.append(keras.models.load_model(v_f))
    vote_items = []
    for i in range(len(data_x)):
        vote_y = []
        for j in range(len(data_x[i])):
            vote_y.append(compute_label_vote_C(data_x[i][j].reshape(1, -1), vote_models))
        vote_items.append(numpy.concatenate((data_x[i], numpy.array(vote_y).reshape(-1, 1)), axis=1))
    return vote_items


#  对扰动顺序进行排序
# def sort_perturbation_direction(direction, protected):
#     """
#     对非保护属性梯度，按从小到大的顺序排序
#     :return:
#     """
#     sort_result = []
#     for i in range(direction.shape[1]):
#         min_id = i
#         min_data = abs(direction[0, i])
#         for j in range(direction.shape[1]):
#             if abs(direction[0, j]) < min_data:
#                 min_id = j
#                 min_data = direction[0, j]
#             j += 1
#         #  设置最小值位置为无穷大
#         direction[0, min_id] = 1000
#         if min_id not in protected and direction[0, min_id] != 0:
#             sort_result.append(min_id)
#     return sort_result
def sort_perturbation_direction(direction):
    """
    对非保护属性梯度，按从小到大的顺序排序
    :return:
    """
    sort_result = []
    for i in range(direction.shape[1]):
        min_id = i
        min_data = abs(direction[0, i])
        for j in range(direction.shape[1]):
            if abs(direction[0, j]) < min_data:
                min_id = j
                min_data = direction[0, j]
            j += 1
        #  设置最小值位置为无穷大
        direction[0, min_id] = 1000
        if direction[0, min_id] != 0:
            sort_result.append(min_id)
    return sort_result


# 数据增强
def data_augmentation_ctrip(data, protected_index):
    """
    对待测数据进行公平数据增强，生成保护属性不同，非保护属性，标签相同的样本
    :param data: 待测样本
    :param protected_index: 社会敏感属性，如性别、年龄、种族、地域信息等
    :return: 一组非保护属性、标签相同、敏感属性不同的相似样本
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_0 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
            for a_1 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                for a_2 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                    for a_3 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                        for a_4 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                            for a_5 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                                aug_data = data[i].tolist()
                                aug_data[protected_index[0]] = a_0
                                aug_data[protected_index[1]] = a_1
                                aug_data[protected_index[2]] = a_2
                                aug_data[protected_index[3]] = a_3
                                aug_data[protected_index[4]] = a_4
                                aug_data[protected_index[5]] = a_5
                                # data_list.append(aug_data)
                                # 生成测试数据集相似样本时，去除真实标记
                                data_list.append(aug_data[:-1])
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_ctrip_item(data, protected_index):
    """
    对数据进行数据增强，生成保护属性不同，非保护属性，标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_0 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
            for a_1 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                for a_2 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                    for a_3 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                        for a_4 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                            for a_5 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                                aug_data = data[i].tolist()
                                aug_data[protected_index[0]] = a_0
                                aug_data[protected_index[1]] = a_1
                                aug_data[protected_index[2]] = a_2
                                aug_data[protected_index[3]] = a_3
                                aug_data[protected_index[4]] = a_4
                                aug_data[protected_index[5]] = a_5
                                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_adult(data, protected_index):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_0 in [uniform(17, 30), uniform(30, 60), uniform(60, 90)]:
            for a_1 in [uniform(0, 2), uniform(2, 4)]:
                for a_2 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                    aug_data = data[i].tolist()
                    aug_data[protected_index[0]] = round(a_0) / (90 - 17)  # 归一化
                    aug_data[protected_index[1]] = round(a_1) / (4)
                    aug_data[protected_index[2]] = round(a_2) / (1)
                    # data_list.append(aug_data)
                    # 生成测试数据集相似样本时，去除真实标记
                    data_list.append(aug_data[:-1])
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_adult_item(data, protected_index):
    """
    对数据进行数据增强，生成保护属性不同，非保护属性，标签相同的样本
    :return:
    """
    aug = []
    if len(protected_index) > 1:
        for i in range(data.shape[0]):
            data_list = []
            for a_0 in [uniform(17, 30), uniform(30, 60), uniform(60, 90)]:
                for a_1 in [uniform(0, 2), uniform(2, 4)]:
                    for a_2 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                        aug_data = data[i].tolist()
                        aug_data[protected_index[0]] = round(a_0) / (90 - 17)  # 归一化
                        aug_data[protected_index[1]] = round(a_1) / (4)
                        aug_data[protected_index[2]] = round(a_2) / (1)
                        data_list.append(aug_data)
            aug.append(data_list)
    else:  # 单个保护属性数据增强
        # age augmentation
        if protected_index[0] == 10:
            for i in range(data.shape[0]):
                data_list = []
                for a_0 in [uniform(17, 30), uniform(30, 60), uniform(60, 90)]:
                    aug_data = data[i].tolist()
                    aug_data[protected_index[0]] = round(a_0) / (90 - 17)  # 归一化
                    data_list.append(aug_data)
                aug.append(data_list)

        # race augmentation
        elif protected_index[0] == 11:
            for i in range(data.shape[0]):
                data_list = []
                for a_1 in [uniform(0, 2), uniform(2, 4)]:
                    aug_data = data[i].tolist()
                    aug_data[protected_index[0]] = round(a_1) / (4)
                    data_list.append(aug_data)
                aug.append(data_list)
        # sex augmentation
        elif protected_index[0] == 12:
            for i in range(data.shape[0]):
                data_list = []
                for a_2 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                    aug_data = data[i].tolist()
                    aug_data[protected_index[0]] = round(a_2) / (1)
                    data_list.append(aug_data)
                aug.append(data_list)

    return numpy.squeeze(numpy.array(aug))


def data_augmentation_credit(data, protected_index):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_0 in [0, 1]:
            for a_1 in [uniform(19, 35), uniform(35, 55), uniform(55, 75)]:
                aug_data = data[i].tolist()
                aug_data[protected_index[0]] = a_0
                aug_data[protected_index[1]] = round(a_1) / (75 - 19)  # 归一化
                # data_list.append(aug_data)
                # 生成测试数据集相似样本时，去除真实标记
                data_list.append(aug_data[:-1])
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_credit_item(data, protected_index):
    """
    对数据进行数据增强，生成保护属性不同，非保护属性，标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_0 in [0, 1]:
            for a_1 in [uniform(19, 35), uniform(35, 55), uniform(55, 75)]:
                aug_data = data[i].tolist()
                aug_data[protected_index[0]] = a_0
                aug_data[protected_index[1]] = round(a_1) / (75 - 19)  # 归一化
                data_list.append(aug_data)
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_compas(data, protected_index):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_0 in [uniform(18, 30), uniform(30, 60), uniform(60, 96)]:
            for a_1 in [uniform(0, 2.5), uniform(2.5, 5)]:
                for a_2 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                    aug_data = data[i].tolist()
                    aug_data[protected_index[0]] = round(a_0) / (96 - 18)  # 归一化
                    aug_data[protected_index[1]] = round(a_1) / (5)
                    aug_data[protected_index[2]] = round(a_2) / (1)
                    # data_list.append(aug_data)
                    # 生成测试数据集相似样本时，去除真实标记
                    data_list.append(aug_data[:-1])
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_compas_item(data, protected_index):
    """
    对数据进行数据增强，生成保护属性不同，非保护属性，标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_0 in [uniform(18, 30), uniform(30, 60), uniform(60, 96)]:
            for a_1 in [uniform(0, 2.5), uniform(2.5, 5)]:
                for a_2 in [uniform(0, 0.5), uniform(0.5, 1.0)]:
                    aug_data = data[i].tolist()
                    aug_data[protected_index[0]] = round(a_0) / (96 - 18)  # 归一化
                    aug_data[protected_index[1]] = round(a_1) / (5)
                    aug_data[protected_index[2]] = round(a_2) / (1)
                    data_list.append(aug_data)
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_bank(data, protected_index):
    """
    对数据进行数据增强，生成保护属性取极值，非保护属性、标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_1 in [uniform(18, 35), uniform(35, 55), uniform(55, 75), uniform(75, 95)]:
            aug_data = data[i].tolist()
            aug_data[protected_index[0]] = round(a_1) / (95 - 18)  # 归一化
            # data_list.append(aug_data)
            # 生成测试数据集相似样本时，去除真实标记
            data_list.append(aug_data[:-1])
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


def data_augmentation_bank_item(data, protected_index):
    """
    对数据进行数据增强，生成保护属性不同，非保护属性，标签相同的样本
    :return:
    """
    aug = []
    for i in range(data.shape[0]):
        data_list = []
        for a_1 in [uniform(18, 35), uniform(35, 55), uniform(55, 75), uniform(75, 95)]:
            aug_data = data[i].tolist()
            aug_data[protected_index[0]] = round(a_1) / (95 - 18)  # 归一化
            data_list.append(aug_data)
        aug.append(data_list)
    return numpy.squeeze(numpy.array(aug))


# gradient normalization during local search
def normal_prob(grad1, grad2, protected_attribs, epsilon):
    gradient = numpy.zeros_like(grad1)
    grad1 = numpy.abs(grad1)
    grad2 = numpy.abs(grad2)
    for i in range(len(gradient)):
        saliency = grad1[i] + grad2[i]
        gradient[i] = 1.0 / (saliency + epsilon)
        if i in protected_attribs:
            gradient[i] = 0.0
    gradient_sum = numpy.sum(gradient)
    probability = gradient / gradient_sum
    return probability

# def save_generation_result_ctrip(test_generation, protected, model_file, file1, file2, eval_file, epsilon, K, times):
#     """
#
#     :return:
#     """
#     numpy.save(file1, test_generation)
#     numpy.save(file2, data_augmentation_ctrip(test_generation, protected))
#
#     result_evaluation = get_model_evaluation1(model_file, file1, file2, epsilon, K)
#     header_name = ["avg", "std", "acc R", "false R", "acc N", "false N", "SUM",
#                    "IFR", "IBR", "IFN", "IBN", "SUM",
#                    "A&F R", "F|B R", "A&F", "F|B", "SUM",
#                    "TFR", "TBR", "FFR", "FBR", 'F_recall', 'F_precision', 'F_F1', 'TF', 'TB', 'FF', "FB", "SUM",
#                    "search time"]
#     workbook_name = xlsxwriter.Workbook(eval_file)
#     worksheet = workbook_name.add_worksheet("Generation Details")
#     write_worksheet_header(header_name, worksheet)
#     write_worksheet_2d_data([result_evaluation + [times]], worksheet)
#     print(result_evaluation + [times])
#     workbook_name.close()

# individual fairness
# def get_IF_global_seeds_R(file1, file2, file3, cluster_num, sample_times, K):
#     """
#     获取迭代过程中的种子
#     :return:
#     """
#     model = keras.models.load_model(file1)
#     data1 = numpy.load(file2)
#     x, y = numpy.split(data1, [-1, ], axis=1)
#     pre = model.predict(x)
#     data2 = numpy.squeeze(numpy.load(file3))
#     similar_x = []
#     similar_pre = []
#     for j in range(data2.shape[1]):
#         predicate = model.predict(data2[:, j, :])
#         similar_x.append(data2[:, j, :])
#         similar_pre.append(predicate)
#     # 获取FTA公平性测试结果
#     conditions = get_IF_condition(pre, similar_pre, x, similar_x, K)
#     seeds_x = []
#     seeds_y = []
#     for h in range(len(conditions) - 1):
#         true_x, true_y = get_item_label_by_condition(conditions[h + 1], x, y)
#         if len(true_x) > 0:
#             c_true_x, c_true_y = clustering(true_x, true_y, cluster_num)
#             for i in range(sample_times):
#                 s_x, s_y = get_seed(c_true_x, c_true_y, true_x.shape[0], cluster_num)
#                 seeds_x.append(s_x)
#                 seeds_y.append(s_y)
#     return numpy.array(seeds_x), numpy.array(seeds_y)
#
#
# def get_IF_local_seeds_R(model, data, similar_data, K):
#     """
#     获取局部搜索时歧视样本
#     :return:
#     """
#     # 测试集
#     x, y = numpy.split(data, [-1, ], axis=1)
#     pre = model.predict(x)
#     # 相似样本测试集
#     similar_x = []
#     similar_pre = []
#     for j in range(similar_data.shape[1]):
#         predicate = model.predict(similar_data[:, j, :])
#         similar_x.append(similar_data[:, j, :])
#         similar_pre.append(predicate)
#     conditions = get_IF_condition(pre, similar_pre, x, similar_x, K)
#
#     bias_x, bias_y = get_item_label_by_condition(conditions[1], x, y)
#     return bias_x, bias_y
#
#
# def get_IF_global_seeds_C(file1, file2, file3, cluster_num, sample_times, K):
#     """
#     获取迭代过程中的种子
#     :return:
#     """
#     model = keras.models.load_model(file1)
#     data1 = numpy.load(file2)
#     x, y = numpy.split(data1, [-1, ], axis=1)
#     pre = numpy.argmax(model.predict(x), axis=1).reshape(-1, 1)
#     data2 = numpy.squeeze(numpy.load(file3))
#     similar_x = []
#     similar_pre = []
#     for j in range(data2.shape[1]):
#         predicate = model.predict(data2[:, j, :])
#         similar_x.append(data2[:, j, :])
#         similar_pre.append(numpy.argmax(predicate, axis=1).reshape(-1, 1))
#     # 获取FTA公平性测试结果
#     conditions = get_IF_condition(pre, similar_pre, x, similar_x, K)
#     seeds_x = []
#     seeds_y = []
#     for h in range(len(conditions) - 1):
#         true_x, true_y = get_item_label_by_condition(conditions[h + 1], x, y)
#         if len(true_x) > 0:
#             c_true_x, c_true_y = clustering(true_x, true_y, cluster_num)
#             for i in range(sample_times):
#                 s_x, s_y = get_seed(c_true_x, c_true_y, true_x.shape[0], cluster_num)
#                 seeds_x.append(s_x)
#                 seeds_y.append(s_y)
#     return numpy.array(seeds_x), numpy.array(seeds_y)
#
#
# def get_IF_local_seeds_C(model, data, similar_data, K):
#     """
#     获取局部搜索时歧视样本
#     :return:
#     """
#     # 测试集
#     x, y = numpy.split(data, [-1, ], axis=1)
#     pre = numpy.argmax(model.predict(x), axis=1).reshape(-1, 1)
#     # 相似样本测试集
#     similar_x = []
#     similar_pre = []
#     for j in range(similar_data.shape[1]):
#         predicate = model.predict(similar_data[:, j, :])
#         similar_x.append(similar_data[:, j, :])
#         similar_pre.append(numpy.argmax(predicate, axis=1).reshape(-1, 1))
#     conditions = get_IF_condition(pre, similar_pre, x, similar_x, K)
#
#     bias_x, bias_y = get_item_label_by_condition(conditions[1], x, y)
#     return bias_x, bias_y


# # 链接测试生成样本
# def connect_generated_data(file_lists, similar_file_list, result_file, similar_result_file):
#     """
#     将生成的测试数据整合
#     :return:
#     """
#     test_data = []
#     test_data_similar = []
#     for i in range(len(file_lists)):
#         if os.path.exists(file_lists[i]):
#             test_data.append(numpy.load(file_lists[i]))
#             test_data_similar.append(numpy.load(similar_file_list[i]))
#
#     initial_test_data = test_data[0]
#     initial_test_data_similar = test_data_similar[0]
#
#     for j in range(len(test_data) - 1):
#         initial_test_data = numpy.concatenate((initial_test_data, test_data[j + 1]), axis=0)
#         initial_test_data_similar = numpy.concatenate((initial_test_data_similar, test_data_similar[j + 1]), axis=0)
#
#     uniques_test, unique_similar_test = unique_data(initial_test_data, initial_test_data_similar)
#     numpy.save(result_file, uniques_test)
#     numpy.save(similar_result_file, unique_similar_test)


# def get_accurate_fairness_bug_num(model_name, test_file, test_file_similar, epsilon, K):
#     """
#     根据模型获取结果
#     :return:
#     """
#     # AF_table.field_names =
#     model = keras.models.load_model("../dataset/ctrip/model/{}.h5".format(model_name))
#     # 测试集
#     test_dataset = numpy.load(test_file)
#     test_x, test_y = numpy.split(test_dataset, [-1, ], axis=1)
#     test_pre = model.predict(test_x)
#     # 相似样本测试集
#     similar_test_data = numpy.squeeze(numpy.load(test_file_similar))
#     similar_x = []
#     similar_pre = []
#     for j in range(similar_test_data.shape[1]):
#         predicate = model.predict(similar_test_data[:, j, :])
#         similar_x.append(similar_test_data[:, j, :])
#         similar_pre.append(predicate)
#
#     AF_bug_num = check_accurate_fairness_bug(test_y, test_pre, similar_pre, test_x, similar_x, epsilon, K)
#     data_sum = test_x.shape[0]
#     return AF_bug_num, data_sum
#
#
# def check_accurate_fairness_bug(label, pre_x, pre_similar_x, x, similar_x, epsilon, K):
#     """
#     计算数据集的的准确公平性 bug 数量
#     """
#     # 样本performance
#     performance = calculate_MSE(pre_x, label)
#     performance_cond = check_epsilon(performance, epsilon)
#     # 计算相似样本的condition
#     performance_cond_similar = []
#     for i in range(len(pre_similar_x)):
#         # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
#         D_distance = calculate_MSE(label, pre_similar_x[i])
#         Kd_distance = K * calculate_MSE(x, similar_x[i])
#         performance_cond_similar.append(check_epsilon(D_distance - Kd_distance, epsilon))
#     similar_fair_cond = performance_cond_similar[0]
#     for j in range(len(performance_cond_similar)):
#         similar_fair_cond = numpy.logical_and(similar_fair_cond, performance_cond_similar[j])
#
#     # 计算 TF TB FF FB condition
#     TF_cond = numpy.logical_and(performance_cond, similar_fair_cond)
#     TB_cond = numpy.logical_and(performance_cond, ~similar_fair_cond)
#     FF_cond = numpy.logical_and(~performance_cond, similar_fair_cond)
#     FB_cond = numpy.logical_and(~performance_cond, ~similar_fair_cond)
#
#     TF = numpy.sum(TF_cond)
#     TB = numpy.sum(TB_cond)
#     FF = numpy.sum(FF_cond)
#     FB = numpy.sum(FB_cond)
#
#     return TB + FF + FB


# def compute_loss_grad_similar_sum(similar_x, label, model, loss_func=mean_squared_error):
#     "计算相似样本距离函数对非保护属性X的累加导数 D(f(x,a'),y)"
#     label = tensorflow.constant([label], dtype=tensorflow.float32)
#     gradients = []
#     for similar in similar_x:
#         similar = tensorflow.constant(similar.reshape(1, -1), dtype=tensorflow.float32)
#         with tensorflow.GradientTape() as tape:
#             tape.watch(similar)
#             D_distance = loss_func(label, model(similar))
#         gradient = tape.gradient(D_distance, similar)
#         gradients.append(gradient[0].numpy())
#     result = gradients[0]
#     for j in range(len(gradients) - 1):
#         result += gradients[j + 1]
#     return result


# def abs_sum_list(data):
#     """
#     求list中的绝对值和
#     :return:
#     """
#     abs_sum = 0
#     for i in range(len(data)):
#         abs_sum += abs(data[i])
#     return abs_sum
#
#
# def compute_loss_grad_similar_max(similar_x, label, model, loss_func=mean_squared_error):
#     "计算相似样本距离函数对非保护属性X的最大导数 D(f(x,a'),y)"
#     label = tensorflow.constant([label], dtype=tensorflow.float32)
#     gradients = []
#     for similar in similar_x:
#         similar = tensorflow.constant(similar.reshape(1, -1), dtype=tensorflow.float32)
#         with tensorflow.GradientTape() as tape:
#             tape.watch(similar)
#             D_distance = loss_func(label, model(similar))
#         gradient = tape.gradient(D_distance, similar)
#         gradients.append(gradient[0].numpy())
#     max_id = 0
#     max_gradient = abs_sum_list(gradients[0])
#     for j in range(len(gradients) - 1):
#         if abs_sum_list(gradients[j + 1]) > max_gradient:
#             max_id = j + 1
#             max_gradient = abs_sum_list(gradients[j + 1])
#     return gradients[max_id]
#
#


# def compute_distance_grad_FF_FB(x, similar_x, model, loss_func=mean_squared_error):
#     "false fair, false bias 计算距离函数对非保护属性X的导数 z=D(f(x,a),f(x,a'))"
#     x = tensorflow.constant(x, dtype=tensorflow.float32)
#     gradients = []
#     for similar in similar_x:
#         similar = tensorflow.constant(similar.reshape(1, -1), dtype=tensorflow.float32)
#         with tensorflow.GradientTape(persistent=True) as tape:
#             tape.watch(x)
#             y_pred = model(x)
#             tape.watch(similar)
#             similar_pred = model(similar)
#             D_distance = loss_func(y_pred, similar_pred)
#         gradient = tape.gradient(D_distance, x)
#         # gradient = tape.gradient(D_distance, similar)
#         del tape
#         gradients.append(gradient[0].numpy())
#
#     result = gradients[0]
#     for j in range(len(gradients) - 1):
#         result += gradients[j + 1]
#     return result

# "ctrip样本数据增强"


# 检查样本的准确公平性
# def check_accurate_fairness_item(label, pre_x, pre_similar_x, x, similar_x, epsilon, K):
#     """
#     检查样本的准确公平性
#     :return:
#     """
#     # 样本performance
#     performance = calculate_MSE(pre_x.reshape(1, 1), label.reshape(1, 1))
#     performance_cond = check_epsilon(performance, epsilon)
#     # 计算相似样本的condition
#     performance_cond_similar = []
#     for i in range(len(pre_similar_x)):
#         # D(y,f(similar_x))<=Kd(x,similar_x)+epsilon
#         D_distance = calculate_MSE(label.reshape(1, 1), pre_similar_x[i].reshape(1, 1))
#         Kd_distance = K * calculate_MSE(x, similar_x[i].reshape(1, -1))
#         performance_cond_similar.append(check_epsilon(D_distance - Kd_distance, epsilon))
#     similar_cond = performance_cond_similar[0]
#     for j in range(len(performance_cond_similar)):
#         similar_cond = numpy.logical_and(similar_cond, performance_cond_similar[j])
#
#     if performance_cond[0]:
#         # True predication
#         if similar_cond[0]:
#             # Fair
#             return [True, False, False, False]
#         else:
#             # Bias
#             return [False, True, False, False]
#     else:
#         # False predication
#         if similar_cond[0]:
#             # Fair
#             return [False, False, True, False]
#         else:
#             # bias
#             return [False, False, False, True]


# 检查样本的FTA公平性


# def check_item_fair_epsilon(pre_x, pre_similar_x, x, similar_x, epsilon, K):
#     """
#     检查指定item的模型预测结果是否满足 D(f(x),f(similar_x))<=Kd(x,similar_x)+epsilon
#     :return:
#     """
#     D_distance = calculate_item_MSE(pre_x, pre_similar_x)
#     d_distance = calculate_MSE(x, similar_x)[0]
#     if D_distance <= K * d_distance + epsilon:
#         return True
#     else:
#         return False


# def get_global_search_seeds_true_false(model_name, test_file, c_num, search_times, epsilon):
#     """
#     获取初始全局搜索的种子
#     :return:
#     """
#     model = keras.models.load_model("../dataset/ctrip/model/{}.h5".format(model_name))
#
#     test_dataset = numpy.load(test_file)
#     test_x, test_y = numpy.split(test_dataset, [-1, ], axis=1)
#     seeds_x = []
#     seeds_y = []
#     MSE_performance = calculate_MSE(model.predict(test_x), test_y)
#     MSE_performance_cond = check_epsilon(MSE_performance, epsilon)
#
#     true_x, true_y = get_item_label_by_condition(MSE_performance_cond, test_x, test_y)
#     c_true_x, c_true_y = clustering(true_x, true_y, c_num)
#     for i in range(search_times):
#         seed_x, seed_y = get_seed(c_true_x, c_true_y, true_x.shape[0], c_num)
#         seeds_x.append(seed_x)
#         seeds_y.append(seed_y)
#
#     false_x, false_y = get_item_label_by_condition(~MSE_performance_cond, test_x, test_y)
#     c_false_x, c_false_y = clustering(false_x, false_y, c_num)
#     for i in range(search_times):
#         seed_x, seed_y = get_seed(c_false_x, c_false_y, false_x.shape[0], c_num)
#         seeds_x.append(seed_x)
#         seeds_y.append(seed_y)
#
#     global_x = numpy.array(seeds_x)
#     global_y = numpy.array(seeds_y)
#
#     return global_x, global_y

# individual fairness


# def get_AF_global_seeds_R(file1, file2, file3, cluster_num, sample_times, epsilon, K):
#     """
#     聚类，采样准确公平bug
#     :return:
#     """
#     model = keras.models.load_model(file1)
#     data1 = numpy.load(file2)
#     x, y = numpy.split(data1, [-1, ], axis=1)
#     pre = model.predict(x)
#     data2 = numpy.squeeze(numpy.load(file3))
#     similar_x = []
#     similar_pre = []
#     for j in range(data2.shape[1]):
#         predicate = model.predict(data2[:, j, :])
#         similar_x.append(data2[:, j, :])
#         similar_pre.append(predicate)
#     conditions = get_AF_condition(y, pre, similar_pre, x, similar_x, epsilon, K)
#     seeds_x = []
#     seeds_y = []
#     for h in range(len(conditions) - 1):
#         cluster_x, cluster_y = get_item_label_by_condition(conditions[h + 1], x, y)
#         if len(cluster_x) > 0:
#             x_1, y_1 = clustering(cluster_x, cluster_y, cluster_num)
#             for i in range(sample_times):
#                 x_2, y_2 = get_seed(x_1, y_1, cluster_x.shape[0], cluster_num)
#                 seeds_x.append(x_2)
#                 seeds_y.append(y_2)
#     return numpy.array(seeds_x), numpy.array(seeds_y)
#
#
# def get_AF_local_seeds_R(model, data, similar_data, epsilon, K):
#     """
#     获取局部搜索时的accurate fairness bug
#     :return:
#     """
#     # 测试集
#     x, y = numpy.split(data, [-1, ], axis=1)
#     pre = model.predict(x)
#     # 相似样本测试集
#     similar_x = []
#     similar_pre = []
#     for j in range(similar_data.shape[1]):
#         predicate = model.predict(similar_data[:, j, :])
#         similar_x.append(similar_data[:, j, :])
#         similar_pre.append(predicate)
#     conditions = get_AF_condition(y, pre, similar_pre, x, similar_x, epsilon, K)
#
#     x1, y1 = get_item_label_by_condition(conditions[1], x, y)
#     x2, y2 = get_item_label_by_condition(conditions[2], x, y)
#     x3, y3 = get_item_label_by_condition(conditions[3], x, y)
#
#     return numpy.concatenate((x1, x2, x3,), axis=0), numpy.concatenate((y1, y2, y3,), axis=0)
#
#
# def get_AF_global_seeds_C(file1, file2, file3, cluster_num, sample_times, epsilon, K):
#     """
#     聚类，采样准确公平bug
#     :return:
#     """
#     model = keras.models.load_model(file1)
#     data1 = numpy.load(file2)
#     x, y = numpy.split(data1, [-1, ], axis=1)
#     pre = numpy.where(model.predict(x).reshape(-1, 1) > 0.5, 1, 0)
#     data2 = numpy.squeeze(numpy.load(file3))
#     similar_x = []
#     similar_pre = []
#     for j in range(data2.shape[1]):
#         predicate = model.predict(data2[:, j, :])
#         similar_x.append(data2[:, j, :])
#         similar_pre.append(numpy.where(predicate.reshape(-1, 1) > 0.5, 1, 0))
#     conditions = get_AF_condition(y, pre, similar_pre, x, similar_x, epsilon, K)
#     seeds_x = []
#     seeds_y = []
#     for h in range(len(conditions) - 1):
#         cluster_x, cluster_y = get_item_label_by_condition(conditions[h + 1], x, y)
#         if len(cluster_x) > 0:
#             x_1, y_1 = clustering(cluster_x, cluster_y, cluster_num)
#             for i in range(sample_times):
#                 x_2, y_2 = get_seed(x_1, y_1, cluster_x.shape[0], cluster_num)
#                 seeds_x.append(x_2)
#                 seeds_y.append(y_2)
#     return numpy.array(seeds_x), numpy.array(seeds_y)
#
#
# def get_AF_local_seeds_C(model, data, similar_data, epsilon, K):
#     """
#     获取局部搜索时的accurate fairness bug
#     :return:
#     """
#     # 测试集
#     x, y = numpy.split(data, [-1, ], axis=1)
#     pre = numpy.where(model.predict(x) > 0.5, 1, 0)
#     # 相似样本测试集
#     similar_x = []
#     similar_pre = []
#     for j in range(similar_data.shape[1]):
#         predicate = model.predict(similar_data[:, j, :])
#         similar_x.append(similar_data[:, j, :])
#         similar_pre.append(numpy.where(predicate > 0.5, 1, 0))
#     conditions = get_AF_condition(y, pre, similar_pre, x, similar_x, epsilon, K)
#
#     x1, y1 = get_item_label_by_condition(conditions[1], x, y)
#     x2, y2 = get_item_label_by_condition(conditions[2], x, y)
#     x3, y3 = get_item_label_by_condition(conditions[3], x, y)
#
#     return numpy.concatenate((x1, x2, x3,), axis=0), numpy.concatenate((y1, y2, y3,), axis=0)

# # 合并生成结果
# def merge_generation(global_search, local_search):
#     """
#     处理EIDIG生成的测试样本,将本轮搜索结果相对于之前搜索结果进行去重
#     :return:
#     """
#     # 全局搜索结果
#     g_search = global_search[0]
#     g_search_similar = global_search[1]
#     g_try_times = global_search[2]
#     # 局部搜索结果
#     l_search = local_search[0]
#     l_search_similar = local_search[1]
#     l_try_times = local_search[2]
#     # 对全局及局部搜索结果进行合并
#     search_data = numpy.concatenate((g_search, l_search), axis=0)
#     similar_search_data = numpy.concatenate((g_search_similar, l_search_similar), axis=0)
#     search_data, similar_search_data = unique_data(search_data, similar_search_data)
#     return search_data, similar_search_data, g_try_times + l_try_times
