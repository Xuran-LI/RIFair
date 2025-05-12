import keras
import numpy
import tensorflow

from utils.Z_RobustFair_loss import add_perturbation_to_similar_items
from utils.utils_RIFair import generate_similar_items
from utils.utils_evaluate import check_item_IF
from utils.utils_generate import compute_label_vote_C, compute_grad_EIDIG


def compute_dataset_vote_label(data_x, vote_files):
    """
    计算数据集的 vote label
    :param data:
    :param vote_files:
    :return:
    """
    vote_models = []
    for v_f in vote_files:
        vote_models.append(keras.models.load_model(v_f))
    vote_labels = []
    for i in range(len(vote_models)):
        vote_labels.append(numpy.argmax(vote_models[i].predict(data_x), axis=1))

    vote_labels = numpy.array(vote_labels)
    label = []
    for j in range(vote_labels.shape[1]):
        label.append(numpy.array([numpy.argmax(numpy.bincount(vote_labels[:, j]))]))

    return numpy.concatenate((data_x, label), axis=1)


def check_result_individual_fair(dataset, model, D_tag, P_attr, D_file, C_file):
    """
    检测DICE检测结果的准确公平性
    :return:
    """
    DICE_C = []
    DICE_D = []
    for i in range(dataset.shape[0]):
        if i == 0:
            DICE_D = dataset[i]
        else:
            DICE_D = numpy.concatenate((DICE_D, dataset[i]), axis=0)
        Bias_T = False
        generated_data = dataset[i]
        for j in range(len(generated_data)):
            data = numpy.array(generated_data[j]).reshape(1, -1)
            similar_data = generate_similar_items(data, D_tag, P_attr)
            IF = check_item_IF(numpy.argmax(model.predict(data), axis=1),
                               numpy.argmax(model.predict(similar_data), axis=1), data, similar_data, 0, K=0)
            if not IF:
                Bias_T = True
            break
        DICE_C.append(Bias_T)

    numpy.save(D_file, DICE_D)
    numpy.save(C_file, DICE_C)


def clustering(probs, m_sample, sens_params, epsillon):
    cluster_dic = {}
    cluster_dic['Seed'] = m_sample[0]
    bins = numpy.arange(0, 1, epsillon)
    digitized = numpy.digitize(probs, bins) - 1
    for k in range(len(digitized)):
        if digitized[k] not in cluster_dic.keys():
            cluster_dic[digitized[k]] = [[m_sample[k][j] for j in sens_params]]
        else:
            cluster_dic[digitized[k]].append([m_sample[k][j] for j in sens_params])
    return cluster_dic


def global_sample_select(clus_dic, sens_params):
    leng = 0
    for key in clus_dic.keys():
        if key == 'Seed':
            continue
        if len(clus_dic[key]) > leng:
            leng = len(clus_dic[key])
            largest = key
    sample_ind = numpy.random.randint(len(clus_dic[largest]))
    n_sample_ind = numpy.random.randint(len(clus_dic[largest]))
    sample = clus_dic['Seed']
    for i in range(len(sens_params)):
        sample[sens_params[i]] = clus_dic[largest][sample_ind][i]
    # returns one sample of largest partition and its pair
    return numpy.array([sample]), clus_dic[largest][n_sample_ind]


def clip(input, y):
    """
    Clip the generating instance with each feature to make sure it is valid
    :param input: generating instance
    :param conf: the configuration of dataset
    :return: a valid generating instance
    """
    input = input[0].tolist()
    result = []
    for i in range(len(input)):
        if input[i] > 1:
            result.append(1)
        elif input[i] < 0:
            result.append(0)
        else:
            result.append(input[i])

    result.append(y[0, 0])

    return result


def check_DICE(sample, y_i, dataset, protected_attr, max_iter, extent, model, data_size):
    """

    :return:
    """
    for iter in range(max_iter):
        similar_x_i = generate_similar_items(sample, dataset, protected_attr)
        pred = model.predict(similar_x_i)
        n, pred = numpy.split(pred, [-1, ], axis=1)
        clus_dic = clustering(pred.reshape(-1), similar_x_i, protected_attr, 0.01)
        sample, n_values = global_sample_select(clus_dic, protected_attr)
        dis_sample = sample.copy()
        for sens in protected_attr:
            dis_sample[0][sens] = 0
        # Making up n_sample
        n_sample = sample.copy()
        for i in range(len(protected_attr)):
            n_sample[0][protected_attr[i]] = n_values[i]
        # global perturbation
        grad1 = compute_grad_EIDIG(sample, model)
        s_grad = numpy.sign(grad1)

        grad2 = compute_grad_EIDIG(n_sample, model)
        n_grad = numpy.sign(grad2)
        # find the feature with same impact
        if numpy.zeros(data_size - 1).tolist() == s_grad.tolist():
            g_diff = n_grad
        elif numpy.zeros(data_size - 1).tolist() == n_grad.tolist():
            g_diff = s_grad
        else:
            g_diff = numpy.array(s_grad == n_grad, dtype=float)
        for sens in protected_attr:
            g_diff[sens] = 0
        cal_grad = s_grad * g_diff
        if numpy.zeros(data_size - 1).tolist() == cal_grad.tolist():
            index = numpy.random.randint(len(cal_grad) - 1)
            for i in range(len(protected_attr) - 1, -1, -1):
                if index == protected_attr[i]:
                    index = index

            cal_grad[index] = numpy.random.choice([1.0, -1.0])
        perturbation = extent * cal_grad
        perturbation = tensorflow.sign(perturbation) * tensorflow.minimum(tensorflow.math.abs(perturbation), 0.05)
        perturbation = numpy.array(perturbation)
        sample = clip(sample + perturbation, y_i)
        sample = numpy.array(sample[:-1]).reshape(1, -1)

        similar_x_i = add_perturbation_to_similar_items(similar_x_i, perturbation)
        IF = check_item_IF(numpy.argmax(model.predict(sample), axis=1),
                           numpy.argmax(model.predict(similar_x_i), axis=1), sample, similar_x_i, 0, 0)
        if not IF:
            return numpy.concatenate((sample, y_i.reshape(1, -1)), axis=1)

    return numpy.concatenate((sample, y_i.reshape(1, -1)), axis=1)


def fair_DICE_experiment(M_file, test_file, dataset, protected_attr, max_iter, extent, Dice_f):
    """

    :return:
    """
    model = keras.models.load_model(M_file)
    test_data = numpy.load(test_file)
    seed_num = 0
    result = []

    for num in range(test_data.shape[0]):
        print('input ', seed_num)
        test_item = test_data[num].copy().reshape(1, -1)
        x, y = numpy.split(test_item, [-1, ], axis=1)
        sample = x.copy().reshape(1, -1)
        y_i = y.copy().reshape(1, -1)
        R = check_DICE(sample, y_i, dataset, protected_attr, max_iter, extent, model, test_data.shape[1])
        result.append(R)
        seed_num += 1

    print('Search Done!')
    numpy.save(Dice_f, result)
