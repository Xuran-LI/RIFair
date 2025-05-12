import numpy
import seaborn
from matplotlib import pyplot


def draw_heatmap(heat_data, x_label, y_label, title, save_file):
    """
    绘制公平混淆矩阵
    [[True],[False],[SUM]
    [True bias],[False bias],[bias]
    [True fair],[False fair],[fair]]

    confusion_data=[TF,TB,FF,FB]
    :return:
    """
    fig, ax = pyplot.subplots()
    heatmap = ax.imshow(heat_data, cmap="YlGn")
    # 设置colorbar
    cbar = ax.figure.colorbar(heatmap, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")
    # 设置x，y轴标签
    ax.set_xticks(numpy.arange(len(x_label)))
    ax.set_yticks(numpy.arange(len(y_label)))
    ax.set_xticklabels(x_label)
    ax.set_yticklabels(y_label)
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # # 在热力图各个方格中标记数据
    # for i in range(len(y_label)):
    #     for j in range(len(x_label)):
    #         ax.text(j, i, "{:.4f}".format(heat_data[i][j]), fontsize=15, ha="center", va="center", color="black")
    # 绘制图片标题
    if title[0]:
        ax.set_title("TF")
    elif title[1]:
        ax.set_title("TB")
    elif title[2]:
        ax.set_title("FF")
    else:
        ax.set_title("FB")
    # 保存并展示图片
    fig.tight_layout()
    pyplot.savefig(save_file, bbox_inches='tight')
    pyplot.show()


def draw_adv_loss(loss, names, x_label, y_label, title, file):
    """
    绘制折线图
    :return:
    """
    seaborn.set()
    fig, axs = pyplot.subplots(1, 1)
    x = [n for n in range(len(loss[0]))]
    loss0 = loss[0]
    loss1 = loss[1]
    loss2 = loss[2]

    axs.plot(x, loss0, label=names[0])
    axs.plot(x, loss1, label=names[1])
    axs.plot(x, loss2, label=names[2])
    # axs.set_ylim(-0.1, 1.1)
    axs.set_xlabel(x_label)
    axs.set_xticks(range(0, len(loss0), 1))
    axs.set_ylabel(y_label)
    axs.set_title(title)
    fig.tight_layout()
    pyplot.legend(loc="upper left", ncol=2)
    seaborn.despine()
    seaborn.axes_style()
    pyplot.savefig(file, bbox_inches='tight')
    pyplot.show()


def draw_adv_pre(pre, labels, names, title, file):
    """
    绘制折线图
    :return:
    """
    seaborn.set()
    fig, axs = pyplot.subplots(1, 1)

    x = [n for n in range(len(labels))]
    pre0 = pre[0]
    pre1 = pre[1]

    # # 绘制标签点，决策分界点
    # axs.scatter(0, 1, label="Positive Label")
    # axs.scatter(1, 0, label="Negative Label")
    # axs.scatter(0.5, 0.5, label="Decision Boundary")
    #
    # x0 = []
    y0 = []
    # x1 = []
    y1 = []
    for i in range(len(labels)):
        numpy.nonzero(labels[i])
        y0.append(pre0[i][numpy.nonzero(labels[i])])
        y1.append(pre1[i][numpy.nonzero(labels[i])])

    axs.plot(x, y0, label=names[0])
    axs.plot(x, y1, label=names[1])
    axs.set_ylim(-0.1, 1.1)

    axs.set_title(title)
    fig.tight_layout()
    pyplot.legend(loc="upper right", ncol=2)
    seaborn.despine()
    seaborn.axes_style()
    pyplot.savefig(file, bbox_inches='tight')
    pyplot.show()


def draw_confusion(ax, confusion_data, x_label, y_label, model_name):
    """
    绘制公平混淆矩阵
    [[True],[False],[SUM]
    [True bias],[False bias],[bias]
    [True fair],[False fair],[fair]]

    confusion_data=[TF,TB,FF,FB]
    :return:
    """
    draw_data = [[confusion_data[0] + confusion_data[1], confusion_data[2] + confusion_data[3], 1],
                 [confusion_data[1], confusion_data[3], confusion_data[1] + confusion_data[3]],
                 [confusion_data[0], confusion_data[2], confusion_data[0] + confusion_data[2]]]
    im = ax.imshow(draw_data, cmap="YlGn", vmax=1.0, vmin=0)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")
    # ax.set_title(model_name)

    ax.set_xticks(numpy.arange(len(x_label)))
    ax.set_yticks(numpy.arange(len(y_label)))
    ax.set_xticklabels(x_label)
    ax.set_yticklabels(y_label)

    for i in range(len(y_label)):
        for j in range(len(x_label)):
            ax.text(j, i, "{:.4f}".format(draw_data[i][j]), ha="center", va="center", color="black")


def draw_detection_fairness_confusion(confusion_data, output_file):
    """
    绘制detection数据的公平混淆矩阵
    :return:
    """
    x_labels = ["TRUE", "FALSE", "SUM"]
    y_labels = ["SUM", "BIAS", "FAIR"]
    # BL 模型，及数据增强BL模型
    fig, ax = pyplot.subplots(2, 4, figsize=(20, 10))
    draw_confusion(ax[0, 0], confusion_data[0], x_labels, y_labels, "FGSM")
    draw_confusion(ax[0, 1], confusion_data[1], x_labels, y_labels, "PGD")
    draw_confusion(ax[0, 2], confusion_data[2], x_labels, y_labels, "APGD")
    draw_confusion(ax[0, 3], confusion_data[3], x_labels, y_labels, "ACG")
    draw_confusion(ax[1, 0], confusion_data[4], x_labels, y_labels, "ADF")
    draw_confusion(ax[1, 1], confusion_data[4], x_labels, y_labels, "EIDIG")
    draw_confusion(ax[1, 2], confusion_data[4], x_labels, y_labels, "AccFair")
    ax[1, 3].remove()

    fig.tight_layout()
    pyplot.savefig(output_file, bbox_inches='tight')
    pyplot.show()


def draw_evaluation_fairness_confusion(confusion_data, output_file):
    """
    绘制evaluation数据的公平混淆矩阵
    :return:
    """
    x_labels = ["TRUE", "FALSE", "SUM"]
    y_labels = ["SUM", "BIAS", "FAIR"]
    # BL 模型，及数据增强BL模型
    fig, ax = pyplot.subplots(2, 4, figsize=(20, 10))
    draw_confusion(ax[0, 0], confusion_data[0], x_labels, y_labels, "BL")
    draw_confusion(ax[0, 1], confusion_data[1], x_labels, y_labels, "Re-FGSM")
    draw_confusion(ax[0, 2], confusion_data[2], x_labels, y_labels, "Re-PGD")
    draw_confusion(ax[0, 3], confusion_data[3], x_labels, y_labels, "Re-APGD")
    draw_confusion(ax[1, 0], confusion_data[4], x_labels, y_labels, "Re-ACG")
    draw_confusion(ax[1, 1], confusion_data[5], x_labels, y_labels, "Re-ADF")
    draw_confusion(ax[1, 2], confusion_data[5], x_labels, y_labels, "Re-EIDIG")
    draw_confusion(ax[1, 3], confusion_data[5], x_labels, y_labels, "Re-AccFair")

    fig.tight_layout()
    pyplot.savefig(output_file, bbox_inches='tight')
    pyplot.show()


def draw_lines_loss(x, y, names, x_label, y_label, P_title, output_file):
    """
    绘制折线图
    :return:
    """
    seaborn.set()
    fig, axs = pyplot.subplots(1, 1)
    for i in range(len(names)):
        axs.plot(x, y[i], label=names[i])
        # axs.set_ylim(-0.1, 1.1)
    axs.set_xlabel(x_label, fontsize=10)
    axs.set_ylabel(y_label, fontsize=10)

    # axs.xaxis.get_major_formatter().set_powerlimits((0, 1))
    # axs.yaxis.get_major_formatter().set_powerlimits((0, 1))
    axs.set_title(P_title, fontsize=30)
    fig.tight_layout()
    pyplot.legend(loc="upper left", ncol=2, fontsize=25)
    seaborn.despine()
    seaborn.axes_style()
    pyplot.savefig(output_file, bbox_inches='tight')
    pyplot.show()


def draw_lines_num(x, y, names, x_label, y_label, P_title, output_file):
    """
    绘制折线图
    :return:
    """
    fig, axs = pyplot.subplots(1, 1, figsize=(5, 3.5))
    for i in range(len(names)):
        axs.plot(x, y[i], label=names[i])
    axs.set_xlabel(x_label, fontsize=10)
    axs.set_ylabel(y_label, fontsize=10)

    # axs.xaxis.get_major_formatter().set_powerlimits((0, 1))
    # axs.yaxis.get_major_formatter().set_powerlimits((0, 1))
    axs.set_title(P_title, fontsize=10)
    fig.tight_layout()
    pyplot.legend(loc="upper left")
    pyplot.savefig(output_file, bbox_inches='tight')
    pyplot.show()


def draw_model_performance_lines(x, y, bottom, top, names, x_label, y_label, P_title, output_file):
    """
    绘制折线图
    :return:
    """
    fig, axs = pyplot.subplots(1, 1, figsize=(5, 3.5))
    for i in range(len(names)):
        axs.plot(x, y[:, i], label=names[i])
        axs.set_ylim(bottom, top)
        axs.set_xlim(0.75, 0.95)
    axs.set_xlabel(x_label, fontsize=10)
    axs.set_ylabel(y_label, fontsize=10)

    # axs.xaxis.get_major_formatter().set_powerlimits((0, 1))
    # axs.yaxis.get_major_formatter().set_powerlimits((0, 1))
    axs.set_title(P_title, fontsize=10)
    fig.tight_layout()
    pyplot.legend(loc="lower left")
    pyplot.savefig(output_file, bbox_inches='tight')
    pyplot.show()
