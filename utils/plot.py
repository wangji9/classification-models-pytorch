import matplotlib.pyplot as plt


def plot_loss(epoch, loss1, loss2,modelname):
    """

    :param epoch:
    :param loss1:
    :param loss2:
    :param model:
    :return:
    """
    plt.figure(figsize=(8, 6))
    plt.plot(epoch, loss1, '.-', label="Train_Loss")
    plt.plot(epoch, loss2, '.-', label="Val_Loss")
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.title('{}_model'.format(modelname),color='blue',fontstyle='italic')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('plot/Loss_{}.jpg'.format(modelname))
    plt.close()


def plot_acc(epoch, acc1, acc2,modelname):
    """

    :param epoch:
    :param acc1:
    :param acc2:
    :param model:
    :return:
    """
    plt.figure(figsize=(8, 6))
    plt.plot(epoch, acc1, 'o-', label="Train_Accuracy")
    plt.plot(epoch, acc2, 'o-', label="Val_Accuracy")
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.title('{}_model'.format(modelname),color='blue',fontstyle='italic')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('plot/Accuracy_{}.jpg'.format(modelname))
    plt.close()

