import matplotlib.pyplot as plt


def plot_loss(epoch, loss1, loss2,model):
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
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('plot/Loss_{}.jpg'.format(model))
    plt.close()


def plot_acc(epoch, acc1, acc2,model):
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
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('plot/Accuracy_{}.jpg'.format(model))
    plt.close()

