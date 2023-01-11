import matplotlib.pyplot as plt


def plot_loss(epoch, loss1, loss2):
    plt.figure(figsize=(8, 6))
    plt.plot(epoch, loss1, '.-', label="Train_Loss")
    plt.plot(epoch, loss2, '.-', label="Val_Loss")
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('Loss.jpg')


def plot_acc(epoch, acc1, acc2):
    plt.figure(figsize=(8, 6))
    plt.plot(epoch, acc1, 'o-', label="Train_Accuracy")
    plt.plot(epoch, acc2, 'o-', label="Val_Accuracy")
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('Accuracy.jpg')

