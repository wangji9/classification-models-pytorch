import torch.optim as optim

def getoptim(optimizer,parameters,lr):
    """
            params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
            lr (float, 可选) – 学习率（默认：1e-3）
            betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
            eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
            weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
            momentum 动量因子 默认为0
            dampening 动量的抑制因子 默认为0
            nesterov 使用nesterov动量

            lr_decay decay越小，学习率衰减地越慢，当decay = 0时，学习率保持不变。decay越大，学习率衰减地越快，当decay = 1时，学习率衰减最快。
            initial_accumulator_value：初始化加速值，默认为0
    """

    if optimizer == "adam":
        optimizer = optim.Adam(params=parameters,lr=lr)
    if optimizer == "sgd":
        optimizer = optim.SGD(params=parameters,lr=lr)
    if optimizer == "sgdmomentum":
        optimizer = optim.SGD(params=parameters,lr=lr,momentum=0.9)
    if optimizer == "adagrad":
        optimizer = optim.Adagrad(params=parameters,lr=lr)
    if optimizer == "radam":
        optimizer = optim.RAdam(params=parameters,lr=lr)
    if optimizer == "rmsprop":
        optimizer = optim.RMSprop(params=parameters,lr=lr)
    if optimizer == "adamw":
        optimizer = optim.AdamW(params=parameters,lr=lr)
    if optimizer == "asgd":
        optimizer = optim.ASGD(params=parameters,lr=lr)
    if optimizer == "adadelta":
        optimizer = optim.Adadelta(params=parameters,lr=lr)


    return optimizer