import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_dim),
            nn.Tanh()
            # 将生成的随机的噪音映射到更高的维度 逐步提取特征
            # 使用LeakyReLU来防止神经元死亡
            # 通过Tanh激活函数使其压缩到-1,1之间 以确保图像像素值在此范围内
            # 增加鲁棒性 减少梯度消失导致的收敛困难的可能性
        )

    def forward(self, x):
        return self.model(x)
        # 前向传播

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
            # 将输入的图像(真实或生成的)通过多层线性变换映射为二分类输出
            # 通过映射到不同的维度提取其全局特征并同样使用LeakyReLU激活函数
            # 最后经过Sigmoid函数映射到0-1之间 也就是图像为真实的概率
        )

    def forward(self, x):
        return self.model(x)
        # 向前传播
