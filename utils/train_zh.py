import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.GAN_zh import Generator, Discriminator
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def train():

    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 如果cuda可用就是用cuda进行训练

    os.makedirs("images", exist_ok=True)
    # 创建储存训练过程生成图片的文件夹

    generator = Generator(input_dim=100,output_dim=784).to(device)
    discriminator = Discriminator(input_dim=784).to(device)
    # 定义生成器和判别器 因为是MNIST是28x28的单通道图像
    # 所以输入维度是 1x28x28 = 784

    loss_fn = nn.BCELoss()
    # 使用二分类交叉熵损失函数

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # 分别定义生成器和判别器的Adam优化器
    # betas主要适用于控制动量项 0.5和0.999是经验值 能够帮助模型更快的收敛并稳定训练过程

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
    # 定义图像预处理步骤 将图像转换为张量 并标准化到-1到1之间和生成器的输出做匹配

    mnist = datasets.MNIST('../data', train=True, download=True, transform=transform)
    data_loader = DataLoader(mnist, batch_size=64, shuffle=True)
    # 加载MNIST数据集并创建数据加载器

    train_csv = '../mnist_train.csv'
    with open(train_csv, 'w', newline='') as f:
        fieldnames = ['Epoch', 'd_loss', 'g_loss']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # 创建用于记录训练过程损失的CSV文件

    num_epochs = 100
    for epoch in range(num_epochs):

        if epoch % 10 == 0:
            save_images(epoch, generator, device)
        # 每十次训练保存一次训练图片 方便观察到训练进展

        d_total_loss = 0.0
        g_total_loss = 0.0

        for i, (real_images, _) in enumerate(data_loader):

            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1).to(device)
            real_labels = torch.ones(batch_size,1).to(device)
            fake_labels = torch.zeros(batch_size,1).to(device)
            # 取出每一轮训练的图片数量 并根据图片数量将图片拉伸成(batch_size,784)的形式
            # 同时生成对应的真实标签1 和虚假标签0 并将它们移动到设备上

            optimizer_D.zero_grad()
            # 梯度清零

            outputs = discriminator(real_images)
            d_loss_real = loss_fn(outputs, real_labels)
            d_loss_real.backward()
            # 将真实图片投入判别器进行计算并求出损失函数并进行反向传播

            noise = torch.randn(batch_size,100).to(device)
            # 生成随机噪声

            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = loss_fn(outputs, fake_labels)
            d_loss_fake.backward()
            # 通过生成器将随机噪声转化为假图像 并通过判别器计算损失进行反向传播
            # 这里使用detach的原因是要把生成图和生成器完全断开 使其没有关联
            # 把生成图看作一张普通的图片去和判别器进行反向传播
            # 否则会将生成器会被错误的反向传播 导致生成器的参数也被更新

            optimizer_D.step()
            d_total_loss += d_loss_real.item() + d_loss_fake.item()
            # 更新权重并累加总损失

            optimizer_G.zero_grad()

            output = discriminator(fake_images)
            g_loss = loss_fn(output, real_labels)
            g_loss.backward()
            # 这次训练生成器 通过将判别器的输出与真实标签计算损失并反向传播

            optimizer_G.step()
            g_total_loss += g_loss.item()
            # 更新权重并累加总损失

        d_avg_loss = d_total_loss / len(data_loader)
        g_avg_loss = g_total_loss / len(data_loader)
        # 计算判别器和生成器在当前epoch的平均损失

        print(f"Epoch: {epoch+1}, d_loss: {d_avg_loss:.4f}, g_loss: {g_avg_loss:.4f}")

        with open(train_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'Epoch': epoch+1,'d_loss': d_avg_loss,'g_loss': g_avg_loss})
            # 将当前epoch的损失写入CSV文件 便于后续分析和绘图

    torch.save(generator.state_dict(), 'generator.pt')
    torch.save(discriminator.state_dict(), 'discriminator.pt')

def save_images(epoch,generator,device):

    noise = torch.randn(64,100).to(device)
    fake_images = generator.forward(noise).view(-1,1,28,28)
    fake_images = fake_images.cpu().detach().numpy()
    # 使用生成器生成64张假图像 因为numpy只能处理cpu上的张量 所以要移回cpu

    fig, ax = plt.subplots(8,8,figsize=(8,8))
    for i in range(8):
        for j in range(8):
            ax[i,j].imshow(fake_images[i * 8 + j][0], cmap='gray')
            ax[i,j].axis('off')
    # 创建8x8的子图网格 每个子图显示一张生成的图像 并关闭坐标轴显示

    plt.savefig(f'images/epoch_{epoch}.png')
    plt.close()
    # 保存生成的图像网格为PNG文件

train()
