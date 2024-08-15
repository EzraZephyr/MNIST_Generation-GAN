import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.GAN_en import Generator, Discriminator
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def train():

    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # If CUDA is available, use it for training.

    os.makedirs("images", exist_ok=True)
    # Create a directory to store images generated during training.

    generator = Generator(input_dim=100, output_dim=784).to(device)
    discriminator = Discriminator(input_dim=784).to(device)
    # Define the generator and discriminator.
    # Since MNIST images are 28x28 single-channel images, the input dimension is 1x28x28 = 784.

    loss_fn = nn.BCELoss()
    # Use binary cross-entropy loss.

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # Define Adam optimizers for the generator and discriminator respectively.
    # The betas values are empirical, helping the model converge faster and stabilize the training process.

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # Define image preprocessing steps: convert images to tensors and normalize them to match the generator's output range of -1 to 1.

    mnist = datasets.MNIST('../data', train=True, download=True, transform=transform)
    data_loader = DataLoader(mnist, batch_size=64, shuffle=True)
    # Load the MNIST dataset and create a data loader.

    train_csv = '../mnist_train.csv'
    with open(train_csv, 'w', newline='') as f:
        fieldnames = ['Epoch', 'd_loss', 'g_loss']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Create a CSV file to record the training loss.

    num_epochs = 100
    for epoch in range(num_epochs):

        if epoch % 10 == 0:
            save_images(epoch, generator, device)
        # Save training images every 10 epochs to observe training progress.

        d_total_loss = 0.0
        g_total_loss = 0.0

        for i, (real_images, _) in enumerate(data_loader):

            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1).to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            # Get the number of images in each batch and reshape them to (batch_size, 784).
            # Generate corresponding real labels (1) and fake labels (0), and move them to the device.

            optimizer_D.zero_grad()
            # Clear gradients.

            outputs = discriminator(real_images)
            d_loss_real = loss_fn(outputs, real_labels)
            d_loss_real.backward()
            # Feed real images into the discriminator, calculate the loss, and perform backpropagation.

            noise = torch.randn(batch_size, 100).to(device)
            # Generate random noise.

            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = loss_fn(outputs, fake_labels)
            d_loss_fake.backward()
            # Convert random noise into fake images using the generator,
            # then feed them into the discriminator to calculate the loss and perform backpropagation.
            # Detach the generated images from the generator to prevent the generator's parameters
            # from being updated erroneously during backpropagation.

            optimizer_D.step()
            d_total_loss += d_loss_real.item() + d_loss_fake.item()
            # Update weights and accumulate total loss.

            optimizer_G.zero_grad()

            output = discriminator(fake_images)
            g_loss = loss_fn(output, real_labels)
            g_loss.backward()
            # Train the generator by calculating the loss between the discriminator's output and real labels,
            # then perform backpropagation.

            optimizer_G.step()
            g_total_loss += g_loss.item()
            # Update weights and accumulate total loss.

        d_avg_loss = d_total_loss / len(data_loader)
        g_avg_loss = g_total_loss / len(data_loader)
        # Calculate the average loss for the discriminator and generator for the current epoch.

        print(f"Epoch: {epoch+1}, d_loss: {d_avg_loss:.4f}, g_loss: {g_avg_loss:.4f}")

        with open(train_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'Epoch': epoch + 1, 'd_loss': d_avg_loss, 'g_loss': g_avg_loss})
            # Write the losses for the current epoch to the CSV file for later analysis and plotting.

    torch.save(generator.state_dict(), '../model/generator.pt')
    torch.save(discriminator.state_dict(), '../model/discriminator.pt')

def save_images(epoch, generator, device):

    noise = torch.randn(64, 100).to(device)
    fake_images = generator.forward(noise).view(-1, 1, 28, 28)
    fake_images = fake_images.cpu().detach().numpy()
    # Use the generator to create 64 fake images.
    # Move them back to the CPU since NumPy can only handle tensors on the CPU.

    fig, ax = plt.subplots(8, 8, figsize=(8, 8))
    for i in range(8):
        for j in range(8):
            ax[i, j].imshow(fake_images[i * 8 + j][0], cmap='gray')
            ax[i, j].axis('off')
    # Create an 8x8 grid of subplots, with each subplot displaying a generated image, and turn off the axis.

    plt.savefig(f'images/epoch_{epoch}.png')
    plt.close()
    # Save the image grid as a PNG file.

train()
