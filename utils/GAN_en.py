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
            # Map the random noise to higher dimensions and gradually extract features.
            # LeakyReLU is used to prevent neuron death.
            # The Tanh activation function is applied to compress the output to the range of -1 to 1,
            # ensuring that the pixel values of the image fall within this range.
            # This increases robustness and reduces the likelihood of convergence difficulties
            # due to vanishing gradients.
        )

    def forward(self, x):
        return self.model(x)
        # Forward propagation.

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
            # Map the input image (real or generated)
            # to a binary classification output through multiple linear transformations.
            # Extract global features by mapping to different dimensions and using LeakyReLU activation.
            # Finally, apply the Sigmoid function to map the output to a range of 0-1,
            # representing the probability that the image is real.
        )

    def forward(self, x):
        return self.model(x)
        # Forward propagation.
