import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

def generate_mnist_data_set():
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# Batch size during training
batch_size = 128
# Size of z latent vector (i.e. size of generator input)
latent_vec_size = 30  # TODO
# Size of feature maps in generator and discriminator
num_channels = 16
# number of input channels
input_channels = 1
# Number of training epochs
num_epochs = 30
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyper-param for Adam optimizers
beta1 = 0.5
# spatial size of the input image
image_size = 28
# number of images to test the generator after every 500 iterations
num_test_samples=24


def tensor_to_plt_im(im: torch.Tensor):
    return im.permute(1, 2, 0)


class AutoEncoderMNIST(nn.Module):
    # def __init__(self):
        # super(AutoEncoderMNIST, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(input_channels * image_size * image_size, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 12),
        #     nn.ReLU(True),
        #     nn.Linear(12, latent_vec_size))
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_vec_size, 12),
        #     nn.ReLU(True),
        #     nn.Linear(12, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 128),
        #     nn.ReLU(True), nn.Linear(128, input_channels * image_size * image_size),
        #     nn.Unflatten(1, (input_channels, image_size, image_size)),
        #     nn.Tanh())
    def __init__(self):
        super(AutoEncoderMNIST, self).__init__()
        self.encoder = nn.Sequential(
            # input size is 3 x 28 x 28
            nn.Conv2d(input_channels, num_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # size (num_channels*2) x 14 x 14
            nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # size (num_channels*4) x 7 x 7
            nn.Flatten(),
            # size (396,)
            nn.Linear(num_channels * 4 * 7 * 7, 396),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(396, latent_vec_size),  # output size (latent_vec_size,)
        )
        self.decoder = nn.Sequential(
            # size (latent_vec_size,)
            nn.Linear(latent_vec_size, 396),
            nn.ReLU(True),
            # size (396,)
            nn.Linear(396, num_channels * 4 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (num_channels*4, 7, 7)),
            # size (num_channels*4) x 7 x 7
            nn.ConvTranspose2d(num_channels * 4, num_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels*2),
            nn.ReLU(True),
            # size (num_channels) x 14 x 14
            nn.ConvTranspose2d(num_channels * 2, input_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # the final output image size. (num_input_channels) x 28 x 28
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def calc_kurtosis(t, mean, std):
    """
    Computes the kurtosis of a :class:`Tensor`
    """
    return torch.mean(((t - mean) / std) ** 4)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(net: AutoEncoderMNIST, dataloader, criterion=nn.MSELoss()):
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            batch = data[0].to(device)  # Format batch

            # Forward pass batch through AutoEncoder
            enc_output = net.encoder(batch)  # latent vector that is being outputted by the encoder
            image_AE_output = net(batch)  # image that is being outputted by the whole net

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            err = criterion(image_AE_output, batch)

            # loss function in order to force latent space into normal standard distribution
            mean, var = torch.mean(enc_output), torch.var(enc_output)
            kurtosis = calc_kurtosis(enc_output, mean, var)
            err += (mean ** 2 + (var - 1) ** 2 + (kurtosis - 3) ** 2)

            err.backward()  # perform back-propagation
            optimizer.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss %.4f\t' % (epoch+1, num_epochs, i, len(dataloader), err.item()))

            # Check how the generator portion of the auto-encoder is doing by saving it's output on fixed_noise
            if i % 500 == 0:
                with torch.no_grad():
                    fixed_noise = torch.randn(num_test_samples, latent_vec_size, device=device)
                    fake = net.decoder(fixed_noise).detach().cpu()
                plt.imshow(tensor_to_plt_im(vutils.make_grid(fake)))
                plt.show()
    torch.save(net.state_dict(), './auto_encoder_mnist')


def test_AE_novel_samples(path, num_tests):
    """
    this function tests a pre-trained auto-encoder by feeding it with random vectors form the the
     standard-distribution over the latent space
    """
    AE = AutoEncoderMNIST()
    AE.load_state_dict(torch.load(path))
    AE.eval()
    img_lst = []
    for i in range(num_tests):
        noise = torch.randn(2 * batch_size, latent_vec_size, device=device)
        with torch.no_grad():
            im = AE.decoder(noise).detach().cpu()
        img_lst.append(im)
        plt.imshow(tensor_to_plt_im(img_lst[-1][-1]), cmap='gray')
        plt.show()
        time.sleep(2)


if __name__ == '__main__':
    test_AE_novel_samples('./auto_encoder_mnist', 10)
    AE = AutoEncoderMNIST().to(device)
    dl = generate_mnist_data_set()
    AE.apply(weights_init)

    # Create a batch of latent vectors to check the generator's progress
    train(AE, dataloader=dl)
