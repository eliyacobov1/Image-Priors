import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt

path = './auto_encoder_mnist_loss_modified with masking'


class AddGaussianNoise(object):
    """
    this class is used in order to add gaussian noise to images
    """
    def __init__(self, mean=0., std=0.75):
        self.std = std
        self.mean = mean
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class AddMaskPaintingNoise(object):
    """
    this class is used in order to add in-painting mask noise to images
    """
    def __call__(self, tensor):
        mask = torch.ones(tensor.size())
        mask[..., 15:] = 0
        return tensor * mask


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

noiser = AddGaussianNoise()
in_painter = AddMaskPaintingNoise()


def generate_mnist_data_set():
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# Batch size during training
batch_size = 128
# Size of z latent vector (i.e. size of generator input)
latent_vec_size = 12  # TODO
# Size of feature maps in generator and discriminator
num_channels = 24
# number of input channels
input_channels = 1
# Number of training epochs
num_epochs = 20
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
            nn.Linear(num_channels * 4 * 7 * 7, 396),
            # size (396,)
            nn.BatchNorm1d(396),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(396, latent_vec_size),  # output size (latent_vec_size,)
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            # size (latent_vec_size,)
            nn.Linear(latent_vec_size, 396),
            nn.ReLU(True),
            # size (396,)
            nn.Linear(396, num_channels * 4 * 7 * 7),
            nn.BatchNorm1d(num_channels * 4 * 7 * 7),
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


def train(net: AutoEncoderMNIST, dataloader, criterion=nn.MSELoss(), modify_loss=False,
          add_noise=False, add_mask=False):
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            batch = data[0].to(device)  # Format batch
            
            # Forward pass batch through AutoEncoder
            input_batch = noiser(batch) if add_noise else(in_painter(batch) if add_mask else batch)
            image_AE_output = net(input_batch)
            enc_output = net.encoder(input_batch)
            
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            err = criterion(image_AE_output, batch)
            
            if modify_loss:
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
                    plt.imshow(tensor_to_plt_im(vutils.make_grid(input_batch[:24])), cmap='gray')
                    plt.show()
                    plt.imshow(tensor_to_plt_im(vutils.make_grid(image_AE_output[:24])), cmap='gray')
                    plt.show()
    torch.save(net.state_dict(), f'./auto_encoder_mnist{"_loss_modified" if modify_loss else ""}' +
               f'{" with noise" if add_noise else("with masking" if add_mask else "")}')


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
        noise = torch.sigmoid(torch.randn(2 * batch_size, latent_vec_size, device=device))
        with torch.no_grad():
            im = AE.decoder(noise).detach().cpu()
        img_lst.append(im)
        plt.imshow(tensor_to_plt_im(img_lst[-1][-1]), cmap='gray')
        plt.show()
        time.sleep(2)
    return img_lst


def calc_posterior_distribution(I, Ic, AE, sigma=.75):
    """
    calculate the posterior distribution of the form: -logP(Ic|I)-logP(I) = (Ic-I)^2/2*sigma^2 + ||AE(I)-I||
    :param I: image from the original image distribution
    :param Ic: corrupted image to be restored
    :param AE: a pre-trained auto encoder
    :param sigma:
    """
    return torch.norm(Ic-I)/sigma**2 + torch.norm(AE(I) - I)


def reconstruct_image(Ic, AE, dataloader):
    """
    this function returns the most suitable candidate form the given image distribution to be the reconstructed
    image of the given image
    """
    posterior_distribution_minimizer = [float('inf'), None]
    for data in dataloader:
        batch = data[0].to(device)
        for i in range(batch.shape[0]):
            I = batch[i:i+1]
            posterior_distribution = calc_posterior_distribution(I, Ic, AE)
            if posterior_distribution < posterior_distribution_minimizer[0]:
                posterior_distribution_minimizer[0] = posterior_distribution
                posterior_distribution_minimizer[1] = I
    return posterior_distribution_minimizer[1]


def test_AE_reconstruction(path, num_tests, dataloader, corruption='noise'):
    AE = AutoEncoderMNIST()
    AE.load_state_dict(torch.load(path))
    AE.eval()
    
    batch = next(iter(dataloader))[0].to(device)
        
    for i in range(num_tests):
        img = batch[i:i+1]
        noised_image = noiser(img) if corruption == 'noise' else in_painter(img)
        reconstructed_image = reconstruct_image(noised_image, AE, dataloader)
        plt.imshow(tensor_to_plt_im(torch.clip(noised_image[0], 0, 1)), cmap='gray')
        plt.show()
        plt.imshow(tensor_to_plt_im(reconstructed_image.detach()[0]), cmap='gray')
        # plt.imshow(tensor_to_plt_im(AE(noised_image).detach()[0]), cmap='gray')
        plt.show()

    
def scatter_2d_plane_form_latent_space(dataloader):
    """
    this function scatters the 2D plots of index-pairings of 384 encoded images
    """
    AE = AutoEncoderMNIST()
    AE.load_state_dict(torch.load(path))
    AE.eval()
    latent_vec_list, i = [], 0
    
    # ----------- concatenate the 3 first batches -----------
    for data in dataloader:
        batch = data[0].to(device)  # Format batch
        latent_vec_list.append(AE.encoder(batch).detach())
        i += 1
        if i >= 3:  # 3 first batches consist of 384 image vectors
            break
    
    concat_latent_vectors = torch.cat((latent_vec_list[0], latent_vec_list[1], latent_vec_list[2]), 0)
    index_pairs = [(1, 5), (2, 7), (3, 9), (4, 11), (0, 8)]
    
    # ---------------------- plot the 2D scatter if each of the index pairings above ----------------------
    for p in index_pairs:
        plt.scatter(concat_latent_vectors[:, p[0]], concat_latent_vectors[:, p[1]],
                    label=f"2D scatter of coordinates {p[0]} vs {p[1]}")
        plt.legend()
        plt.title(f"2D scatter of latent vectors for 300 images, coordinates {p[0]} vs {p[1]}")
        plt.ylabel(f"coordinate {p[1]}")
        plt.xlabel(f"coordinate {p[0]}")
        plt.savefig(f'2d_scatter_{p[0]}_vs_{p[1]}.png')
        plt.show()


if __name__ == '__main__':
    # test_AE_novel_samples(path, 10)
    AE = AutoEncoderMNIST().to(device)
    dl = generate_mnist_data_set()
    test_AE_reconstruction(path=path, num_tests=2, dataloader=dl, corruption='in-paint')
    # scatter_2d_plane_form_latent_space(dl)
    AE.apply(weights_init)
    # Create a batch of latent vectors to check the generator's progress
    # train(AE, dataloader=dl, modify_loss=True)
