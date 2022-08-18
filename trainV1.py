import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(0) # Set for testing purposes, please do not change!
from Networks import Generator, Critic
from CustomDataLoader import CustomDataLoader, Custom_real_DataLoader

def show_tensor_images(image_tensor, num_images=8, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def make_grad_hook():
    '''
    Function to keep track of gradients for visualization purposes,
    which fills the grads list when using model.apply(grad_hook).
    '''
    grads = []
    def grad_hook(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            grads.append(m.weight.grad)
    return grads, grad_hook

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)


n_epochs = 10
display_step = 20
batch_size = 8
lr = 0.001
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 3
device = 'cuda'
height, width = 224, 224
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# dataset
dataset1 = CustomDataLoader(fmri_file='./data/fmri/sub-01_perceptionNaturalImageTraining_original_VC.h5',
                            imagenet_folder='./data/images/training',
                            transform=trans)
dataset2 = CustomDataLoader(fmri_file='./data/fmri/sub-02_perceptionNaturalImageTraining_original_VC.h5',
                            imagenet_folder='./data/images/training',
                            transform=trans)
dataset3 = CustomDataLoader(fmri_file='./data/fmri/sub-03_perceptionNaturalImageTraining_original_VC.h5',
                            imagenet_folder='./data/images/training',
                            transform=trans)
dataset_real = Custom_real_DataLoader('data/real_images', transform=trans)
# dataloader
dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True)
dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True, drop_last=True)
dataloader3 = DataLoader(dataset3, batch_size=batch_size, shuffle=True, drop_last=True)
dataloader_real = DataLoader(dataset_real, batch_size=batch_size, shuffle=True, drop_last=True)
fmri_dim = torch.max(torch.tensor([next(iter(dataloader1))[1].shape[1], next(iter(dataloader2))[1].shape[1], next(iter(dataloader3))[1].shape[1]]))
gen = Generator(fmri_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
crit = Critic().to(device)
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

gen = gen.apply(weights_init)
crit = crit.apply(weights_init)
def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        inputs=mixed_images,
        outputs=mixed_scores,
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = ((1 - gradient_norm) * (1 - gradient_norm)).sum()/len(gradient)
    return penalty

def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    gen_loss = -crit_fake_pred.mean()

    return gen_loss

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''

    crit_loss = -(crit_real_pred - crit_fake_pred - c_lambda*gp).mean()
    return crit_loss



import matplotlib.pyplot as plt
scheduler = StepLR(gen_opt, step_size=3, gamma=0.2)
scheduler1 = StepLR(crit_opt, step_size=3, gamma=0.2)
cur_step = 0
generator_losses = []
critic_losses = []
upsampler = torch.nn.Upsample(size=fmri_dim, mode='nearest')
for epoch in tqdm(range(n_epochs)):
    # Dataloader returns the batches
    start = time.time()
    for i, data in enumerate(zip(dataloader_real,dataloader1,dataloader2)):

        real_images = data[0][0]
        cur_batch_size = batch_size
        real_images = real_images.float().to(device)
        for subject in [data[1], data[2]]:

            mean_iteration_critic_loss = 0
            sub_images = subject[0].float().to(device)
            sub_fmri = subject[1]
            sub_fmri = upsampler(sub_fmri.view(batch_size, 1, -1)).view(batch_size, -1)
            sub_fmri /= torch.max(torch.max(sub_fmri),torch.abs(torch.min(sub_fmri)))
            for _ in tqdm(range(crit_repeats)):

                ### Update critic ###
                crit_opt.zero_grad()
                fake_noise = sub_fmri.float().to(device)
                fake = gen(fake_noise)
                crit_fake_pred = crit(fake.detach())
                crit_real_pred = crit(real_images)
                epsilon = torch.rand(len(real_images), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(crit, real_images, fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                # Update gradients
                crit_loss.backward(retain_graph=True)
                # Update optimizer
                crit_opt.step()

            critic_losses += [mean_iteration_critic_loss]
            ### Update generator ###
            gen_opt.zero_grad()
            fake_noise_2 = sub_fmri.float().to(device)
            fake_2 = gen(fake_noise_2)
            crit_fake_pred = crit(fake_2)
            #because we want to predict fake images as reals, so we expect that loss of generator as negative as possible.
            gen_loss = get_gen_loss(crit_fake_pred)
            mse_loss = F.mse_loss(fake_2, sub_images, reduction='mean')
            tot_loss = gen_loss + mse_loss
            tot_loss.backward()

            # Update the weights
            gen_opt.step()

            # Keep track of the average generator loss
            # gen_loss.item()
            generator_losses += [tot_loss.item()]

            ### Visualization code ###
            if cur_step % display_step == 0 and cur_step > 0:
                print(f'tot loss is {tot_loss}')
                print(f'gen loss is {gen_loss}')
                print(f'mse loss is {mse_loss}')
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                crit_mean = sum(critic_losses[-display_step:]) / display_step
                print(f"Epoch {epoch}, step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
                show_tensor_images(fake_2)
                show_tensor_images(sub_images)
                step_bins = 20
                num_examples = (len(generator_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss"
                )
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Critic Loss"
                )
                plt.legend()
                plt.show()

            cur_step += 1

    end = time.time()
    scheduler.step()
    scheduler1.step()
    print(f'time for each epoch is {end - start}')