#Chris Metzler
#2/13/20

import argparse
import os
import numpy as np

import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch_dct as dct #https://github.com/zh217/torch-dct
from matplotlib.pyplot import imsave


import torch.nn as nn
import torch
import xcorr2 as xcorr2
import time

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")#Only used in loading data, nut used for reconstructions
parser.add_argument("--lr", type=float, default=0.02, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

show_results = False



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

for UseFashionMnist in [False,True]:
    # Loss function
    loss = torch.nn.MSELoss()
    if UseFashionMnist:
        generator = torch.load('./Generators/FashionMNIST_DCGAN_generator_epoch199.pth')
    else:
        generator = torch.load('./Generators/MNIST_DCGAN_generator_epoch199.pth')

    if cuda:
        generator.cuda()
        loss.cuda()

    if UseFashionMnist:
        dataloader1 = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                "./data/fashion_mnist",
                train=False,#Use the test set
                # train=True,#Use the same set as used for training
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )
    else:
        dataloader1 = torch.utils.data.DataLoader(
            datasets.MNIST(
                "./data/mnist",
                train=False,#Use the test set
                # train=True,#Use the same set as used for training
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )



    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for numImages in [2,3,4]:

        for measurement_type in ['Gaussian','CDP_complex','Fourier']:
            if measurement_type=='Fourier':
                def forward_model(x):
                    Ax = []
                    [n_batch, n_c, ha, wa] = x.shape
                    y = torch.zeros(size=(n_batch, n_c, 2 * ha - 1, 2 * wa - 1),device='cuda')
                    for i in range(n_c):
                        y[:,i,:,:] = xcorr2.xcorr2_torch(x[:,i:i+1,:,:])
                    return y, Ax
            elif measurement_type=='Fourier_explicit':
                dtype = torch.cuda.FloatTensor
                def forward_model(x):
                    [n_batch, n_c, ha, wa] = x.shape
                    Ax = torch.zeros(size=(n_batch, n_c, 2 * ha - 1, 2 * wa - 1,2),device='cuda')
                    y = torch.zeros(size=(n_batch, n_c, 2 * ha - 1, 2 * wa - 1),device='cuda')
                    for i in range(n_c):
                        tmp2, Ax_r, Ax_i = xcorr2.FourierMod2(x[:,i:i+1,:,:])  # This is a 2D Fourier transform
                        Ax[:, i, :, :, 0] = Ax_r
                        Ax[:, i, :,:, 1] = Ax_i
                        y[:, i, :,:] = tmp2
                    return y, Ax
            elif measurement_type=='CDP_complex':#CDP that supports negative values on the SLM
                dtype = torch.cuda.FloatTensor
                Ms=[]
                num_patterns = 4
                for k in range(num_patterns):
                    phase = 2*np.pi*np.random.rand(opt.img_size,opt.img_size)
                    M_k_real = np.cos(phase)
                    M_k_imag = np.sin(phase)
                    Ms.append(Tensor(np.stack([M_k_real,M_k_imag],axis=-1)).type(dtype))#Four different random patterns
                def forward_model(x,Ms=Ms):
                    [n_batch, n_c, ha, wa] = x.shape
                    Ax = torch.zeros(size=(n_batch, n_c, num_patterns*opt.img_size**2, 2), device='cuda')
                    y = torch.zeros(size=(n_batch, n_c, num_patterns*opt.img_size**2),device='cuda')
                    for i in range(n_c):
                        for k in range(num_patterns):
                            tmp_r = Ms[k][:,:,0] * x
                            tmp_i = Ms[k][:, :, 1] * x
                            tmp = torch.stack([tmp_r,tmp_i],dim=-1)
                            tmp2, Ax_r, Ax_i = xcorr2.FourierMod2_nopad_complex(tmp)#This is a 2D Fourier transform
                            Ax[:,i,k*opt.img_size**2:(k+1)*opt.img_size**2,0]=Ax_r.view(-1,)
                            Ax[:,i,k*opt.img_size**2:(k+1)*opt.img_size**2,1]=Ax_i.view(-1,)
                            y[:,i,k*opt.img_size**2:(k+1)*opt.img_size**2] = tmp2.view(-1,)
                    Ax = Ax.view(-1, n_c, int(np.sqrt(num_patterns)) * ha, int(np.sqrt(num_patterns)) * wa,2)
                    y=y.view(-1,n_c,int(np.sqrt(num_patterns))*ha,int(np.sqrt(num_patterns))*wa)
                    return y, Ax
            elif measurement_type=='Gaussian':
                dtype = torch.cuda.FloatTensor
                A = Tensor(np.random.randn(4*opt.img_size**2,opt.img_size**2,2)).type(dtype)
                # A = Tensor(np.random.randn(16 * opt.img_size ** 2, opt.img_size ** 2,2)).type(dtype)
                def forward_model(x,A=A):
                    [n_batch, n_c, ha, wa] = x.shape
                    [m,n,_]=A.shape
                    assert n==ha*wa, "Dimensions don't match"
                    assert np.sqrt(m)==int(np.sqrt(m)), "The square root of m must be an integer"
                    Ax = torch.zeros(size=(n_batch, n_c, int(np.sqrt(m)), int(np.sqrt(m)),2), device='cuda')
                    y = torch.zeros(size=(n_batch, n_c, int(np.sqrt(m)),int(np.sqrt(m))),device='cuda')
                    for i in range(n_c):
                        Ax[:,i,:,:,0]=torch.mm(A[:,:,0],x[:,i,:,:].reshape(n_batch,ha*wa).transpose(1,0)).transpose(1,0).reshape(n_batch,int(np.sqrt(m)),int(np.sqrt(m)))
                        Ax[:,i,:,:,1]=torch.mm(A[:,:,1],x[:,i,:,:].reshape(n_batch,ha*wa).transpose(1,0)).transpose(1,0).reshape(n_batch,int(np.sqrt(m)),int(np.sqrt(m)))
                        y[:,i,:,:] = torch.abs(Ax[:,i,:,:,0])**2 + torch.abs(Ax[:,i,:,:,1])**2
                    return y, Ax
            else:
                print('Measurement not supported')
                break

            for i, (imgs, _) in enumerate(dataloader1):

                if i>=10:
                    break

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                x_GT = real_imgs[i :i + numImages, :, :, :]  # Should be 1x1x32x32

                r = 0
                for l in range(numImages):
                    r = r + forward_model(x_GT[l:l+1,:,:,:])[0]

                SNR = 50
                if SNR!=np.inf:
                    w = torch.randn(r.shape).cuda()
                    w = w/w.norm()
                    w = w * r.norm()/np.sqrt(SNR)
                    r = r + w

                best_loss = np.inf
                num_attempts=5
                t_start = time.time()
                for attempt in range(num_attempts):
                    # Sample noise as generator input
                    z_variables = []
                    x_origs = []
                    optimizers=[]
                    x_noGAN_variables = []
                    for l in range(numImages):
                        z_variables.append(Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))),requires_grad=True))
                        x_noGAN_variables.append(Variable(Tensor(np.random.normal(0,1,(1,1,opt.img_size,opt.img_size))),requires_grad=True))
                        x_origs.append(generator(z_variables[-1]))#My code now assumes all the variables use the same generator
                        optimizers.append(torch.optim.Adam([z_variables[-1]], lr=opt.lr, betas=(opt.b1, opt.b2)))

                    K1=2000
                    reset_every = 1e10
                    losses = np.zeros((K1,))
                    for j in range(K1):
                        if j % reset_every == 0:  # Every optimizer_reset_the optimizer: resets momentum terms.
                            optimizers = []
                            for l in range(numImages):
                                optimizers.append(torch.optim.Adam([z_variables[l]], lr=opt.lr, betas=(opt.b1, opt.b2)))
                        for l in range(numImages):
                            optimizers[l].zero_grad()

                        # Generate a batch of images
                        x_pred = Tensor(size=x_GT.shape)
                        for l in range(numImages):
                            x_pred[l:l+1,:,:,:]=generator(z_variables[l])

                        r_pred = 0
                        for l in range(numImages):
                            r_pred = r_pred + forward_model(x_pred[l:l + 1, :, :, :])[0]

                        g_loss = loss(r_pred, r)
                        losses[j] = g_loss.cpu().data.numpy()

                        g_loss.backward()

                        optimizers[j%numImages].step()

                        if g_loss.cpu().data.numpy() < best_loss:
                            best_loss = g_loss.cpu().data.numpy()
                            r_pred_final = r_pred.clone().data
                            x_pred_final = x_pred.clone().data
                t_end=time.time()
                print(str(t_end-t_start))

                t_start = time.time()
                recon_noGAN = True
                best_loss_noGAN=np.inf
                num_attempts_noGAN=5
                if recon_noGAN:
                    for attempt in range(num_attempts_noGAN):
                        #Perform PR to estimate \sum_i x_i
                        x_sum_variable = Variable(Tensor(np.random.normal(0, 1, (1, 1, opt.img_size, opt.img_size))), requires_grad=True)

                        K2 = 2000
                        losses_noGAN = np.zeros((K2,))
                        for j in range(K2):
                            if j % reset_every == 0:  # Every optimizer_reset_the optimizer: resets momentum terms.
                                noGAN_optimizer = torch.optim.Adam([x_sum_variable], lr=opt.lr, betas=(opt.b1, opt.b2))
                            noGAN_optimizer.zero_grad()

                            r_pred_sum = forward_model(x_sum_variable[0:1, :, :, :])[0]

                            g_loss_noGAN_l = loss(r, r_pred_sum)
                            losses_noGAN[j] = g_loss_noGAN_l.cpu().data.numpy()

                            g_loss_noGAN_l.backward()
                            noGAN_optimizer.step()
                        x_sum_variable_fixed = x_sum_variable.clone().data

                        #Perform SS
                        def Psi(x):
                            X=dct.idct_2d(x,norm='ortho')#This will apply an inverse DCT of x
                            return X

                        alpha_DCT_variables = []
                        for l in range(numImages):
                            alpha_DCT_variables.append(Variable(Tensor(np.random.normal(0, 1, (1, 1, opt.img_size, opt.img_size))), requires_grad=True))


                        K3 = 2000
                        for j in range(K3):
                            if j % reset_every == 0:  # Every optimizer_reset_the optimizer: resets momentum terms.
                                no_GAN_optimizers = []
                                for l in range(numImages):
                                    no_GAN_optimizers.append(torch.optim.Adam([alpha_DCT_variables[l]], lr=opt.lr, betas=(opt.b1, opt.b2)))

                            for l in range(numImages):
                                no_GAN_optimizers[l].zero_grad()

                            x_sum_pred = 0
                            x_noGAN = []
                            L1_loss = 0
                            for l in range(numImages):
                                x_noGAN.append( Psi(alpha_DCT_variables[l]))  # Eventually this will be the DCT of the sparse coefficients alpha
                                x_sum_pred = x_sum_pred + x_noGAN[l]
                                L1_loss = L1_loss + alpha_DCT_variables[l].abs().mean()#The L2 loss is also normalized by default

                            my_lambda=1
                            L2_Loss = loss(x_sum_pred, x_sum_variable_fixed)
                            g_loss_noGAN = L2_Loss + my_lambda * L1_loss

                            losses_noGAN[j] = g_loss_noGAN.cpu().data.numpy()

                            g_loss_noGAN.backward()
                            no_GAN_optimizers[j % numImages].step()

                        #Calculate residual loss
                        r_pred_noGAN_total=0
                        for l in range(numImages):
                            r_pred_noGAN_total=r_pred_noGAN_total+forward_model(x_noGAN[l][0:1, :, :, :])[0]
                        this_loss = loss(r,r_pred_noGAN_total)

                        if this_loss.cpu().data.numpy() < best_loss_noGAN:
                            best_loss_noGAN = this_loss.cpu().data.numpy()
                            r_pred_noGAN_final = r_pred_noGAN_total.clone().data
                            x_pred_noGAN_final = torch.cat(x_noGAN).clone().data
                t_end = time.time()
                print(str(t_end - t_start))


                r_image = np.fft.fftshift(r[0,0,:,:].detach().cpu())
                r_pred_image = np.fft.fftshift(r_pred_final[0,0,:,:].detach().cpu())
                x_GT_images = x_GT[:,0,:,:].detach().cpu()
                x_pred_images = x_pred_final[:,0,:,:].detach().cpu()
                if recon_noGAN:
                    r_pred_noGAN_image = np.fft.fftshift(r_pred_noGAN_final[0, 0, :, :].detach().cpu())
                    x_pred_images_noGAN = x_pred_noGAN_final[:,0,:,:].detach().cpu()

                if show_results:
                    plt.subplot(2+recon_noGAN,numImages+1,1)
                    plt.imshow(r_image)
                    plt.title("True Measurement")
                    plt.subplot(2+recon_noGAN,numImages+1,numImages+2)
                    plt.imshow(r_pred_image)
                    plt.title("Fit Measurement w Prior")
                    if recon_noGAN:
                        plt.subplot(3,numImages+1,2*numImages+3)
                        plt.imshow(r_pred_noGAN_image)
                        plt.title("Fit Measurement wout Prior")
                    for l in range(numImages):
                        plt.subplot(2+recon_noGAN,numImages+1,l+2)
                        plt.imshow(x_GT_images[l])
                        plt.title("GT")
                        plt.subplot(2+recon_noGAN,numImages+1, numImages+1 + l+2)
                        plt.imshow(x_pred_images[l])
                        plt.title("Prediction")
                        if recon_noGAN:
                            plt.subplot(3, numImages + 1,  2*numImages +4 + l)
                            plt.imshow(x_pred_images_noGAN[l])
                            plt.title("Fit Measurement wout Prior")
                    plt.show()

                save_results=True
                if save_results:
                    if UseFashionMnist:
                        save_dir = './DemoResults/Fashion_' + measurement_type + "_numattempts" + str(num_attempts) + "_numImages" + str( numImages) + "_" + str(i) + "/"
                    else:
                        save_dir = './DemoResults/' + measurement_type +  "_numattempts"+str(num_attempts)+"_numImages"+str(numImages)+"_" + str(i) + "/"
                    os.makedirs(save_dir, exist_ok=True)

                    if recon_noGAN:
                        np.savez(save_dir + "results.npz", r_image=r_image, r_pred_image=r_pred_image,  r_pred_noGAN_image=r_pred_noGAN_image, x_GT_images=x_GT_images, x_pred_images_GAN=x_pred_images, x_pred_images_noGAN = x_pred_images_noGAN)
                    else:
                        np.savez(save_dir + "results.npz", r_image=r_image, r_pred_image_GAN=r_pred_image, x_GT_images=x_GT_images, x_pred_images_GAN=x_pred_images)

                    imsave(save_dir + 'TrueMeasurement.png', r_image)
                    imsave(save_dir + 'PredMeasurement.png', r_pred_image)
                    if recon_noGAN:
                        imsave(save_dir + 'noGanPredMeasurement.png', r_pred_noGAN_image)
                    for l in range(numImages):
                        imsave(save_dir + 'GT_' + str(l) +'.png', x_GT_images[l])
                        imsave(save_dir + 'Pred_' + str(l) +'.png', x_pred_images[l])
                        if recon_noGAN:
                            imsave(save_dir + 'noGanPred_' + str(l) +'.png', x_pred_images_noGAN[l])
                            imsave(save_dir + 'noGanPred_neg_' + str(l) + '.png', -x_pred_images_noGAN[l])

