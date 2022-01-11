# NOTE: This is all only regarding the implementation, the why and how are explained in the paper referenced

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log2

#Factors below indicate the output size required, ie 512 for the first 4, then 256, then 128 then 64 etc.
factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

#Equalized learning rate for the conv2d
class EQConv2D(nn.Module):
    #Gain is for the initialization constant of 2
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain/(kernel_size**2) * in_channels) ** 0.5
        #Bias should not be scaled with it, so it should be declared here as well
        self.bias = self.conv.bias
        self.conv.bias = None
        
        #initialize the conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
    
    #For each forward feed through it and reshaping bias for self.conv
    def forward(self, x):
        return self.conv(x*self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

#For the vector normalization in generator
class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        #Torch.mean of every pixel value squared, across the channels (which is in dim 1 cause dim 0 is just all the examples), 
        # then we do keepdim=true so the elementwise division will work, finally add episolon on all of this
        return x/torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

#Conv block will be 2 3x3's
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = EQConv2D(in_channels, out_channels)
        self.conv2 = EQConv2D(out_channels, out_channels)
        self.LRelu = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.LRelu(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.LRelu(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x

#Generator will use pixelnorm as well, discriminator will not
class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            #Decided not to use EQconv2d weighting here, paper says to use it
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0), # 1x1 -> 4x4
            nn.LeakyReLU(0.2),
            EQConv2D(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        self.initial_rgb = EQConv2D(in_channels, img_channels, kernel_size=1, stride=1, padding=0)

        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList([self.initial_rgb])

        for i in range(len(factors) -1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(EQConv2D(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upscale, generated):
        #We want output to be between -1 and 1 therefore tanh
        return torch.tanh(alpha * generated + (1 - alpha) * upscale)

    def forward(self, x, alpha, steps): #if steps are 0, 4x4, steps is 1 then 8x8 etc.
        out = self.initial(x) #4x4
        if steps == 0:
            return self.initial_rgb(out)
        
        for step in range(steps):
            #upsample before running through prog blocks
            upscale = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscale)

        final_upscale = self.rgb_layers[steps-1](upscale)
        final_out = self.rgb_layers[steps](out)

        return self.fade_in(alpha, final_upscale, final_out)


class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.Lrelu = nn.LeakyReLU(0.2)

        for i in range(len(factors)-1, 0, -1):
            conv_in_c = int(in_channels* factors[i])
            conv_out_c = int(in_channels* factors[i-1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c, use_pixelnorm=False))
            self.rgb_layers.append(EQConv2D(img_channels, conv_in_c, kernel_size=1, stride=1, padding=0))

        #This is for the 4x4 resolution at the end, named the same to match with generator even though it goes at the end
        self.inital_rgb = EQConv2D(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        #Block for the 4x4
        self.final_block = nn.Sequential(
            EQConv2D(in_channels+1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            EQConv2D(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyRelu(0.2),
            EQConv2D(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )

    def fade_in(self, alpha, downscaled, out):
        # Used to fade in downscaled using avg pooling and output from CNN
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        #Take the std from each example (across all channels) then repeat for a single channel and concat with the image
        #Allows discriminator to learn variance in the batch/img
        batch_stats = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_stats], dim=1)
    
    def forward(self, x, alpha, steps):
        #Have to index in reverse order for discrmiinator while keeping the property of step 0 = 4x4, step 1 = 8x8 etc.
        cur_steps = len(self.prog_blocks) - steps
        out = self.Lrelu(self.rgb_layers[cur_steps](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0]-1)

        #Because prog blocks also downscaled, we have to use the rgb layers from the previous/smaller size reason we do +1
        downscaled = self.Lrelu(self.rgb_layers[cur_steps +1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_steps](out))

        #Done first between the downscaled and the input
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_steps+1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


if __name__ == "__main__":
    Z_DIM = 100
    IN_CHANNELS = 256
    gen = Generator(Z_DIM, IN_CHANNELS, img_channels=3)
    critic = Discriminator(IN_CHANNELS, img_channels=3)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(img_size / 4))
        x = torch.randn((1, Z_DIM, 1, 1))
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At img size: {img_size}")