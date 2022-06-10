import torch.nn as nn

import torch

import torch.nn.functional as F

class Feedforward(nn.Module):

    def __init__(self,in_channel,out_channel):

        super(Feedforward,self).__init__()

        self.in_channel = in_channel

        self.out_channel = out_channel

        self.linear1 = nn.Linear(in_channel, in_channel)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.linear2 = nn.Linear(in_channel,out_channel)

    def forward(self,input):

        input =  self.linear1(input)

        output = self.linear2(self.relu(input))

        return output


class GBlock(nn.Module):

    def __init__(self,in_channel,out_channel):

        super(GBlock,self).__init__()

        self.in_channel = in_channel

        self.out_channel = out_channel

        self.batchnorm = nn.BatchNorm2d(in_channel)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.mask_predict = nn.Sequential(

            nn.Conv2d(in_channels = in_channel, out_channels = 100, kernel_size = 3, stride=1, padding=1),

            nn.BatchNorm2d(100),

            nn.ReLU(),

            nn.Conv2d(in_channels = 100, out_channels = 1, kernel_size = 1, stride=1, padding=0)
        
        )

            
        self.conv = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 1, stride=1, padding=0)

        self.conv1 = nn.Conv2d(in_channel,out_channel, 3, 1, 1)

        self.mlp1 = Feedforward(256,out_channel)

        self.mlp2 = Feedforward(256,out_channel)

    def forward(self,image_feat,text_feat):

        image_feat_in = F.interpolate(image_feat,scale_factor=2, mode='bilinear', align_corners=True)

        logits = self.mask_predict(image_feat_in)

        mask = torch.sigmoid(logits)

        normed_input = self.batchnorm(image_feat_in)

        size = normed_input.size()

        gamma = self.mlp1(text_feat)

        beta = self.mlp2(text_feat)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1).expand(size)

        beta = beta.unsqueeze(-1).unsqueeze(-1).expand(size)

        gamma_in = gamma * mask
         
        beta_in = beta * mask

        output = gamma_in * normed_input  + beta_in

        residue = None

        if self.in_channel != self.out_channel:

           residue = self.conv(normed_input)

           output = self.conv1(output)

        else:

           residue = normed_input

        out = residue + self.gamma * output

        return out


class Generator(nn.Module):

    def __init__(self,input_dim,noise_dim=100):

        super(Generator, self).__init__()

        self.input_dim = input_dim

        self.fc = nn.Linear(noise_dim,input_dim*8*4*4)

        self.channel_list = [input_dim*8,input_dim*8,input_dim*8,input_dim*8,input_dim*8,input_dim*4,input_dim*2,input_dim]

        self.tanh = nn.Tanh()

        self.batchnorm = nn.BatchNorm2d(input_dim)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv =  nn.Conv2d(input_dim, 3, 3, 1, 1)
            
        self.GBlocks = nn.ModuleList([GBlock(self.channel_list[i],self.channel_list[i+1]) for i in range(7)])

    def forward(self,image,text):

        image_feat = self.fc(image)

        image_feat = image_feat.view(image.size(0), 8 * self.input_dim, 4, 4)

        for i in range(7):

            image_feat = self.GBlocks[i](image_feat,text)

        image_feat = self.batchnorm(image_feat)    

        image_feat = self.relu(image_feat)

        out = self.conv(image_feat)   

        out = self.tanh(out)

        return out


class Discriminator(nn.Module):

    def __init__(self,input_dim):

        super(Discriminator, self).__init__()

        self.conv = nn.Conv2d(in_channels = 3, out_channels =input_dim, kernel_size = 3, stride=1, padding=1)

        self.channel_list = [input_dim,input_dim*2,input_dim*4,input_dim*8,input_dim*16,input_dim*16,input_dim*16]

        self.dblocks = nn.ModuleList([DBlock(self.channel_list[i],self.channel_list[i+1])for i in range(6)])
    
    def forward(self,input):

        image_feat = self.conv(input)

        for i in range(6):

            image_feat = self.dblocks[i](image_feat)

           
        out = image_feat

        return out  


class DBlock(nn.module):


    def __init__(self,in_channel,out_channel):

        super(DBlock, self).__init__()

        self.in_channel = in_channel

        self.out_channel = out_channel

        self.conv_r = nn.Sequential(

            nn.Conv2d(in_channel,out_channel, 4, 2, 1, bias=False),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),

            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv1 = nn.Conv2d(in_channel,out_channel,1,1,0)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,image_feat):

        residue = None

        if self.in_channel != self.out_channel:

           residue = self.conv1(image_feat)

        else:

           residue = image_feat

        image_feat = self.conv(image_feat)

        residue = F.avg_pool2d(residue,2)

        out = residue + self.gamma * image_feat

        return out
         





class DBlock(nn.module):

    def __init__(self,in_channel,out_channel):

        self.in_channel = in_channel

        self.out_channel = out_channel

        self.conv = nn.Conv2d()

        self.conv2 = nn.Conv2d()

    def forward(self,image):

        output = None

        residue = None

        if self.in_channel != self.out_channel:

             output = self.conv(image)

             residue = self.conv1(image)

        else:

              output = image     

              residue = self.conv1(image)
        output = F.avergpol(output,2)

        out = output + self.gamma * residue

        return out 

