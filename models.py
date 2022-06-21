import torch.nn as nn

import torch

import torch.nn.functional as F

import math

class Affine_Projection(nn.Module):
    
    def __init__(self,text_dim,img_length,img_channel,text_channel):

        super(Affine_Projection,self).__init__()        
        self.conv_text = nn.Conv1d(text_channel,img_channel,1)
        self.linear1 = nn.Linear(text_dim,text_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(text_dim,img_length)
        
    def forward(self,text_feat):

        input = self.relu(self.linear1(self.conv_text(text_feat)))

        return self.linear2(input)

class TextImageGate(nn.Module):
    
    def __init__(self,img_channel):
        
        super(TextImageGate,self).__init__()
        
        self.conv_mask = nn.Sequential(
        
            nn.Conv2d(img_channel,img_channel,(1,1)),
            nn.ReLU(),
            nn.Conv2d(img_channel,img_channel,(1,1)),
            nn.Tanh()      
        
        )
        
    def forward(self,image_input,encoded_input):
        
        conv_input = self.conv_mask(encoded_input)
        
        input = conv_input * encoded_input
        
        out = image_input + input
        
        return out
        
        
class TextImageEncoder(nn.Module):
    
    def __init__(self,img_length,img_width,text_channel,img_channel):
        
        super(TextImageEncoder,self).__init__()
        
        self.img_length = img_length
        
        self.img_width = img_width
        
        self.text_channel = text_channel
        
        self.img_channel = img_channel
        
        self.conv_text1 = nn.Conv1d(text_channel,img_channel,1)
        
        self.conv_text2 = nn.Conv1d(text_channel,img_channel,1) 
        
        self.conv_image1 = nn.Conv2d(img_channel,img_channel,(1,1))
        
        self.conv_image2 = nn.Conv2d(img_channel,img_channel,(1,1))
        
        self.conv_last = nn.Conv2d(img_channel,img_channel,(1,1))
        
        self.conv_out = nn.Conv2d(img_channel,img_channel,(1,1))
        
        self.image_norm1 = nn.InstanceNorm2d(img_channel)
        
        self.norm_last = nn.InstanceNorm2d(img_channel)
        
        self.relu1 = nn.ReLU()
        
        self.relu2 = nn.ReLU()
    
        
    def forward(self,img_feat,text_feat):
        
        bs = img_feat.shape[0]
        
        text_feat1 = self.conv_text1(text_feat)
            
        text_feat2 = self.conv_text2(text_feat)
        
        image_feat1 = self.image_norm1(self.conv_image1(img_feat))
        
        image_feat2 = self.relu1(self.conv_image2(img_feat))
        
        image_feat1 = image_feat1.contiguous().view(bs,self.img_channel,-1)
        
        image_feat1 = image_feat1.transpose(1,2)
              
        weight = torch.bmm(image_feat1,text_feat1)
        
        mask = torch.softmax(weight,dim = -1)
        
        mask = mask/math.sqrt(self.img_channel)
        
        text_feat2 = text_feat2.transpose(1,2)
        
        img_text_weight = torch.bmm(mask,text_feat2)
        
        img_text_weight = img_text_weight.transpose(1,2).contiguous().view(bs,self.img_channel,self.img_length,self.img_width)
        
        masked_weight = self.conv_last(img_text_weight)
        
        masked_weight = self.norm_last(masked_weight)
        
        output = masked_weight * image_feat2
        
        output = self.conv_out(output)
        
        output = self.relu2(output)
             
        return output
        
        
        
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

    def __init__(self,in_channel,out_channel,img_length,img_width,args):

        super(GBlock,self).__init__()
        
        self.args = args

        self.in_channel = in_channel

        self.out_channel = out_channel

        self.batchnorm = nn.BatchNorm2d(in_channel)

        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.conv = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 1, stride=1, padding=0)

        self.conv1 = nn.Conv2d(in_channel,out_channel, 3, 1, 1)
        
        if not self.args.attention:

            self.mask_predict = nn.Sequential(

                 nn.Conv2d(in_channels = in_channel, out_channels = 100, kernel_size = 3, stride=1, padding=1),

                 nn.BatchNorm2d(100),

                 nn.ReLU(),

                 nn.Conv2d(in_channels = 100, out_channels = 1, kernel_size = 1, stride=1, padding=0)
        
            )

            
            if self.args.is_sent_emb:

               self.mlp1 = Feedforward(256,in_channel)

               self.mlp2 = Feedforward(256,in_channel)
                
            else:
                
               self.mlp1 = Affine_Projection(16,img_length,in_channel,256)
                   
               self.mlp2 = Affine_Projection(16,img_length,in_channel,256)
            
        else:
            
            self.text_image_encoder = TextImageEncoder(img_length,img_width,256,in_channel)
            
            self.conv_mask = nn.Sequential(
        
                 nn.Conv2d(in_channel,in_channel,(1,1)),
                
                 nn.ReLU(),
            
                 nn.Conv2d(in_channel,in_channel,(1,1)),
                
                 nn.Tanh()      
        
            )
            
            
            
    def attention_image_text(self,image_feat,text_feat):
        
        image_feat_in = F.interpolate(image_feat,scale_factor=2, mode='bilinear', align_corners=True)
        
        normed_input = self.batchnorm(image_feat_in)
        
        encoded_out = self.text_image_encoder(image_feat_in,text_feat)
        
        mask = self.conv_mask(encoded_out)

        output = mask * encoded_out
        
        residue = None

        if self.in_channel != self.out_channel:

           residue = self.conv(normed_input)

           output = self.conv1(output)

        else:

           residue = normed_input

           output = self.conv1(output)

        out = residue + self.gamma * output
        
        return out
        
    def forward(self,image_feat,text_feat):

        image_feat_in = F.interpolate(image_feat,scale_factor=2, mode='bilinear', align_corners=True)

        logits = self.mask_predict(image_feat_in)

        mask = torch.sigmoid(logits)

        normed_input = self.batchnorm(image_feat_in)

        size = normed_input.size()

        gamma = self.mlp1(text_feat)

        beta = self.mlp2(text_feat)

        if self.args.is_sent_emb:
        
           gamma = gamma.unsqueeze(-1).unsqueeze(-1).expand(size)

           beta = beta.unsqueeze(-1).unsqueeze(-1).expand(size)
        
        elif self.args.is_word_emb:
 
           gamma = gamma.unsqueeze(-1).expand(size)

           beta = beta.unsqueeze(-1).expand(size)

        gamma_in = gamma * mask
         
        beta_in = beta * mask

        output = gamma_in * normed_input  + beta_in

        output = nn.ReLU(inplace=True)(output)

        residue = None

        if self.in_channel != self.out_channel:

           residue = self.conv(normed_input)

           output = self.conv1(output)

        else:

           residue = normed_input

           output = self.conv1(output)

        out = residue + self.gamma * output

        return out


class Generator(nn.Module):

    def __init__(self,input_dim,args,noise_dim=100):

        super(Generator, self).__init__()
        
        self.args = args

        self.input_dim = input_dim
        
        self.fc = nn.Linear(noise_dim,input_dim*8*4*4)

        self.channel_list = [input_dim*8,input_dim*8,input_dim*8,input_dim*8,input_dim*8,input_dim*4,input_dim*2,input_dim]

        self.img_length = [8,16,32,64,128,256]
        
        self.tanh = nn.Tanh()

        self.batchnorm = nn.BatchNorm2d(input_dim*2)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv =  nn.Conv2d(input_dim*2, 3, 3, 1, 1)
            
        self.GBlocks = nn.ModuleList([GBlock(self.channel_list[i],self.channel_list[i+1],self.img_length[i],self.img_length[i],self.args) for i in range(6)])

    def forward(self,image,text):

        image_feat = self.fc(image)

        image_feat = image_feat.view(image.size(0), 8 * self.input_dim, 4, 4)

        for i in range(6):
            
            if not self.args.attention:

               image_feat = self.GBlocks[i](image_feat,text)
                
            else:
                
               image_feat = self.GBlocks[i].attention_image_text(image_feat,text)

        image_feat = self.batchnorm(image_feat)    

        image_feat = self.relu(image_feat)

        out = self.conv(image_feat)   

        out = self.tanh(out)


        print(out.shape)

        return out

class DBlock(nn.Module):


    def __init__(self,in_channel,out_channel):

        super(DBlock, self).__init__()

        self.in_channel = in_channel

        self.out_channel = out_channel

        self.conv = nn.Sequential(

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

        image_feat_in = self.conv(image_feat)

        residue = F.avg_pool2d(residue,2)

        out = residue + self.gamma * image_feat_in

        return out

class Discriminator(nn.Module):

    def __init__(self,input_dim,args):

        super(Discriminator, self).__init__()
 
        self.args = args

        self.conv = nn.Conv2d(in_channels = 3, out_channels =input_dim, kernel_size = 3, stride=1, padding=1)

        self.channel_list = [input_dim,input_dim*2,input_dim*4,input_dim*8,input_dim*16,input_dim*16,input_dim*16]

        self.dblocks = nn.ModuleList([DBlock(self.channel_list[i],self.channel_list[i+1]) for i in range(6)])

        if self.args.attention or self.args.is_word_emb:

           self.conv1 = nn.Conv2d(256,256, 1, 5, 0, bias=False)
    
        self.conv_last = nn.Sequential(

                nn.Conv2d(input_dim * 16 + 256, input_dim * 2, 3, 1, 1, bias=False),

                nn.LeakyReLU(0.2, inplace=True),
            
                nn.Conv2d(input_dim * 2, 1, 4, 1, 0, bias=False)
        )
         

    def calculate_loss(self,out,text_feat):

        if not self.args.attention and not self.args.is_word_emb:

           text_feat = text_feat.view(-1, 256, 1, 1)

           text_feat = text_feat.repeat(1, 1, 4, 4)

        else:

       
           text_feat = text_feat.unsqueeze(-1)

           #text_feat = text_feat.view(-1, 256,-1, 1)

           text_feat = text_feat.repeat(1, 1, 1, 16)

           text_feat = self.conv1(text_feat)

        img_text = torch.cat((out, text_feat), 1)

        output = self.conv_last(img_text)

        return output


    def forward(self,input):

        image_feat = self.conv(input)

        for i in range(6):

            image_feat = self.dblocks[i](image_feat)

        out = image_feat

        return out  



