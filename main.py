import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import random
import numpy as np
from Dataset import Dataset,get_data
from models import Generator,Discriminator
from TextEncoder import TextEncoder
import os
import io
import sys
import yaml
import argparse
from torch.utils.data import DataLoader
import tqdm

def train(dataloader,ixtoword,NetG,NetD,text_encoder,NetG_optimizer,NetD_optimizer,cfg,args,device):

   
    start_epoch = 0
    
    text_encoder.train()

    for epoch in tqdm.tqdm(range(start_epoch + 1, cfg["MAX_EPOCH"] + 1)):

        #data_iter = iter(dataloader)
        for step, data in enumerate(dataloader, 0):
        #for step in tqdm.tqdm(range(len(data_iter))):
            #data = data_iter.next()
            img,caption,cap_len = get_data(data,device)
            print(img[0].shape)
            print(caption.shape)
            print(cap_len.shape)
            hidden = torch.randn(2,caption.shape[0],cfg["LSTM_HIDDEN_SIZE"])
            cell = torch.randn(2,caption.shape[0],cfg["LSTM_HIDDEN_SIZE"])
            hidden = hidden.cuda()
            cell = cell.cuda()
            initial_hidden = (hidden,cell)
            
            sent_emb_feat,word_emb_feat = text_encoder(caption,cap_len,initial_hidden)

            sent_emb_feat = sent_emb_feat.detach()

            word_emb_feat = word_emb_feat.detach()
            
            bs = caption.shape[0]

            
            # update Discriminator
            img_features = NetD(img[0])

            
            if args.is_sent_emb:

               text_emb = sent_emb_feat

            else:

               text_emb = word_emb_feat



            d1 = NetD.calculate_loss(img_features,text_emb)

            error_1 = torch.relu(1-d1)
 
            error_1 = error_1.squeeze().mean()
         
            
            d2 = NetD.calculate_loss(img_features[1:bs],text_emb[:bs-1])

            error_2 = torch.relu(1+d2).mean()

            
            noise = torch.rand(bs,100)

            noise = noise.cuda()
        
            fake = NetG(noise,text_emb)

          

            fake_img_features = NetD(fake.detach())

            d3 = NetD.calculate_loss(fake_img_features,text_emb)
        
            error_3 = torch.relu(1+d3).mean()
        
            
            total_errors = error_1 + 0.5 * error_2 


            total_errors += 0.5 * error_3

            
            NetD_optimizer.zero_grad()
        
            total_errors.backward()
        
            NetD_optimizer.step()

            

            

            # update Generator

            fake_features = NetD(fake.detach())
 
            g = NetD.calculate_loss(fake_features,text_emb)
        
            error_g = -1 * g.mean()

            NetG_optimizer.zero_grad()
        
            error_g.backward()

            NetG_optimizer.step()

            
      
                 


if __name__ == "__main__":

   config_file = 'C://Users//chris//Text2Image_GAN//coco//config.yml'
    
   parser = argparse.ArgumentParser(description='Train a Text2Image_GAN')

   parser.add_argument('--mode', type = str, default="train")

   parser.add_argument('--is_sent_emb', type = str, default=False)

   parser.add_argument('--is_word_emb', type = str, default=True)

   parser.add_argument('--attention', type = str, default=False)


   args = parser.parse_args()
        
   with open(config_file, "r") as ymlfile:
    
        cfg = yaml.safe_load(ymlfile)

  
   random.seed(cfg["seed"])

   np.random.seed(cfg["seed"])
    
   torch.manual_seed(cfg["seed"])

   imsize = cfg["IMAGE_SIZE"]

   batch_size = cfg["BATCH_SIZE"]

   image_transform = transforms.Compose([
    
        transforms.Resize(int(imsize * 76 / 64)),
    
        transforms.RandomCrop(imsize),
    
        transforms.RandomHorizontalFlip()])

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   if args.mode == "train":

      dataset = Dataset(cfg,"train",image_transform)

      ixtoword = dataset.ixtoword

      wordtoix = dataset.wordtoix

      nwords = dataset.n_words
        
      print(dataset.n_words)

      print(dataset.filenames)

      print(dataset.captions)

      print(len(dataset.filenames))

      print(len(dataset.captions))

      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)


   else:
 
      dataset = Dataset(cfg,"test",image_transform)
 
      ixtoword = dataset.ixtoword

      wordtoix = dataset.wordtoix
        
      nwords = dataset.n_words

      print(dataset.n_words)
 
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)

   NetG = Generator(64,args,100).cuda()

   NetD = Discriminator(64,args).cuda()

   NetG_optimizer = torch.optim.Adam(NetG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    
   NetD_optimizer = torch.optim.Adam(NetD.parameters(), lr=0.0004, betas=(0.0, 0.9))


   textEncoder = TextEncoder(nwords,cfg["EMBEDDING_DIM"],cfg["LSTM_HIDDEN_SIZE"])


   textEncoder.cuda()

   for p in textEncoder.parameters():
        
       p.requires_grad = True

   textEncoder.eval()



   if args.mode == "train":

      train(dataloader,ixtoword,NetG,NetD,textEncoder,NetG_optimizer,NetD_optimizer,cfg,args,device)

