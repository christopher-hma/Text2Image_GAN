import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
import pickle


def get_data(data,device):
    
    imgs, captions, captions_lens = data

    real_imgs = []
        
    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)
       
    imgs[0] = imgs[0][sorted_cap_indices]
      
    captions = captions[sorted_cap_indices].squeeze()

    if device == torch.device("cuda"):

       captions = Variable(captions).cuda()
        
       sorted_cap_lens = Variable(sorted_cap_lens).cuda()

       real_imgs.append(Variable(imgs[0]).cuda())
    
    else:

       captions = Variable(captions)
        
       sorted_cap_lens = Variable(sorted_cap_lens)

       real_imgs.append(Variable(imgs[0]))
    
    return [real_imgs, captions, sorted_cap_lens]


def retrieve_imgs(path,transform=None, normalize=None):
    
    image = Image.open(path).convert('RGB')
    
    if transform is not None:
        
       image = transform(image)

    output = list()
    
    output.append(normalize(image))

    return output


class Dataset():
    
    def __init__(self, cfg, mode ='train',transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.num_captions_per_image = cfg["text_captions_per_image"]
        self.mode = mode      
        self.cfg = cfg
        self.filenames = self.get_filenames()
        self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_caption_data()

    def get_filenames(self):
        
        if self.mode == 'train':
            
           file_path = self.cfg["train_filenames_pickle"]
            
            
        else:
            
           file_path = self.cfg["test_filenames_pickle"]
        
        with open(file_path, 'rb') as f:
            
             filenames = pickle.load(f)
            
        print('Load filenames from: %s (%d)' % (file_path, len(filenames)))
        
        return filenames
            
        
    def load_caption_data(self):
        
        file_path = self.cfg["captions"]
        
        with open(file_path, 'rb') as f:
            
             out = pickle.load(f)
                
             train_captions, test_captions, ixtoword, wordtoix = out[0], out[1], out[2], out[3]
                
             del out
                
             n_words = len(ixtoword)
                
             print('Load from: ', file_path)
                
        if self.mode == 'train':
            
            captions = train_captions

        else:  
            
            captions = test_captions
            
            
        return captions, ixtoword, wordtoix, n_words


    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        
        if (sent_caption == 0).sum() > 0:
            
            print('ERROR: do not need END (0) token', sent_caption)
            
        num_words = len(sent_caption)
        
        caption = np.zeros((self.cfg["MAX_WORD_PER_CAPTION"], 1), dtype='int64')
        
        cap_len = num_words
        
        if num_words <= self.cfg["MAX_WORD_PER_CAPTION"]:
            
           caption[:num_words, 0] = sent_caption

           cap_len = self.cfg["MAX_WORD_PER_CAPTION"]
        
        else:
            ix = list(np.arange(num_words))            
            np.random.shuffle(ix)            
            ix = ix[:self.cfg["MAX_WORD_PER_CAPTION"]]           
            ix = np.sort(ix)
            caption[:, 0] = sent_caption[ix]
            cap_len = self.cfg["MAX_WORD_PER_CAPTION"]
            
        return caption, cap_len

    def __getitem__(self, index):
        
        file_name = self.filenames[index]
        
        data_dir = self.cfg["train_image_dir"] if self.mode == "train" else self.cfg["test_image_dir"]
        
        img_name = '%s/%s.jpg' % (data_dir, file_name)
       
        imgs = retrieve_imgs(img_name,self.transform, normalize=self.norm)
        
        ix = random.randint(0, self.num_captions_per_image)
        
        sent_ix = index * self.num_captions_per_image + ix
        
        caps, cap_len = self.get_caption(sent_ix)
        
        return imgs, caps, cap_len

    def __len__(self):
        return len(self.filenames)