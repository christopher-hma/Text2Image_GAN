import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TextEncoder(nn.Module):
    
    def __init__(self,nwords,input_size,hidden_size):

        super(TextEncoder,self).__init__()

        self.hidden_size = hidden_size

        self.input_size = input_size

        self.encoder = nn.Embedding(nwords,input_size)
        
        self.drop = nn.Dropout(0.1)
       
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size,num_layers = 1,batch_first=True,bidirectional=True)
        
    def forward(self,captions,caplen,initial_hidden):


        captions = captions.cuda()

        caplen = caplen.cuda()

        caplen = caplen.data.tolist()

        captions = self.drop(self.encoder(captions))
        
        packed_seq_batch = pack_padded_sequence(captions, lengths=caplen, batch_first=True)
        
        output, (hn, cn) = self.lstm(packed_seq_batch,initial_hidden)

        print(caplen)

        print("hihi")

        print(hn.shape)

        print(cn.shape)
        
        padded_output, output_lens = pad_packed_sequence(output, batch_first=True)
 
        print(padded_output.shape)
        
        sent_emb = hn.transpose(0,1).contiguous().view(-1,self.hidden_size * 2)
 
        print(sent_emb.shape)
        
        print(padded_output.shape)

        word_emb = padded_output.transpose(1,2)

        print(word_emb.shape)
        
        return sent_emb,word_emb