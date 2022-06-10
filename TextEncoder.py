import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TextEncoder(nn.Module):
    
    def __init__(self,input_size,hidden_size):

        super(TextEncoder,self).__init__()

        self.hidden_size = hidden_size

        self.input_size = input_size
        
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size ,num_layers = 2,batch_first=True,bidirectional=True)
        
    def forward(self,captions,caplen,initial_hidden):

        caplen = caplen.data.tolist()
        
        packed_seq_batch = pack_padded_sequence(captions, lengths=caplen, batch_first=True)
        
        output, (hn, cn) = self.lstm(packed_seq_batch,initial_hidden)
        
        padded_output, output_lens = pad_packed_sequence(output, batch_first=True)
        
        sent_emb = hn.transpose(0,1).contiguous().view(-1,self.hidden_size * 2)
        
        word_emb = padded_output[0].transpose(1,2)
        
        return sent_emb,word_emb