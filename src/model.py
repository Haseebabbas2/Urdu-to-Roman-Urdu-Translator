import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src len, batch size] (if batch_first=False)
        # But my dataloader uses batch_first=True. 
        # LSTM expects [seq_len, batch, input_size] by default unless batch_first=True is set.
        # I will use batch_first=True in LSTM for consistency.
        
        # src = [batch size, src len]
        embedded = self.dropout(self.embedding(src))
        
        # embedded = [batch size, src len, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # outputs = [batch size, src len, hid dim * 2]
        # hidden = [n layers * 2, batch size, hid dim]
        # cell = [n layers * 2, batch size, hid dim]
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=False, batch_first=True)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        # input = [batch size] (one char at a time)
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(1)
        # input = [batch size, 1]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch size, 1, emb dim]
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # output = [batch size, 1, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(1))
        # prediction = [batch size, output dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Assertions to ensure dimensions match
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers * 2 == decoder.n_layers, \
            "Encoder layers * 2 must be equal to Decoder layers!"
            
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        # teacher_forcing_ratio is probability to use teacher forcing
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        # hidden = [n layers * 2, batch size, hid dim] -> Matches decoder [n layers, batch size, hid dim]
        
        # first input to the decoder is the <sos> tokens
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = output
            
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[:, t] if teacher_force else top1
            
        return outputs
