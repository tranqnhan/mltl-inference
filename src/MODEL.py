import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from tqdm import tqdm

from SETUP import TRAIN_DATALOADER, TEST_DATALOADER, ENCODING, DEVICE, TOTAL_INPUT_TRACES, OUTPUT_LENGTH

class EncoderNN(nn.Module):
    def __init__(self):
        super(EncoderNN, self).__init__()
        self.lstm_hidden_size = 128
        self.transformer_heads = 8
        self.num_transformer_layer = 2
        self.poslstms = []
        self.neglstms = []

        for _ in range(TOTAL_INPUT_TRACES // 2):
            self.poslstms.append(nn.LSTM(ENCODING.inSize, self.lstm_hidden_size, batch_first=True).to(DEVICE))
            self.neglstms.append(nn.LSTM(ENCODING.inSize, self.lstm_hidden_size, batch_first=True).to(DEVICE))
         
        transformer_enc_layer = nn.TransformerEncoderLayer(dim_feedforward=512, d_model=self.lstm_hidden_size, nhead=self.transformer_heads, dropout=0.3, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(transformer_enc_layer, self.num_transformer_layer).to(DEVICE)
        
    def avg_lstm(self):
        for lstms in [self.poslstms, self.neglstms]:
            states = [lstm.state_dict() for lstm in lstms]
            
            avg = {k: sum([state[k] for state in states]) / len(states) for k in states[0]}

            for key in avg.keys():
                states[0][key] = avg[key]

            for lstm in lstms:
                lstm.load_state_dict(states[0])

    def forward(self, pos, neg):
        T = []
        
        # Probably should parallelize this but I don't know how
        for i in range(len(pos)):
            _, (hn, _) = self.poslstms[i](pos[i].to(DEVICE))
            T.append(hn)
        
        for i in range(len(neg)):
            out, (hn, _) = self.neglstms[i](neg[i].to(DEVICE))
            T.append(hn)

        T = torch.cat(T, dim=1).to(DEVICE)
        out = self.transformer_enc(T)

        return out
    
class DecoderNN(nn.Module):
    def __init__(self):
        super(DecoderNN, self).__init__()
        self.transformer_heads = 8 # embed_dim must be divisible by num_heads
        self.num_transformer_layer = 2
        transformer_dec_layer = nn.TransformerDecoderLayer(dim_feedforward=512, d_model=ENCODING.outDimensionality, nhead=self.transformer_heads, dropout=0.3, batch_first=True)
        self.transformer_dec = nn.TransformerDecoder(decoder_layer=transformer_dec_layer, num_layers=self.num_transformer_layer).to(DEVICE)
        self.softmax = nn.LogSoftmax(dim=2)

    def create_pad_mask(self, src_tensor):
        pad = src_tensor
        pad = pad.masked_fill(pad == ENCODING.outSymbol['<PAD>'], -float('inf'))
        pad = pad.masked_fill(pad != -float('inf'), 0.0)
        return pad #(src_tensor == ENCODING.outSymbol['<PAD>'])

    def forward(self, encoder_mem, tgt, tgt_mask, pad_mask):
        out = self.transformer_dec(tgt, encoder_mem, tgt_key_padding_mask=pad_mask, tgt_is_causal=True, tgt_mask=tgt_mask)
        out = self.softmax(out)
        return out
    
if __name__ == "__main__":
    '''
    # SANITY CHECKING
    encoder = EncoderNN().to(DEVICE)
    decoder = DecoderNN().to(DEVICE)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.0001)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.0001)
    criterion = nn.NLLLoss().to(DEVICE)
    
    tgt_mask = nn.Transformer().generate_square_subsequent_mask(sz = OUTPUT_LENGTH - 1).to(DEVICE)
    
    tl = []
    vl = []

    for _ in range(30):
        total_loss = 0
        encoder.train()
        decoder.train()
        for data in tqdm(TRAIN_DATALOADER):
            pos, neg, enc_formula, raw_formula = data
            
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            input_f = torch.tensor(np.array([enc_formula[0][:-1]]), requires_grad=True).to(DEVICE)
            output_f = torch.tensor(np.array([raw_formula[0][1:]])).to(DEVICE)
            encoder_outputs = encoder(pos, neg)
            pad_mask = decoder.create_pad_mask(torch.tensor(np.array([raw_formula[0][:-1]])).float()).to(DEVICE)
            pad_begin = np.argwhere(raw_formula[0] == ENCODING.outSymbol['<PAD>'])[0][0]
            decoder_outputs = decoder(encoder_outputs.float(), input_f.float(), tgt_mask, pad_mask)
            loss = criterion(decoder_outputs[0][:pad_begin].expand(1, -1, -1).mT.float(), output_f.long()[0][:pad_begin].expand(1, -1))
            loss.backward()
            total_loss += loss.item()
            encoder_optimizer.step()
            decoder_optimizer.step() 
            encoder.avg_lstm()
 
            enc_formula, raw_formula = ENCODING.encodeNegateFormula(enc_formula[0], raw_formula[0])
            enc_formula = [enc_formula]
            raw_formula = [raw_formula]
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            input_f = torch.tensor(np.array([enc_formula[0][:-1]]), requires_grad=True).to(DEVICE)
            output_f = torch.tensor(np.array([raw_formula[0][1:]])).to(DEVICE)
            encoder_outputs = encoder(neg, pos)
            pad_mask = decoder.create_pad_mask(torch.tensor(np.array([raw_formula[0][:-1]])).float()).to(DEVICE)
            pad_begin = np.argwhere(raw_formula[0] == ENCODING.outSymbol['<PAD>'])[0][0]
            decoder_outputs = decoder(encoder_outputs.float(), input_f.float(), tgt_mask, pad_mask)
            loss = criterion(decoder_outputs[0][:pad_begin].expand(1, -1, -1).mT.float(), output_f.long()[0][:pad_begin].expand(1, -1))
            loss.backward()
            total_loss += loss.item()
            encoder_optimizer.step()
            decoder_optimizer.step() 
            encoder.avg_lstm()

        print(f"TRAINING LOSS: {total_loss / (len(TRAIN_DATALOADER) * 2)}")
        tl.append(total_loss / (len(TRAIN_DATALOADER) * 2))

        encoder.eval()
        decoder.eval()
        total_loss = 0
        for data in tqdm(TEST_DATALOADER):
            pos, neg, enc_formula, raw_formula = data
            
            input_f = torch.tensor(np.array([enc_formula[0][:-1]]), requires_grad=True).to(DEVICE)
            output_f = torch.tensor(np.array([raw_formula[0][1:]])).to(DEVICE)
            encoder_outputs = encoder(pos, neg)
            pad_mask = decoder.create_pad_mask(torch.tensor(np.array([raw_formula[0][:-1]])).float()).to(DEVICE)
            pad_begin = np.argwhere(raw_formula[0] == ENCODING.outSymbol['<PAD>'])[0][0]
            decoder_outputs = decoder(encoder_outputs.float(), input_f.float(), tgt_mask, pad_mask)
            loss = criterion(decoder_outputs[0][:pad_begin].expand(1, -1, -1).mT.float(), output_f.long()[0][:pad_begin].expand(1, -1))
            total_loss += loss.item()
            
            enc_formula, raw_formula = ENCODING.encodeNegateFormula(enc_formula[0], raw_formula[0])
            enc_formula = [enc_formula]
            raw_formula = [raw_formula]
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            input_f = torch.tensor(np.array([enc_formula[0][:-1]]), requires_grad=True).to(DEVICE)
            output_f = torch.tensor(np.array([raw_formula[0][1:]])).to(DEVICE)
            encoder_outputs = encoder(neg, pos)
            pad_mask = decoder.create_pad_mask(torch.tensor(np.array([raw_formula[0][:-1]])).float()).to(DEVICE)
            pad_begin = np.argwhere(raw_formula[0] == ENCODING.outSymbol['<PAD>'])[0][0]
            decoder_outputs = decoder(encoder_outputs.float(), input_f.float(), tgt_mask, pad_mask)
            loss = criterion(decoder_outputs[0][:pad_begin].expand(1, -1, -1).mT.float(), output_f.long()[0][:pad_begin].expand(1, -1))
            total_loss += loss.item()
        print(f"VALIDATION LOSS: {total_loss / (len(TEST_DATALOADER) * 2)}")
        #print(loss.detach().item())

        vl.append(total_loss / (len(TEST_DATALOADER) * 2))
    
    print(tl)
    print(vl)
    torch.save(encoder.state_dict(), f"ENCODER_NN_STATES_-1")
    torch.save(decoder.state_dict(), f"DECODER_NN_STATES_-1")
    exit()
'''
    # PREDICTION
    encoder = EncoderNN()
    decoder = DecoderNN()
    
    encoder.load_state_dict(torch.load(f"ENCODER_NN_STATES_-1"))
    decoder.load_state_dict(torch.load(f"DECODER_NN_STATES_-1"))

    encoder.to(DEVICE)
    decoder.to(DEVICE)
    
    encoder.eval()
    decoder.eval()
    criterion = nn.NLLLoss().to(DEVICE)
    print("BEGIN TESTING")
    for data in tqdm(TEST_DATALOADER):
        pos, neg, enc_formula, raw_formula = data
        formula = [ENCODING.encodeOutputToken('<SOS>')[0]]
        posmem = encoder(pos, neg)
        negmem = encoder(neg, pos)
        for mem in [posmem, negmem]:
            for _ in range(50): 
                tgt_mask = nn.Transformer().generate_square_subsequent_mask(sz = len(formula)).to(DEVICE)
                formula_tensor = torch.tensor([formula]).to(DEVICE)
                pred = decoder(mem, formula_tensor, tgt_mask, None)
                next_item = pred.cpu().detach().numpy()[0][-1].argmax() # num with highest probability
                val = [0.0] * 128
                val[next_item] = 1.0
                formula.append(val)
                # Stop if model predicts end of sentence
                if next_item == ENCODING.outSymbol['<EOS>'] or next_item == ENCODING.outSymbol['<PAD>'] :
                    padding = OUTPUT_LENGTH - len(formula)
                    if padding > 0:
                        for _ in range(padding):
                            o, _ = ENCODING.encodeOutputToken('<PAD>')
                            formula.append(o)
                    break

            output_f = torch.tensor([raw_formula[0][1:]]).to(DEVICE)
            predict_f = torch.tensor([formula[1:]]).to(DEVICE)


            print(f"\n {ENCODING.decodeRawFormula(raw_formula[0])} -> {ENCODING.decodeFormula(formula[1:])} \n")

    # Rearranging experiment
    exit()
    random.shuffle(pos)
    random.shuffle(neg)

    formula = [ENCODING.encodeOutputToken('<SOS>')[0]]
    mem = encoder(pos, neg)
    for _ in range(256): 
        tgt_mask = nn.Transformer().generate_square_subsequent_mask(sz = len(formula))
        formula_tensor = torch.tensor([formula])
        pred = decoder(mem, formula_tensor, tgt_mask, None)
        next_item = pred.detach().numpy()[0][-1].argmax() # num with highest probability
        val = [0.0] * 128
        val[next_item] = 1.0
        formula.append(val)
        print(next_item)
        # Stop if model predicts end of sentence
        if next_item == ENCODING.outSymbol['<EOS>']:
            break
    print(ENCODING.decodeFormula(formula))

    # Dependency experiment
    formula = [ENCODING.encodeOutputToken('<SOS>')[0]]
    mem = encoder(neg, pos)
    for _ in range(256): 
        tgt_mask = nn.Transformer().generate_square_subsequent_mask(sz = len(formula))
        formula_tensor = torch.tensor([formula])
        pred = decoder(mem, formula_tensor, tgt_mask, None)
        next_item = pred.detach().numpy()[0][-1].argmax() # num with highest probability
        val = [0.0] * 128
        val[next_item] = 1.0
        formula.append(val)
        print(next_item)
        # Stop if model predicts end of sentence
        if next_item == ENCODING.outSymbol['<EOS>']:
            break
    print(ENCODING.decodeFormula(formula))
