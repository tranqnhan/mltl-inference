from datagen import random_sampling, west_sampling
from random_mltl_formulas import generate_random_formula
from utils import west

import torch
import torch.nn as nn
import numpy as np

from MODEL import EncoderNN, DecoderNN 
from SETUP import DEVICE, ENCODING, TRAIN_DATALOADER, TEST_DATALOADER, OUTPUT_LENGTH, EPOCH, EPOCH_SAVE

def train():
    encoder = EncoderNN().to(DEVICE)
    decoder = DecoderNN().to(DEVICE)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.0001)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.0001)
    criterion = nn.NLLLoss().to(DEVICE)

    tgt_mask = nn.Transformer().generate_square_subsequent_mask(sz = OUTPUT_LENGTH - 1).to(DEVICE)
    print("BEGIN TRAINING")
    for epoch in range(EPOCH):
        total_loss = 0
        for data in TRAIN_DATALOADER:
            pos, neg, enc_formula, raw_formula = data
            
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            input_f = torch.tensor(np.array([enc_formula[0][:-1]]), requires_grad=True).to(DEVICE)
            output_f = torch.tensor(np.array([raw_formula[0][1:]])).to(DEVICE)
            encoder_outputs = encoder(pos, neg)
            pad_mask = decoder.create_pad_mask(torch.tensor(np.array([raw_formula[0][:-1]])).float()).to(DEVICE)
            pad_begin = np.argwhere(raw_formula[0] == ENCODING.outSymbol['<PAD>'])[0][0]
            decoder_outputs = decoder(encoder_outputs.float(), input_f.float(), tgt_mask, pad_mask)
            loss = criterion(decoder_outputs[:pad_begin].mT.float(), output_f.long()[:pad_begin])
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step() 
            encoder.avg_lstm()
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
            loss = criterion(decoder_outputs[:pad_begin].mT.float(), output_f.long()[:pad_begin])
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step() 
            encoder.avg_lstm()
            total_loss += loss.item()

        total_loss = total_loss / (len(TRAIN_DATALOADER) * 2)
        print(f"LOSS: {total_loss}")
        if (epoch % EPOCH_SAVE == 0):
            torch.save(encoder.state_dict(), f"ENCODER_NN_STATES_{epoch}")
            torch.save(decoder.state_dict(), f"DECODER_NN_STATES_{epoch}")

    return epoch

def test(at_epoch):

    encoder = EncoderNN()
    decoder = DecoderNN()
    
    encoder.load_state_dict(torch.load(f"ENCODER_NN_STATES_{at_epoch}"))
    decoder.load_state_dict(torch.load(f"DECODER_NN_STATES_{at_epoch}"))

    encoder.to(DEVICE)
    decoder.to(DEVICE)
    criterion = nn.NLLLoss().to(DEVICE)
    print("BEGIN TESTING")
    for data in TEST_DATALOADER:
        pos, neg, enc_formula, raw_formula = data
        formula = [ENCODING.encodeOutputToken('<SOS>')[0]]
        
        mem = encoder(pos, neg)
        for i in range(255): 
            tgt_mask = nn.Transformer().generate_square_subsequent_mask(sz = len(formula)).to(DEVICE)
            formula_tensor = torch.tensor([formula]).to(DEVICE)
            pred = decoder(mem, formula_tensor, tgt_mask, None)
            next_item = pred.cpu().detach().numpy()[0][-1].argmax() # num with highest probability
            val = [0.0] * 128
            val[next_item] = 1.0
            formula.append(val)
            # Stop if model predicts end of sentence
            if next_item == ENCODING.outSymbol['<EOS>']:
                padding = OUTPUT_LENGTH - len(formula)
                if padding > 0:
                    for _ in range(padding):
                        o, _ = ENCODING.encodeOutputToken('<PAD>')
                        formula.append(o)
                break
        output_f = torch.tensor([raw_formula[0][1:]]).to(DEVICE)
        predict_f = torch.tensor([formula[1:]]).to(DEVICE)

        loss = criterion(predict_f.mT.float(), output_f.long())
        print(f"TEST LOSS: {loss.item()}")

if __name__ == '__main__':
    at_epoch = train()
    test(at_epoch)