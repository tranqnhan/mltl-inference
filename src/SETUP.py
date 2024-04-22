import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from datagen import random_sampling, west_sampling
from random_mltl_formulas import generate_random_formula
import random
import pickle
import time

TOTAL_INPUT_TRACES = 256
OUTPUT_LENGTH = 256
MAX_VARIABLES = 10
MAX_TRAIN_VARIABLES = 4
MAX_DEPTH = 3

MAX_M_DELTA = 30

TEMPORAL_PROB = .5
MAX_TRACE_DELTA = 20
MAX_BOUND_VAL = 20

class Encoding:
    def __init__(self):
        # Input Token
        self.inSymbol = {
            '<SOS>' : 0,
            '<EOS>' : 1,
        }
        for i in range(MAX_VARIABLES):
            self.inSymbol[i] = i + self.inSymbol['<EOS>'] + 1
        self.inSize = len(self.inSymbol)

        # Output Token
        self.outSymbol = {
            '<SOS>' : 0,
            '<EOS>' : 1,
            '<PAD>' : 2,
            'F' : 3,
            'G' : 4,
            'U' : 5,
            'R' : 6,
            '[' : 7,
            ']' : 8,
            '(' : 9,
            ')' : 10,
            ',' : 11,
            '&' : 12,
            '|' : 13,
            '!' : 14
        }
        l = 15
        for i, l in enumerate(range(l, l + MAX_VARIABLES)):
            self.outSymbol["p" + str(i)] = l
        l += 1
        for i, l in enumerate(range(l, l + 10)):
            self.outSymbol[str(i)] = l
        
        self.outSize = len(self.outSymbol)
        self.outDimensionality = 128

        self.decoder = list(self.outSymbol.keys())
        print(self.decoder)
        print("Encoding initialized!")
        

    def encodeInputToken(self, idx):
        inputToken = [0.0] * self.inSize
        inputToken[self.inSymbol[idx]] = 1.0
        return inputToken
    
    def encodeTrace(self, postrace, negtrace):
        posseq = []
        for trace in postrace:
            encodedseq = []
            encodedseq.append(self.encodeInputToken('<SOS>'))
            for var in trace:
                multiHotInputToken = [0.0] * self.inSize
                for i in range(len(var)):
                    if (var[i] == '1'):
                        multiHotInputToken[self.inSymbol[i]] = 1.0 
                encodedseq.append(multiHotInputToken)
            
            encodedseq.append(self.encodeInputToken('<EOS>'))
            posseq.append(encodedseq)

        negseq = []
        for trace in negtrace:
            encodedseq = []
            encodedseq.append(self.encodeInputToken('<SOS>'))
            for var in trace:
                multiHotInputToken = [0.0] * self.inSize
                for i in range(len(var)):
                    if (var[i] == '1'):
                        multiHotInputToken[self.inSymbol[i]] = 1.0 
                encodedseq.append(multiHotInputToken)
            encodedseq.append(self.encodeInputToken('<EOS>'))        
            negseq.append(encodedseq)
        return [posseq, negseq]
    
    def encodeOutputToken(self, idx):
        outputToken = [0.0] * self.outDimensionality
        outputToken[self.outSymbol[idx]] = 1.0
        return outputToken, self.outSymbol[idx]
    
    def decodeFormula(self, formula):
        final = []
        for encoded in formula:
            i = np.array(encoded).argmax()
            if (i >= len(self.decoder)):
                v = "?"
            else:
                v = self.decoder[i]
            if (v == '<PAD>'):
                break
            final.append(v)
        return "".join(final)

    def encodeFormula(self, formula):
        seq = []
        n = []
        o, d = self.encodeOutputToken('<SOS>')
        seq.append(o)
        n.append(d)
        for i in formula:
            o, d = self.encodeOutputToken(i)
            seq.append(o)
            n.append(d)
        o, d = self.encodeOutputToken('<EOS>')
        seq.append(o)
        n.append(d)
        
        padding = OUTPUT_LENGTH - len(seq)
        if padding < 0:
            return None
        for i in range(padding):
            o, d = self.encodeOutputToken('<PAD>')
            seq.append(o)
            n.append(d)
        return seq, n

ENCODING = Encoding()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLTLDataset(Dataset):
    def __init__(self, directory, num_formula, sets_per_formula):
        self.traces_per_set = TOTAL_INPUT_TRACES
        self.num_formula = num_formula
        self.sets_per_formula = sets_per_formula
        self.directory = directory
        self.cached = {}

    def __len__(self):
        return self.num_formula

    def __getitem__(self, idx):
        if (not os.path.isdir(f'{self.directory}/{idx}')):
            while(True):
                try:
                    # Generate
                    pos_raw_list = []
                    neg_raw_list = []
                    traces_list = []
                    num_vars = random.randint(1, MAX_TRAIN_VARIABLES)
                    depth = random.randint(1, MAX_DEPTH)
                    
                    formula_raw = generate_random_formula(num_vars, depth, TEMPORAL_PROB, MAX_TRACE_DELTA, MAX_BOUND_VAL)
                    # + EOS and SOS
                    while(len(formula_raw) + 2 > OUTPUT_LENGTH):
                        formula_raw = generate_random_formula(num_vars, depth, TEMPORAL_PROB, MAX_TRACE_DELTA, MAX_BOUND_VAL)
                    
                    formula_string = "".join(formula_raw)
                    
                    formula, formula_raw = ENCODING.encodeFormula(formula_raw)
                    
                    k = random.randint(0, self.sets_per_formula-1)
                    for t in range(self.sets_per_formula):
                        pos_raw, neg_raw = west_sampling(formula_string, TOTAL_INPUT_TRACES, MAX_M_DELTA)
                        
                        traces = ENCODING.encodeTrace(pos_raw, neg_raw)
                        if (t == k):
                            outtraces = traces
                        pos_raw_list.append(pos_raw)
                        neg_raw_list.append(neg_raw)
                        traces_list.append(traces)
                    break
                except Exception as ex:
                    print(ex, flush=True)
                    continue
            
            # Saving things to files
            filename = f'{self.directory}/{idx}/rawformula.txt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as formula_file:
                formula_file.write(formula_string)
            
            outrawformula = np.array(formula_raw)
            np.save(f'{self.directory}/{idx}/rawformula', outrawformula)

            outencformula = np.array(formula)
            np.save(f'{self.directory}/{idx}/encodedformula', outencformula)

            for t in range(len(traces_list)):
                with open(f'{self.directory}/{idx}/postrace{t+1}.pickle', 'wb') as trace_file:
                    pickle.dump(pos_raw_list[t], trace_file)
                with open(f'{self.directory}/{idx}/negtrace{t+1}.pickle', 'wb') as trace_file:
                    pickle.dump(neg_raw_list[t], trace_file)
                with open(f'{self.directory}/{idx}/encodedtrace{t+1}.pickle', 'wb') as trace_file:
                    pickle.dump(traces_list[t], trace_file)
        else:
            # get a random number from 1 to num_traces_set
            k = random.randint(1, self.sets_per_formula)
            with open(f'{self.directory}/{idx}/encodedtrace{k}.pickle', 'rb') as trace_file:
                outtraces = pickle.load(trace_file)
            outencformula = np.load(f'{self.directory}/{idx}/encodedformula.npy')
            outrawformula = np.load(f'{self.directory}/{idx}/rawformula.npy')
            filename = f'{self.directory}/{idx}/rawformula.txt'
            with open(filename, 'r') as formula_file:
                formula_string = formula_file.read()

        return outtraces[0], outtraces[1], outencformula, outrawformula

# Leave the batch size as 1
TRAIN_DATASET = MLTLDataset("dataset/SEQ2SEQ/TRAIN", 500000, 1)
TEST_DATASET = MLTLDataset("dataset/SEQ2SEQ/TEST", 5000, 1)

def collate_fn(minibatch):
    p = TOTAL_INPUT_TRACES // 2
    l = len(minibatch)
    newpos = []
    newneg = []
    enc_formulas = []
    raw_formulas = []

    for i in range(p):
        newneg.append([])
        newpos.append([])

    for i in range(p):
        for j in range(l):
            pos, neg, _, _ = minibatch[j]

            newpos[i].append(pos[i])
            newneg[i].append(neg[i])

    for i in range(p):
        newneg[i] = torch.tensor(newneg[i]).float()
        newpos[i] = torch.tensor(newpos[i]).float()

    
    for _, _, formula, raw_formula in minibatch:
        enc_formulas.append(formula)
        raw_formulas.append(raw_formula)
    return newpos, newneg, np.array(enc_formulas), np.array(raw_formulas)

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, collate_fn=collate_fn, batch_size=1, shuffle=True)
TEST_DATALOADER = DataLoader(TEST_DATASET, collate_fn=collate_fn, batch_size=1, shuffle=False)

EPOCH = 1000
EPOCH_SAVE = 50

if __name__ == "__main__":
    pos, neg, formula, r = next(iter(TRAIN_DATALOADER))
