import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from datagen import generate_traces, west_sampling
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
        idx = []
        o, d = self.encodeOutputToken('<SOS>')
        seq.append(o)
        idx.append(d)
        for i in formula:
            o, d = self.encodeOutputToken(i)
            seq.append(o)
            idx.append(d)
        o, d = self.encodeOutputToken('<EOS>')
        seq.append(o)
        idx.append(d)
        
        padding = OUTPUT_LENGTH - len(seq)
        if padding < 0:
            return None
        for i in range(padding):
            o, d = self.encodeOutputToken('<PAD>')
            seq.append(o)
            idx.append(d)
        return seq, idx

    def encodeNegateFormula(self, formula, idx):
        open_par_one_hot, open_par_idx = self.encodeOutputToken('(')
        close_par_one_hot, close_par_idx = self.encodeOutputToken(')')
        negate_one_hot, negate_idx = self.encodeOutputToken('!')
        _, eos_idx = self.encodeOutputToken('<EOS>')
        formula_list = formula.tolist()
        idx_list = idx.tolist()
        eos_idx_i = idx_list.index(eos_idx)
        formula_list.insert(eos_idx_i, close_par_one_hot)
        formula_list.insert(1, open_par_one_hot)
        formula_list.insert(1, negate_one_hot)
        idx_list.insert(eos_idx_i, close_par_idx)
        idx_list.insert(1, open_par_idx)
        idx_list.insert(1, negate_idx)

        idx = np.array(idx_list[:-3])
        formula = np.array(formula_list[:-3])

        return formula, idx

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
            # Generate
            pos_raw_list = []
            neg_raw_list = []
            traces_list = []
            num_vars = random.randint(1, MAX_TRAIN_VARIABLES)
            depth = random.randint(1, MAX_DEPTH)
            
            formula_raw = generate_random_formula(num_vars, depth, TEMPORAL_PROB, MAX_TRACE_DELTA, MAX_BOUND_VAL)
            # + EOS and SOS and !()
            while(len(formula_raw) + 5 > OUTPUT_LENGTH):
                formula_raw = generate_random_formula(num_vars, depth, TEMPORAL_PROB, MAX_TRACE_DELTA, MAX_BOUND_VAL)
            
            formula_string = "".join(formula_raw)
            
            formula, formula_raw = ENCODING.encodeFormula(formula_raw)
            
            k = random.randint(0, self.sets_per_formula-1)
            for t in range(self.sets_per_formula):
                pos_raw, neg_raw = generate_traces(formula_string, TOTAL_INPUT_TRACES, MAX_M_DELTA)

                while (pos_raw is None):
                    formula_raw = generate_random_formula(num_vars, depth, TEMPORAL_PROB, MAX_TRACE_DELTA, MAX_BOUND_VAL)
                    # + EOS and SOS and !()
                    while(len(formula_raw) + 5 > OUTPUT_LENGTH):
                        formula_raw = generate_random_formula(num_vars, depth, TEMPORAL_PROB, MAX_TRACE_DELTA, MAX_BOUND_VAL)
                    
                    formula_string = "".join(formula_raw)
                    formula, formula_raw = ENCODING.encodeFormula(formula_raw)
                    pos_raw, neg_raw = generate_traces(formula_string, TOTAL_INPUT_TRACES, MAX_M_DELTA)

                traces = ENCODING.encodeTrace(pos_raw, neg_raw)
                if (t == k):
                    outtraces = traces
                pos_raw_list.append(pos_raw)
                neg_raw_list.append(neg_raw)
                traces_list.append(traces)
            
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
TRAIN_DATASET = MLTLDataset("dataset/SEQ2SEQ/TRAIN", 10000, 1)
TEST_DATASET = MLTLDataset("dataset/SEQ2SEQ/TEST", 200, 1)

def collate_fn(minibatch):
    num_of_traces = TOTAL_INPUT_TRACES // 2
    num_of_batches = len(minibatch)
    pos = []
    neg = []
    
    one_hot_encoded_formulas = []
    token_encoded_formulas = []


    for i in range(num_of_traces):
        neg.append([])
        pos.append([])        

    for i in range(num_of_traces):
        for j in range(num_of_batches):
            posm, negm, _, _ = minibatch[j]
            pos[i].append(posm[i])
            neg[i].append(negm[i])
    
    for _, _, one_hot_formula, token_formula in minibatch:
        one_hot_encoded_formulas.append(one_hot_formula)
        token_encoded_formulas.append(token_formula)

    for i in range(num_of_traces):
        pos[i] = torch.tensor(pos[i])
        neg[i] = torch.tensor(neg[i])

    return pos, neg, np.array(one_hot_encoded_formulas), np.array(token_encoded_formulas)

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, collate_fn=collate_fn, batch_size=1, shuffle=True)
TEST_DATALOADER = DataLoader(TEST_DATASET, collate_fn=collate_fn, batch_size=1, shuffle=False)

EPOCH = 100
EPOCH_SAVE = 1

def datagen(directory, num_of_examples):
    for idx in range(num_of_examples):
        print(f"WORKING ON EXAMPLE {idx}", flush=True)
        if os.path.isdir(f'{directory}/{idx}'):
            continue
        # Generate
        pos_raw_list = []
        neg_raw_list = []
        traces_list = []
        num_vars = random.randint(1, MAX_TRAIN_VARIABLES)
        depth = random.randint(1, MAX_DEPTH)
        
        formula_raw = generate_random_formula(num_vars, depth, TEMPORAL_PROB, MAX_TRACE_DELTA, MAX_BOUND_VAL)
        # + EOS and SOS and !()
        while(len(formula_raw) + 5 > OUTPUT_LENGTH):
            formula_raw = generate_random_formula(num_vars, depth, TEMPORAL_PROB, MAX_TRACE_DELTA, MAX_BOUND_VAL)
        
        formula_string = "".join(formula_raw)
        
        formula, formula_raw = ENCODING.encodeFormula(formula_raw)
        
        pos_raw, neg_raw = generate_traces(formula_string, TOTAL_INPUT_TRACES, MAX_M_DELTA)

        while (pos_raw is None):
            formula_raw = generate_random_formula(num_vars, depth, TEMPORAL_PROB, MAX_TRACE_DELTA, MAX_BOUND_VAL)
            # + EOS and SOS and !()
            while(len(formula_raw) + 5 > OUTPUT_LENGTH):
                formula_raw = generate_random_formula(num_vars, depth, TEMPORAL_PROB, MAX_TRACE_DELTA, MAX_BOUND_VAL)
            
            formula_string = "".join(formula_raw)
            formula, formula_raw = ENCODING.encodeFormula(formula_raw)
            pos_raw, neg_raw = generate_traces(formula_string, TOTAL_INPUT_TRACES, MAX_M_DELTA)

        traces = ENCODING.encodeTrace(pos_raw, neg_raw)
        
        pos_raw_list.append(pos_raw)
        neg_raw_list.append(neg_raw)
        traces_list.append(traces)
    
        # Saving things to files
        filename = f'{directory}/{idx}/rawformula.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as formula_file:
            formula_file.write(formula_string)
        
        outrawformula = np.array(formula_raw)
        np.save(f'{directory}/{idx}/rawformula', outrawformula)

        outencformula = np.array(formula)
        np.save(f'{directory}/{idx}/encodedformula', outencformula)

        for t in range(len(traces_list)):
            with open(f'{directory}/{idx}/postrace{t+1}.pickle', 'wb') as trace_file:
                pickle.dump(pos_raw_list[t], trace_file)
            with open(f'{directory}/{idx}/negtrace{t+1}.pickle', 'wb') as trace_file:
                pickle.dump(neg_raw_list[t], trace_file)
            with open(f'{directory}/{idx}/encodedtrace{t+1}.pickle', 'wb') as trace_file:
                pickle.dump(traces_list[t], trace_file)

        print(f"FINISH EXAMPLE IDX: {idx}", flush=True)

if __name__ == "__main__":
    datagen("dataset/SEQ2SEQ/TRAIN", 1000)