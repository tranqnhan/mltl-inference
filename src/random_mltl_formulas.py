# Author: Zili Wang
# Generates random MTL formulas
# Arguments: 
#      -samples : number of formulas to generate
#      -n : number of variables named p0, p1, ... (default: 5)
#      -d : maximum depth of the formula (default: 2)
#      -p : probability of choosing temporal operators (default: 0.5)
#      -delta : maximum delta between i and j in [i, j] (default: 10)
#      -m : maximum bound value - max of all i and j in intervals [i, j] (default: 10)
#      -seed: seed for random number generator (default: None)
#      (--help will produce a usage message)
# 
# Output: writes to ./n[n]_d[d]_p[p]_delta[delta]_m[m]_seed[seed]/formulas.txt

import random

def generate_random_formula(n:int, d:int, p:float, delta:int, m:int):
    '''
    Generates a random MTL formula
    Arguments:
        n : number of variables named p0, p1, ...
        d : maximum depth of the formula
        p : probability of choosing temporal operators
        delta : maximum delta between i and j in [i, j]
        m : maximum bound value - max of all i and j in intervals [i, j]
    Returns:
        a random MTL formula
    '''
    def random_formula(depth):
        if depth == 0:
            return [f"p{random.randint(0, n-1)}"]
        if random.random() < p: # temporal operator
            i = random.randint(0, m)
            j = random.randint(i, min(i+delta, m))
            op = random.choice(["F", "G", "U", "R"])
            if op in ["U", "R"]:
                f1 = random_formula(depth-1)
                f2 = random_formula(depth-1)
                formula = ["("] + f1 + [op, "["]+ list(str(i)) + [","] + list(str(j)) + ["]"] + f2 + [")"]
                return formula #f"({f1} {op}[{i},{j}] {f2})"
            else: # op in ["F", "G"]
                f = random_formula(depth-1)
                formula = [op, "["] + list(str(i)) + [","] + list(str(j)) + ["]"] + f 
                return formula #f"{op}[{i},{j}] {f}"
        else: # boolean operator
            op = random.choice(["&", "|", "!"])
            if op == "!":
                f = random_formula(depth-1)
                formula = ["!"] + f
                return formula #f"!{f}"
            else: # op in ["&", "|"]
                f1 = random_formula(depth-1)
                f2 = random_formula(depth-1)
                formula = ["("] + f1 + [op] + f2 + [")"]
                return formula #f"({f1} {op} {f2})"

    return random_formula(d)