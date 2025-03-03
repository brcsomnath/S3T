# +
import os
import math
import random
import itertools
import numpy as np


from tqdm import tqdm
from random import choices 
from collections import defaultdict
from itertools import combinations, permutations
from scipy.optimize import linear_sum_assignment


# -

# ## BMS with scoring

def get_score(probs, t=100):
    c = 0 
    n = 0
    score = 0
    for i, p in enumerate(probs):
        c += p
        n += 1
        score += n*(1-c)**t
    return score


def total_score(all_perms, probs):
    s = 0
    for perm in all_perms:
        P = [probs[n] for n in perm]
        s += get_score(P)
    return s


def edit_dist(p1, p2):
    d = 0
    for a, b in zip(p1, p2):
        d += 0 if a == b else 1
    return d


def avg_edit_dist(all_perms):
    scores = []
    for i in range(len(all_perms)):
        for j in range(len(all_perms)):
            if i != j:
                scores.append(edit_dist(all_perms[i], all_perms[j]))
    return np.mean(scores)


def bms_selection(probs):
    L = len(probs)
    elements = list(range(L))
    all_permutations = []

    if len(all_permutations) == 0:
        for e in elements:
            all_permutations.append([e])

    for _ in range(L-1):
        cost = np.outer(probs, probs)
        for i, perm in enumerate(all_permutations):
            for j in range(L):
                if j in perm:
                    cost[i][j] = -float('inf')
                    continue
                prob_dist = [probs[e] for e in perm + [j]]
                cost[i][j] = get_score(prob_dist)


        row_ind, col_ind = linear_sum_assignment(cost, maximize=True)

        for i, c in zip(row_ind, col_ind):
            all_permutations[i].append(c)
    return all_permutations

print(bms_selection(probs = [0.4, 0.3, 0.2, 0.1]))


# ## Conditional Sampling

def conditional_sampling(probs):
    L = len(probs)
    elements = list(range(L))
    sampling_probs = 1-np.array(probs)
    all_permutations = []
    
    for _ in range(L):
        while True:
            perm = []
            
            for j in range(L):
                p = choices(elements, sampling_probs)
                while p[0] in perm:
                    p = choices(elements, sampling_probs)
                perm.append(p[0])
            
            if perm not in all_permutations:
                all_permutations.append(perm)
                break
    return all_permutations


print(conditional_sampling(probs = [0.4, 0.3, 0.2, 0.1]))


# ## Cyclic Rotation

# +
def sample_perm(L):
    elements = list(range(L))
    perm = []
    for _ in range(L):
        sample_set = [e for e in elements if e not in perm]
        perm.append(random.choice(sample_set))
    return perm

def random_sampling(L, B):
    all_permutations = []
    while len(all_permutations) < B:
        perm = sample_perm(L)
        while perm in all_permutations:
            perm = sample_perm(L)
        all_permutations.append(perm)
    return all_permutations


# -

def rotate_right(lst, k):
    k = k % len(lst)
    return lst[-k:] + lst[:-k]


def get_rotations(lst):
    rotations = []
    for i in range(len(lst)):
        rotations.append(rotate_right(lst, i))
    return rotations


def get_cyclic(L, B):
    elements = list(range(L))
    all_permutations = []
    n_iter = 0
    cum_len = 1    
    
    while len(all_permutations) < B:
        branch_fact = len(all_permutations) // L
        cum_len = cum_len * (L-n_iter)
                
        if n_iter == 0:
            perms = get_rotations(elements)
            all_permutations = perms[:B]
        else:
            per_perm = B // len(all_permutations)
            
            perms = []
            for i, p in enumerate(all_permutations):
                prefix = p[:n_iter]
                all_rotations = get_rotations(p[n_iter:])
                perms.extend([prefix + suffix for suffix in all_rotations[:per_perm]])
                

            if len(perms) < cum_len:
                rem = B % len(all_permutations)
                total = 0
                start = 0
                start_counter = 0
                
                while total < rem:
                    if start >= len(all_permutations):
                        start_counter += 1
                        start = start_counter
                    
                    p = all_permutations[start]
                    prefix = p[:n_iter]
                    new_perm = prefix + get_rotations(p[n_iter:])[per_perm]
                    perms.append(new_perm)
                    perms = sorted(perms)

                    start += branch_fact
                    total += 1

            all_permutations = perms
    
        n_iter += 1
    return all_permutations

print(get_cyclic(4, 13))


