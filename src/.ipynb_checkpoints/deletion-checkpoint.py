# Utility Functions for deletion.

import random

import networkx as nx
import numpy as np
import itertools

from copy import deepcopy


# -

def get_cyclic(L, B=None):
    if B is None:
        B = L
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


def delete(perms, slice_id):
    del_perms = []
    for p in perms:
        if slice_id in p:
            idx = p.index(slice_id)
            if len(p[:idx]) > 0:
                del_perms.append(p[:idx])
        else:
            del_perms.append(p)
    return del_perms


def check_s3t_completion(all_models):
    for k in all_models.keys():
        if len(all_models[k]) > 0:
            return True
    return False


# +
def get_budgeted_models(num_shards, num_slices, budget):
    all_models = {}
    for _ in range(num_shards):
        perms = get_cyclic(num_slices)
        all_models[_] = perms[:budget]
    return all_models

def get_full_budget(num_shards, num_slices):
    all_models = {}
    for _ in range(num_shards):
        perms = list(itertools.permutations(list(range(num_slices))))
        all_models[_] = perms
    return all_models

def get_sisa_models(num_shards, num_slices):
    all_models = {}
    for _ in range(num_shards):
        all_models[_] = [list(range(num_slices))]
    return all_models


# -

def s3t(all_models, num_shards, num_slices):
    N = num_shards * num_slices
    
    count = 0
    while check_s3t_completion(all_models):
        d = random.randint(0, N-1)
        shard_id = d // num_slices
        slice_id = d % num_slices
        avail_models = delete(all_models[shard_id], slice_id)
        all_models[shard_id] = avail_models
        count += 1
    return count


def budgeted_s3t(num_shards, num_slices, budget):
    all_models = get_budgeted_models(num_shards, num_slices, budget)
    count = s3t(all_models, num_shards, num_slices)
    return count


for i in range(1, 7):
    counts = []
    for _ in range(100):
        counts.append(budgeted_s3t(5, 6, i))
    print(f"Budget: {i}, Avg. Count: {np.mean(counts)}, Std. Count: {np.std(counts)}")


# +
def check_completion(delete_set, protected_set):
    for s in protected_set:
        if s not in delete_set:
            return False
    return True
    
def get_sisa_count(num_shards, num_slices):
    N = num_shards * num_slices
    protected_set = list(range(num_shards))
    deleted_set = []
    while not check_completion(deleted_set, protected_set):
        d = random.randint(0, N-1) // num_slices
        deleted_set.append(d)
    return len(deleted_set)

