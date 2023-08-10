import numpy as np
from collections import defaultdict, OrderedDict
from parse import args
import torch.nn as nn
import torch


class DefenseTypes:
    Krum = 'Krum'
    MultiKrum = 'MultiKrum'
    TrimmedMean = 'TrimmedMean'
    Bulyan = 'Bulyan'
    Median = 'Median'

    def __str__(self):
        return self.value

def _krum_create_distances(users_grads):
    distances = defaultdict(dict)
    for i in range(len(users_grads)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(users_grads[i] - users_grads[j]) 
    return distances 

def krum(users_grads, users_count, corrupted_count, distances=None,return_index=False, debug=False):
    if users_count >= 2*corrupted_count + 1 :
        non_malicious_count = users_count - corrupted_count
        minimal_error = 1e20 
        minimal_error_index = -1

        if distances is None:
            distances = _krum_create_distances(users_grads)
        for user in distances.keys():
            errors = sorted(distances[user].values()) 
            current_error = sum(errors[:non_malicious_count])）
            if current_error < minimal_error:
                minimal_error = current_error 
                minimal_error_index = user 

        if return_index: 
            return minimal_error_index 
        else:
            return users_grads[minimal_error_index]*len(users_grads) 
    else:
        return np.sum(users_grads,0)

def multikrum(users_grads, users_count, corrupted_count):
    if users_count >= 4*corrupted_count + 3:
        set_size = users_count - 2*corrupted_count
        sel_grads = []
        distances = _krum_create_distances(users_grads)
        while len(sel_grads) < set_size:
            currently_selected = krum(users_grads, users_count - len(sel_grads), corrupted_count, distances, return_index=True)
            sel_grads.append(users_grads[currently_selected])
            # remove the selected from next iterations:
            distances.pop(currently_selected)
            for remaining_user in distances.keys():
                distances[remaining_user].pop(currently_selected)
        return trimmed_mean(np.array(sel_grads), len(sel_grads), 2*corrupted_count, multikrum=True)
    else:
        return np.sum(users_grads,0)

def trimmed_mean(users_grads, users_count, corrupted_count, multikrum=False):
    number_to_consider = users_count - corrupted_count
    current_grads = np.empty((users_grads.shape[1],), users_grads.dtype)
    for i, param_across_users in enumerate(users_grads.T): 
        if param_across_users.size == 0 :
            current_grads[i] = 0.0
        elif multikrum == True:
            current_grads[i] = np.mean(param_across_users)
        else:
            med = np.median(param_across_users) # 求中位数
            if number_to_consider != 0:
                good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[:number_to_consider]
            else:
                good_vals = sorted(param_across_users - med, key=lambda x: abs(x))
            current_grads[i] = np.mean(good_vals) + med
    return current_grads*len(users_grads) 

def bulyan(users_grads, users_count, corrupted_count): # krum+trimmed_mean
    if users_count >= 4*corrupted_count + 3: 
        set_size = users_count - 2*corrupted_count
        sel_grads = [] 
        distances = _krum_create_distances(users_grads)
        while len(sel_grads) < set_size:
            currently_selected = krum(users_grads, users_count - len(sel_grads), corrupted_count, distances=distances, return_index=True)
            sel_grads.append(users_grads[currently_selected]) 
            # remove the selected from next iterations:
            distances.pop(currently_selected)  
            for remaining_user in distances.keys():
                distances[remaining_user].pop(currently_selected)
        return trimmed_mean(np.array(sel_grads), len(sel_grads), 2*corrupted_count)
    else:
        return np.sum(users_grads,0)

def median(users_grads, users_count, corrupted_count):
    current_grads = np.empty((users_grads.shape[1],), users_grads.dtype)
    for i, param_across_users in enumerate(users_grads.T):
        if param_across_users.size == 0 :
            current_grads[i] = 0.0
        else:
            med = np.median(param_across_users)
            current_grads[i] =  med
    return current_grads*len(users_grads)


defend = {DefenseTypes.Krum: krum,
          DefenseTypes.MultiKrum: multikrum,
          DefenseTypes.TrimmedMean: trimmed_mean, 
          DefenseTypes.Bulyan: bulyan,
          DefenseTypes.Median: median
          }
