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
            distances[i][j] = distances[j][i] = np.linalg.norm(users_grads[i] - users_grads[j]) # 默认求2范数
    return distances # 每个usr的所有item_emb和其他usr的距离 256*256

def krum(users_grads, users_count, corrupted_count, distances=None,return_index=False, debug=False):
    if users_count >= 2*corrupted_count + 1 :
        non_malicious_count = users_count - corrupted_count # krum每次排除掉给定的恶意客户数的梯度
        minimal_error = 1e20 # 初使设置的最小误差阈值，一般都会比这个小
        minimal_error_index = -1 # 表示选误差最小的

        if distances is None:
            distances = _krum_create_distances(users_grads)
        for user in distances.keys():
            errors = sorted(distances[user].values()) # 升序对距离进行排序，也就是距离差距越小的在越前面
            current_error = sum(errors[:non_malicious_count]) # 计算该用户上传梯度，累积的误差（排除了恶意梯度的情况下）
            if current_error < minimal_error:
                minimal_error = current_error # 更新最小误差
                minimal_error_index = user # 更新最小误差对应的用户索引

        if return_index: 
            return minimal_error_index #如果return_index=True,表示将krum作为一个组件，返回筛选的最小误差用户坐标给主防御策略
        else:
            return users_grads[minimal_error_index]*len(users_grads) # 如果 return_index=False, 则只使用krum，所以返回梯度
    else:
        return np.sum(users_grads,0)

def multikrum(users_grads, users_count, corrupted_count):
    if users_count >= 4*corrupted_count + 3:
        set_size = users_count - 2*corrupted_count
        sel_grads = []
        distances = _krum_create_distances(users_grads)
        while len(sel_grads) < set_size: # 迭代选择多个，直到满足set_size
            currently_selected = krum(users_grads, users_count - len(sel_grads), corrupted_count, distances, return_index=True)
            sel_grads.append(users_grads[currently_selected])
            # remove the selected from next iterations:
            distances.pop(currently_selected)
            for remaining_user in distances.keys():
                distances[remaining_user].pop(currently_selected)
        return trimmed_mean(np.array(sel_grads), len(sel_grads), 2*corrupted_count, multikrum=True) # multikrum求均值
    else:
        return np.sum(users_grads,0)

def trimmed_mean(users_grads, users_count, corrupted_count, multikrum=False):
    number_to_consider = users_count - corrupted_count # 不需要减1
    current_grads = np.empty((users_grads.shape[1],), users_grads.dtype)
    for i, param_across_users in enumerate(users_grads.T): # 针对每一维进行操作
        if param_across_users.size == 0 :
            current_grads[i] = 0.0
        elif multikrum == True:
            current_grads[i] = np.mean(param_across_users) # multikrum直接对每一维度求均值，而非trimmed_mean
        else:
            med = np.median(param_across_users) # 求中位数
            if number_to_consider != 0:
                good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[:number_to_consider] # 根据与中位数的差距选择差距小的
            else:
                good_vals = sorted(param_across_users - med, key=lambda x: abs(x))
            current_grads[i] = np.mean(good_vals) + med # 求均值再+中位数
    return current_grads*len(users_grads) 

def bulyan(users_grads, users_count, corrupted_count): # krum+trimmed_mean

    if users_count >= 4*corrupted_count + 3: # 设定的阈值，防止报错
        set_size = users_count - 2*corrupted_count
        sel_grads = [] # 保存被选中用户的梯度
        distances = _krum_create_distances(users_grads)
        while len(sel_grads) < set_size: # 迭代选择多个，直到满足set_size
            currently_selected = krum(users_grads, users_count - len(sel_grads), corrupted_count, distances=distances, return_index=True)
            sel_grads.append(users_grads[currently_selected]) # 
            # remove the selected from next iterations:
            distances.pop(currently_selected)  # 距离中移除被选中用户统计的距离
            for remaining_user in distances.keys():
                distances[remaining_user].pop(currently_selected) # 剩下用户的距离中，与被选中用户梯度计算的距离也被移除
        return trimmed_mean(np.array(sel_grads), len(sel_grads), 2*corrupted_count) # 根据multi-krum选出set_size个用户梯度后，再执行trimmed_mean
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