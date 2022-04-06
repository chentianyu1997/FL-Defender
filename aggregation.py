import copy
import enum
import torch
import numpy as np
import math
from scipy import stats
from functools import reduce
import time
import sklearn.metrics.pairwise as smp
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from utils import *

eps = np.finfo(float).eps

#################################################################################################################

def flatten_models_gradients(global_model, local_models):
    global_model = torch.nn.utils.parameters_to_vector(global_model).cpu().data.numpy()
    models = [torch.nn.utils.parameters_to_vector(model.parameters()).cpu().data.numpy() for model in local_models]
    models = [global_model - model for model in models]
    return np.asarray(models)

def get_pca(data):
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    return data

class FLDefender:
    def __init__(self, num_peers):
        self.grad_history =  None
        self.fgrad_history =  None
        self.num_peers = num_peers

    def score(self, global_model, local_models, peers_types, selected_peers, epoch, tau = 1.5):
        global_model = list(global_model.parameters())
        last_g = global_model[-2].cpu().data.numpy()
        m = len(local_models)
        f_grads = [None for i in range(m)]
        for i in range(m):
            grad= (last_g - \
                    list(local_models[i].parameters())[-2].cpu().data.numpy())
            f_grads[i] = grad.reshape(-1)
    
        cs = smp.cosine_similarity(f_grads) - np.eye(m)
        cs = get_pca(cs)
        centroid = np.median(cs, axis = 0)
        scores = smp.cosine_similarity([centroid], cs)[0]

        return scores
#################################################################################################################


def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output

def Repeated_Median_Shard(w):
    SHARD_SIZE = 100000
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        if total_num < SHARD_SIZE:
            slopes, intercepts = repeated_median(y)
            y = intercepts + slopes * (len(w) - 1) / 2.0
        else:
            y_result = torch.FloatTensor(total_num).to(device)
            assert total_num == y.shape[0]
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y_shard = y[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                slopes_shard, intercepts_shard = repeated_median(y_shard)
                y_shard = intercepts_shard + slopes_shard * (len(w) - 1) / 2.0
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE] = y_shard
            y = y_result
        y = y.reshape(shape)
        w_med[k] = y
    return w_med


def repeated_median(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts


# Repeated Median estimator
def Repeated_Median(w):
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        slopes, intercepts = repeated_median(y)
        y = intercepts + slopes * (len(w) - 1) / 2.0

        y = y.reshape(shape)
        w_med[k] = y

    print('repeated median aggregation took {}s'.format(time.time() - cur_time))
    return w_med


# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv


class FoolsGold:
    def __init__(self, num_peers):
        self.memory = None
        self.wv_history = []
        self.num_peers = num_peers
       
    def score_gradients(self, global_model, local_models, selectec_peers):
        global_model = list(global_model.parameters())
        m = len(local_models)
        grads = [None for i in range(m)]
        for i in range(m):
            grads[i]= (global_model[-2].cpu().data.numpy() - \
                    list(local_models[i].parameters())[-2].cpu().data.numpy()).reshape(-1)
        
        grads = np.asarray(grads)
        if self.memory is None:
            self.memory = np.zeros((self.num_peers, len(grads[0])))

        self.memory[selectec_peers]+= grads
        wv = foolsgold(self.memory)  # Use FG
        self.wv_history.append(wv)
        return wv[selectec_peers]
        
# simple median estimator
def simple_median(w):
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        median_result = median_opt(y)
        assert total_num == len(median_result)

        weight = torch.reshape(median_result, shape)
        w_med[k] = weight
    return w_med

def trimmed_mean(w, trim_ratio):
    if trim_ratio == 0:
        return average_weights(w, [1 for i in range(len(w))])
        
    assert trim_ratio < 0.5, 'trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        result = y_sorted[:, trim_num:-trim_num]
        result = result.mean(dim=-1)
        assert total_num == len(result)

        weight = torch.reshape(result, shape)
        w_med[k] = weight
    return w_med


# Get average weights
def average_weights(w, marks):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] *(1/sum(marks))
    return w_avg
   
def Krum(updates, f, multi = False):
    n = len(updates)
    updates = [torch.nn.utils.parameters_to_vector(update.parameters()) for update in updates]
    updates_ = torch.empty([n, len(updates[0])])
    for i in range(n):
      updates_[i] = updates[i]
    k = n - f - 2
    # collection distance, distance from points to points
    cdist = torch.cdist(updates_, updates_, p=2)
    dist, idxs = torch.topk(cdist, k , largest=False)
    dist = dist.sum(1)
    idxs = dist.argsort()
    if multi:
      return idxs[:k]
    else:
      return idxs[0]
##################################################################
