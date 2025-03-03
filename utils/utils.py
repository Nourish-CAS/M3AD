import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def load_anomaly(name):
    res = pkl_load(name)
    return res['train_data'], res['valid_data'], \
           res['test_data'],  res['test_labels']

def save_model(model, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(model.state_dict(), fn)


def load_model(model, fn, device = torch.device('cuda')):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=device)
        model.load_state_dict(state_dict)


def find_period(time_series, id):
    spectrum = np.fft.fft(time_series)
    psd = np.abs(spectrum) ** 2
    pos_freq_indices = np.where(np.fft.fftfreq(len(time_series)) >= 0)[0]
    top3_indices = pos_freq_indices[np.argsort(psd[pos_freq_indices])[::-1][:3]]
    freq = np.fft.fftfreq(len(time_series), d=1)
    top3_preiod = np.round(1/np.abs(freq[top3_indices])).astype(int)
    long_period_list = [str(number) for number in list(range(213, 222))] + ['239','240','241']
    period_opt = np.sort(top3_preiod)
    period_len = period_opt[-1]
    # Rectify the period_len
    if period_len < 20:
        period_len = 100
    elif period_len > 500:
        if id in long_period_list:
            period_len = 1000
        elif 20 <= period_opt[0] <= 500:
            period_len = period_opt[0]
        else:
            period_len = 200

    return period_len

# A function to slice the time series
def sliding_window(time_series, window_size, stride=1):
    n = len(time_series)
    num_slices = (n - window_size) // stride + 1
    slices = np.zeros((num_slices, window_size), dtype=time_series.dtype)
    for i in range(num_slices):
        start = i * stride
        end = start + window_size
        slices[i] = time_series[start:end]
    return slices

# A function to normalize the input time series slices of dim T*window_size by each window 
def normalize_arr_2d(array, dim=1):
    norm_array = np.linalg.norm(array, axis=dim, keepdims=True)
    res = array / norm_array
    return res

# A function of robust scaling by (X - X_median) / IQR across the rows
def robust_scaling_2d(array, dim = 1):
    median = np.median(array, axis=dim, keepdims=True)
    iqr = np.percentile(array, 75, axis=dim, keepdims=True) - np.percentile(array, 25, axis=1, keepdims=True)
    epsilon = 1e-7
    scaled_array = (array - median) / (iqr+epsilon)
    return scaled_array

# A function to check if the identified window index is within the ground truth anomaly range
def check_range(numbers, start, end):
    arr = np.array(numbers)
    return np.any((arr >= start) & (arr <= end))

# z dim: G * B * T 
def cal_sim(z1, z2):
    z1 = F.normalize(z1, p=2, dim=2)
    z2 = F.normalize(z2, p=2, dim=2)
    B, T = z1.size(1), z1.size(2) 
    z = torch.cat([z1, z2], dim=1)  # G x 2B x T
    sim = torch.abs(torch.matmul(z, z.transpose(1, 2))) # G x 2B x 2B
    return sim

def summarize_sim(sim):
    D = sim.shape[0]
    B = sim.shape[1]//2
    sim_updated = torch.tril(sim, diagonal=-1)[..., :-1]    # G x 2B x (2B-1)
    sim_updated += torch.triu(sim, diagonal=1)[..., 1:]
    if B > 1:
        pos_sim = sim_updated[:, 0:B,0:B-1].mean(dim=2)
        neg_sim = ((sim_updated[:, 0:B,B-1:]+sim_updated[:, B:, 0:B])/2).mean(dim=2)
        return pos_sim, neg_sim # G x B
    if B == 1:
        pos_sim = torch.ones((D, B))
        neg_sim = sim_updated.mean(dim=1)
        return pos_sim, neg_sim

def sim_MSE(z,t):
    assert z.shape[0] == 2, "The first dimension of z_mean should be 2."
    act = nn.Sigmoid()
    z = act(z)
    glo, loc = z[0], z[1]
    loss_fn = nn.MSELoss(reduction='none')
    loss_glo = loss_fn(t,glo)
    loss_loc = loss_fn(t,loc)
    loss = loss_loc + loss_glo
    loss = torch.mean(loss)
    return loss


def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def find_window_lcm_multi(original_window_size, max_k_offset=2):

    if original_window_size < 1:
        return None

    if original_window_size < 100:
        ratio_candidates = [
            [45, 30, 15]
        ]
    elif original_window_size < 500:
        ratio_candidates = [
            [45, 30, 15]
        ]
    elif original_window_size < 1000 and original_window_size > 500:
        ratio_candidates = [
            [45, 30, 15]
        ]
    else:
        ratio_candidates = [
            [45, 30, 15]
        ]

    best_solution = None
    best_diff = float("inf")

    for ratios in ratio_candidates:
        L = 1
        for r in ratios:
            L = lcm(L, r)

        k_approx = round(original_window_size / L)
        if k_approx < 1:
            k_approx = 1

        k_min = max(1, k_approx - max_k_offset)
        k_max = k_approx + max_k_offset

        for k in range(k_min, k_max + 1):
            W_candidate = L * k
            diff = abs(W_candidate - original_window_size)

            patch_sizes = [W_candidate // r for r in ratios]

            if any(p < 1 for p in patch_sizes):
                continue
            if diff < best_diff:
                best_diff = diff
                best_solution = (W_candidate, patch_sizes)

    return best_solution
