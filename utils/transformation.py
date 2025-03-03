
import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import torch, random
from .utils import robust_scaling_2d


def gen_jitering(ts, window_size):
    num_samples = len(ts)
    noise_amplitude = (ts.max()-ts.min())*1/5
    combined_signal = ts + noise_amplitude * np.random.randn(num_samples)
    anom_len = np.random.randint(window_size//20, window_size//3)
    start_index = np.random.randint(0, len(ts)-anom_len)
    end_index = start_index+anom_len
    modified_ts = ts.copy()
    modified_ts[start_index:end_index] = combined_signal[start_index:end_index]
    return modified_ts, (start_index,end_index)


def gen_warping(ts, fft_values, window_size, verbose = False):

    psd_values = np.abs(fft_values) ** 2
    peak_indices = np.argsort(psd_values)[-30:]

    frequencies = np.fft.fftfreq(len(ts), d=1)
    frequencies = frequencies[peak_indices]
    frequencies = np.unique(frequencies[frequencies>0])
    frequencies = np.sort(frequencies) # frequency sorted from lowest to highest

    pick_idx = np.arange(0, len(frequencies), 3)
    cutoff = np.random.choice(frequencies[pick_idx][0:4], size=2, replace=False)
    low_freq = min(cutoff)  
    high_freq = max(cutoff)
    b, a = signal.butter(4, [low_freq, high_freq], btype='band')
    filtered_signal_lower = signal.lfilter(b, a, ts)
    original = ts.reshape(-1, 1)
    filtered_signal_lower = filtered_signal_lower.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(original.min(), original.max()))
    scaler.fit(original)
    filtered_signal_lower = scaler.transform(filtered_signal_lower).flatten()
    anom_len = np.random.randint(window_size//20, window_size//3)
    start_index = np.random.randint(0, len(ts)-anom_len)
    end_index = start_index+anom_len

    # Copy the original array
    modified_ts = ts.copy()
    modified_ts[start_index:end_index] = filtered_signal_lower[start_index:end_index]
    if verbose:
        print(f'Lower frequency transformation: filter between {low_freq} and {high_freq}, anomaly length: {anom_len}')
    
    return modified_ts, (start_index,end_index)

def gen_scaling(ts, window_size):
    num_samples = len(ts)
    scaling_factors = np.random.uniform(0.5, 1.5, num_samples)
    scaled_signal = ts * scaling_factors
    anom_len = np.random.randint(window_size // 20, window_size // 3)
    start_index = np.random.randint(0, len(ts) - anom_len)
    end_index = start_index + anom_len
    modified_ts = ts.copy()
    modified_ts[start_index:end_index] = scaled_signal[start_index:end_index]

    return modified_ts, (start_index, end_index)


def gen_permutation(ts, window_size):
    num_segments = np.random.randint(3, 10)
    segment_length = len(ts) // num_segments

    segments = [ts[i:i + segment_length] for i in range(0, len(ts), segment_length)]
    np.random.shuffle(segments)
    permuted_signal = np.concatenate(segments)
    if len(permuted_signal) < len(ts):
        permuted_signal = np.concatenate([permuted_signal, ts[len(permuted_signal):]])
    anom_len = np.random.randint(window_size // 20, window_size // 3)
    start_index = np.random.randint(0, len(ts) - anom_len)
    end_index = start_index + anom_len
    modified_ts = ts.copy()
    modified_ts[start_index:end_index] = permuted_signal[start_index:end_index]

    return modified_ts, (start_index, end_index)



def get_cross_domain_features(ts_slices, period_len, window_size):
    tran_ts = []
    anom_indx = []
    h_freq_num = 0
    total_slices = len(ts_slices)
    num_jittering = int(0.4 * total_slices)
    num_warping = int(0.4 * total_slices)
    num_scaling = int(0.1 * total_slices)
    num_permutation = int(0.1 * total_slices)

    # 构建增强方法的列表
    augmentations = (["jittering"] * num_jittering +
                     ["warping"] * num_warping +
                     ["scaling"] * num_scaling +
                     ["permutation"] * num_permutation)

    while len(augmentations) < total_slices:
        augmentations.append("jittering")
    while len(augmentations) > total_slices:
        augmentations.pop()
    random.shuffle(augmentations)

    for slice, augmentation_type in zip(ts_slices, augmentations):
        fft_values = np.fft.fft(slice)

        if augmentation_type == "jittering":
            modified, anom = gen_jitering(slice, window_size)
        elif augmentation_type == "warping":
            modified, anom = gen_warping(slice, fft_values, window_size, verbose=False)
        elif augmentation_type == "scaling":
            modified, anom = gen_scaling(slice, window_size)
        elif augmentation_type == "permutation":
            modified, anom = gen_permutation(slice, window_size)

        tran_ts.append(modified)
        anom_indx.append(anom)

    slices = robust_scaling_2d(ts_slices)
    tran_ts = robust_scaling_2d(tran_ts)

    org_ts = torch.Tensor(slices).unsqueeze(dim=-1) # B*T*1
    tran_ts = torch.Tensor(np.array(tran_ts)).unsqueeze(dim=-1)

    features = [org_ts, tran_ts]
    return features, h_freq_num, anom_indx

def get_test_features(ts_slices, period_len):
    slices = robust_scaling_2d(ts_slices)
    org_ts = torch.Tensor(slices).unsqueeze(dim=-1)
    features = [org_ts]
    return features