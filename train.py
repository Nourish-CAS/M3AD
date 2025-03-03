import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from utils.utils import find_period, sliding_window, load_anomaly, cal_sim, summarize_sim, save_model, load_model, pkl_save,find_window_lcm_multi
from model.losses import  loss_step1, loss_step2,compute_MSE
from utils.tsdata import TrainDataset
from model.M3AD import M3AD
from utils.transformation import get_cross_domain_features, get_test_features
import os, time, datetime
from tqdm import tqdm
import torch.nn as nn
import copy


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_one_epoch_step1(net, train_loader, optimizer, device):
    n_epoch_iters = 0
    train_loss = 0
    net.train(True)
    for x in train_loader:
        optimizer.zero_grad()
        org_ts, tran_ts = x[0].to(device), x[1].to(device)
        loc_results_pos, glo_results_pos, repr = net(org_ts, step1=True) # D * B * T
        loc_results_neg, glo_results_neg, repr_neg = net(tran_ts, step1=True)
        loss = loss_step1(repr, repr_neg) # D * B * T
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_epoch_iters += 1 
    train_loss /= n_epoch_iters
    return train_loss


# The purpose of validation is to maximize the similarity between pos and negatives 
def valid_one_epoch_step1(net, val_features, device):
    net.train(False)
    batches = val_features[0].shape[0]
    org_repr = []
    tran_repr = []

    for val_i in range(batches):
        org_ts = val_features[0][val_i].unsqueeze(0).to(device)
        tran_ts = val_features[1][val_i].unsqueeze(0).to(device)

        loc_results_pos, glo_results_pos, repr = net(org_ts, step1=True)
        org_res = repr.detach().cpu()
        loc_results_neg, glo_results_neg, repr_neg = net(tran_ts, step1=True)
        tran_res = repr_neg.detach().cpu()
        org_repr.append(org_res)
        tran_repr.append(tran_res)

    org_repr = torch.cat(org_repr, dim=1).to(torch.float32) # D x all_window x T
    tran_repr = torch.cat(tran_repr, dim=1).to(torch.float32) # D x all_window x T
    sim = cal_sim(org_repr, tran_repr) # D x 2B x 2B
    pos_sim, neg_sim = summarize_sim(sim) # D x B
    dist_sim = (pos_sim-neg_sim).mean()
    val_dist = dist_sim
    return val_dist



def train_one_epoch_step2(net, train_loader, optimizer, device):
    n_epoch_iters = 0
    train_loss = 0
    net.train(True)
    net.freeze_step1()
    net.unfreeze_step2()
    for x in train_loader:
        optimizer.zero_grad()
        org_ts, tran_ts = x[0].to(device), x[1].to(device)
        loc_results_pos, glo_results_pos, repr = net(org_ts, step1=True) # D * B * T
        loc_results_neg, glo_results_neg, repr_neg = net(tran_ts, step1=True)

        repr_rec = net(org_ts, loc_results=loc_results_pos, glo_results=glo_results_pos, step1=False,
                         step2 = True)
        neg_repr_rec = net(org_ts, loc_results=loc_results_pos, glo_results=glo_results_pos,
                             neg_loc=loc_results_neg, neg_glo=glo_results_neg, step1=False, step2=True)

        loss = loss_step2(repr_rec, neg_repr_rec, org_ts) # D * B * T
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_epoch_iters += 1
    train_loss /= n_epoch_iters
    return train_loss



def valid_one_epoch_step2(net, val_features, device):
    net.train(False)
    batches = val_features[0].shape[0]
    org_repr = []
    tran_repr = []
    total_loss = 0

    for val_i in range(batches):
        org_ts = val_features[0][val_i].unsqueeze(0).to(device)
        tran_ts = val_features[1][val_i].unsqueeze(0).to(device)
        loc_results_pos, glo_results_pos, repr_pos, repr = net(org_ts, all_model=True)
        org_res = repr.detach().cpu()

        rep_neg, repr_neg = net(tran_ts, loc_results=loc_results_pos, glo_results=glo_results_pos, all_model=True)
        tran_res = repr_neg.detach().cpu()
        org_repr.append(org_res)
        tran_repr.append(tran_res)
        mse_loss = compute_MSE(repr, org_ts)
        total_loss += mse_loss.item()

    org_repr = torch.cat(org_repr, dim=1).to(torch.float32) # D x all_window x T
    tran_repr = torch.cat(tran_repr, dim=1).to(torch.float32) # D x all_window x T
    sim = cal_sim(org_repr, tran_repr) # D x 2B x 2B
    pos_sim, neg_sim = summarize_sim(sim) # D x B
    dist_sim = (pos_sim-neg_sim).mean()
    val_dist = dist_sim
    avg_loss = total_loss / batches

    return val_dist, avg_loss



def train_dataset(train_data, val_data, period_len, window_size, stride, patch_size, run_dir, id, device, n_batch, lr, epochs, verbose = False):

    verbose_step2 = False
    model_fn = f'{run_dir}/ucr{id}_model.pkl'
    stride = stride
    patch_size = patch_size
    train_slices = sliding_window(train_data, window_size, stride)
    train_features, _, _ = get_cross_domain_features(train_slices, period_len, window_size)
    train_dataset = TrainDataset(train_features)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=min(len(train_dataset), n_batch), shuffle=False, drop_last=True)

    validation = True
    if len(val_data) < window_size:
        validation = False
    else:
        val_slices = sliding_window(val_data, window_size, stride)
        val_features, _, _ = get_cross_domain_features(val_slices, period_len, window_size)
    model = M3AD(window_size, device, patch_size=patch_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    # ==================================STEP 1============================================
    max_val_dist = -1e10
    best_model_state = None
    early_stop_patience = 10
    epochs_no_improve_step1 = 0
    atleast_step1 = 5
    epoch_final = 0
    for epoch in range(0, epochs):
        train_loss = train_one_epoch_step1(model, train_loader, optimizer, device)
        if validation:
            val_dist = valid_one_epoch_step1(model, val_features, device)
            if verbose:
                print(
                    f'Epoch #{epoch}: STEP_1 Training loss: {train_loss} \t\t Validation distance (distance between pos and neg): {val_dist}')
            if max_val_dist < val_dist:
                if verbose:
                    print(
                        f'STEP_1 Validation Distance Increased({max_val_dist:.6f}--->{val_dist:.6f}) \t Saving The Model')
                max_val_dist = val_dist
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve_step1 = 0
                atleast_step1 = atleast_step1 - 1
            else:
                epochs_no_improve_step1 += 1
                if verbose:
                    print(f'No improvement in validation metrics for {epochs_no_improve_step1} epoch(s).')

            if epochs_no_improve_step1 >= early_stop_patience and atleast_step1 <= 0:
                if verbose:
                    print(f'Early stopping triggered after {epochs_no_improve_step1} epochs with no improvement.')
                epoch_final = epoch
                break
        else:
            if verbose:
                print(f'Epoch #{epoch}: Training loss: {train_loss:.6f}')
            best_model_state = copy.deepcopy(model.state_dict())

    if not validation or max_val_dist == -1e10:
        best_model_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_state)
    print(f"================= STEP 1 END ============= use {epoch_final} epoch")

    # ================================== STEP 2 ============================================
    max_val_dist = -1e10
    min_recon_error = 1e10
    best_epoch = 0
    epochs_no_improve = 0
    patience = 10
    atleast_save_mod = 5
    epoch_final_step2 = 0
    for epoch in range(0, epochs):
        train_loss = train_one_epoch_step2(model, train_loader, optimizer, device)
        if validation:
            val_dist, recon_error = valid_one_epoch_step2(model, val_features, device)
            
            if verbose_step2:
                print(f'Epoch #{epoch}: Training loss: {train_loss:.6f} \t\t ' f'Validation distance: {val_dist:.6f} \t\t '
                      f'Reconstruction Error: {recon_error:.6f}')

            if (val_dist > max_val_dist) and (recon_error < min_recon_error):
                if verbose_step2:
                    print(f'Validation metrics improved (Distance: {max_val_dist:.6f} -> {val_dist:.6f}, '
                          f'Recon Error: {min_recon_error:.6f} -> {recon_error:.6f}) \t Saving the model.')
                max_val_dist = val_dist
                min_recon_error = recon_error
                best_epoch = epoch
                save_model(model, model_fn)
                epochs_no_improve = 0  # 重置耐心计数
                atleast_save_mod = atleast_save_mod - 1
            else:
                epochs_no_improve += 1
                if verbose_step2:
                    print(f'No improvement in validation metrics for {epochs_no_improve} epoch(s).')

            if epochs_no_improve >= patience and atleast_save_mod <= 0:
                if verbose_step2:
                    print(f'Early stopping triggered after {patience} epochs with no improvement.')
                epoch_final_step2 = epoch
                break

        else:

            if verbose_step2:
                print(f'Epoch #{epoch}: Training loss: {train_loss:.6f}')
            save_model(model, model_fn)

    print(f"best epoch in {best_epoch},and use {epoch_final_step2} epochs, best saved sum is{atleast_save_mod}")

    if not validation or max_val_dist == -1e10:
        save_model(model, model_fn)
        if verbose:
            print('Final model saved.')

    if verbose:
        print(f'Training completed. Best model found at epoch {best_epoch} with '
              f'Validation distance: {max_val_dist:.6f} and '
              f'Reconstruction Error: {min_recon_error:.6f}.')

def test_eval(model, test_ft, device):
    model.eval()
    batches = test_ft[0].shape[0]
    repr = []
    repr_kl = []
    MSE = []
    for test_i in range(batches):
        org_ts = test_ft[0][test_i].unsqueeze(0).to(device)
        loc_results_pos, glo_results_pos, res, res_kl = model(org_ts, all_model=True)
        res = res.detach().cpu()
        mse_loss = compute_MSE(res_kl, org_ts)  # 计算MSE损失
        res_kl = res_kl.detach().cpu()  # D x B x T
        mse_loss =mse_loss.detach().cpu()
        MSE.append(mse_loss)
        repr.append(res)
        repr_kl.append(res_kl)
    repr = torch.cat(repr,dim=1).to(torch.float32) # D x B x T
    z = F.normalize(repr, p=2, dim=2)
    sim = torch.abs(torch.matmul(z, z.transpose(1, 2))) # D x B x B
    # Remove the similarity between instance itself
    sim_updated = torch.tril(sim, diagonal=-1)[:, :, :-1]    # D x B x (B-1)
    sim_updated += torch.triu(sim, diagonal=1)[:, :, 1:]
    scores = sim_updated.mean(dim=-1).numpy()
    repr_kl  = torch.cat(repr_kl,dim=1).to(torch.float32) # D x B x T
    z_kl = F.normalize(repr_kl, p=2, dim=2)
    z_kl = z_kl.transpose(0, 1) # B x D x T
    sim_kl = torch.abs(torch.matmul(z_kl, z_kl.transpose(1, 2)))  # D x B x B
    sim_kl_updated = torch.tril(sim_kl, diagonal=-1)[:, :, :-1]    # B x D x (D-1)
    sim_kl_updated += torch.triu(sim_kl, diagonal=1)[:, :, 1:] # B x D x (D-1)
    scores_kl = sim_kl_updated.transpose(0, 1).mean(dim=-1).numpy()
    MSE = np.array(MSE)
    return scores, scores_kl, MSE


def check_range_100(suspects, anomaly_indices, stride, window_size, threshold=100):
    suspects = np.atleast_1d(suspects)
    anomaly_indices = np.atleast_1d(anomaly_indices)
    for s_idx in suspects:
        s_start = s_idx * stride
        s_end = s_start + window_size - 1
        s_start_t = s_start - threshold
        s_end_t = s_end + threshold
        for a_idx in anomaly_indices:
            a_start = a_idx * stride
            a_end = a_start + window_size - 1
            if not (s_end_t < a_start or s_start_t > a_end):
                return True

    return False


if __name__ == '__main__':
    torch.cuda.manual_seed_all(0)
    device = torch.device('cuda')

    params = {'cycles': 2.5, 'epochs': 150, 'repr_dims': 32,
              'stride_ratio': 4, 'alpha': 0.4, 'SEED': 4, 'batch_size':32, 'lr': 0.001}
    
    train_x, valid_x, test_x, test_y = load_anomaly("./dataset/ucr_data.pt")
    id_list = list(train_x.keys())

    all_results = []
    run_dir = f'./trained/'
    os.makedirs(run_dir, exist_ok=True)
    alpha = params['alpha']
    cycles = params['cycles']
    epoch = params['epochs']
    lr = params['lr']
    batch_size = params['batch_size']
    ratio = params['stride_ratio']
    SEED = params['SEED']
    total_inference_time = 0.0
    drop_10 = False

    for i in tqdm(range(0, len(id_list)), miniters=1):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        id = id_list[i]
        train_data = train_x[id]
        val_data = valid_x[id]
        test_data = test_x[id]
        test_labels = test_y[id]
        res_notebook = {}
        period_len = find_period(train_data, id)

        if period_len < 50:
            cycles = 2.5
        if period_len > 200:
            cycles = 1.5
        if drop_10 == True:
            train_data = train_data[len(train_data)//5:]
        window_size = round(cycles * period_len)
        window_size, patch_size = find_window_lcm_multi(window_size)
        stride = window_size // ratio

        train_dataset(train_data, val_data, period_len, window_size, stride,
                      patch_size, run_dir, id, device, batch_size, lr, epoch)

        # test
        test_slices = sliding_window(test_data, window_size, stride)
        test_ft = get_test_features(test_slices, period_len)

        model = M3AD(window_size, device, patch_size=patch_size).to(device)
        model_fn = f'{run_dir}/ucr{id}_model.pkl'
        load_model(model, model_fn)

        t0 = time.time()
        scores, scores_kl, mse_local_glbal = test_eval(model, test_ft, device)
        inference_time = time.time() - t0
        total_inference_time += inference_time
        loc_anom = np.argmin(scores[0])
        glo_anom = np.argmin(scores[1])
        sim_local_global = np.argmin(scores_kl[0])
        mse_local_glbal = np.argmax(mse_local_glbal)

        label_slices = sliding_window(test_labels, window_size, stride)
        index_slices = sliding_window(np.arange(len(test_data)), window_size, stride)
        win_indices = np.where(np.any(label_slices == 1, axis=1))[0]
        suspects = np.unique([loc_anom, glo_anom, sim_local_global, mse_local_glbal])
        is_within_anom = check_range_100(suspects, win_indices, stride, window_size, threshold=100)

        res_notebook['id'] = id
        res_notebook['tri_detected'] = is_within_anom
        res_notebook['num_suspects'] = len(suspects)
        res_notebook['suspects'] = index_slices[suspects]
        res_notebook['inference'] = datetime.timedelta(seconds=inference_time)
        all_results.append(res_notebook)

        is_within_anom_obs = check_range_100(loc_anom, win_indices, stride, window_size, threshold=100)
        is_within_anom_freq = check_range_100(glo_anom, win_indices, stride, window_size, threshold=100)
        is_within_anom_res = check_range_100(sim_local_global, win_indices, stride, window_size, threshold=100)
        is_within_anom_mse = check_range_100(mse_local_glbal, win_indices, stride, window_size, threshold=100)

        if is_within_anom:
            tqdm.write(f"ucr {id}: anomaly DETECTED")
        else:
            tqdm.write(f"ucr {id}: anomaly MISS")

        if is_within_anom_obs:
            tqdm.write("The anomaly is identified in the comparison module(local)")
        if is_within_anom_freq:
            tqdm.write("The anomaly is identified in the comparison module(global)")
        if is_within_anom_res or is_within_anom_mse:
            tqdm.write("The anomaly is identified in the reconstruction module")
        print(f"[{id}] Inference time: {inference_time:.4f} s")


    pkl_save(f'./tri_res.pt', all_results)
    all_results_df = pd.DataFrame(all_results)
    acc = sum(all_results_df['tri_detected']) / len(all_results_df)
    print(f"tri-window prediction accuracy: {acc}")
    print(f"Total inference time for all samples: {total_inference_time:.4f} s")

