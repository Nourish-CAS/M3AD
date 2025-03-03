import torch
import torch.nn.functional as F
import torch.nn as nn


def loss_step1(z1, z2, alpha=0.4):
    loss = alpha * cross_domain(z1) + (1 - alpha) * intra_domain_loss(z1, z2)
    return loss

def loss_step2(z_pos, z_neg, t, beta=0.6):
    loss = beta * compute_MSE(z_pos, t) + (1 - beta) * cross_domain_loss(z_pos, z_neg)
    return loss

def intra_domain_loss(z1, z2):
    z1 = F.normalize(z1, p=2, dim=2)
    z2 = F.normalize(z2, p=2, dim=2)
    D, B = z1.size(0), z1.size(1)
    z = torch.cat([z1, z2], dim=1)  # D x 2B x T
    sim = torch.abs(torch.matmul(z, z.transpose(1, 2)))  # D x 2B x 2B

    sim_updated = torch.tril(sim, diagonal=-1)[:, :, :-1]    # D x 2B x (2B-1)
    sim_updated += torch.triu(sim, diagonal=1)[:, :, 1:]
    pos1 = torch.exp(sim_updated[:, 0:B, 0:B - 1]).sum(dim=-1).unsqueeze(-1)  # D x B x 1
    neg = torch.exp((sim_updated[:, 0:B, B - 1:] + sim_updated[:, B:, 0:B]) / 2).sum(dim=-1).unsqueeze(-1)  # D x B x 1

    logits = torch.cat([pos1, neg], dim=-1)  # D x B x 2
    logits = -torch.log(logits[:, :, 0:1] / (logits.sum(dim=-1).unsqueeze(-1)))
    loss = logits.mean()
    return loss

def cross_domain(z1):
    z1 = F.normalize(z1, p=2, dim=2)
    sim = torch.abs(torch.matmul(z1, z1.transpose(1, 2))) # D x B x B
    sim_updated = torch.tril(sim, diagonal=-1)[:, :, :-1]    # D x B x (B-1)
    sim_updated += torch.triu(sim, diagonal=1)[:, :, 1:] # D x B x (B-1)
    pos = torch.exp(sim_updated).sum(dim=-1).unsqueeze(-1)  # D x B x 1

    z = z1.transpose(0, 1)  # B x D x T
    sim = torch.abs(torch.matmul(z, z.transpose(1, 2))) # B x D x D
    sim_updated = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x D x (D-1)
    sim_updated += torch.triu(sim, diagonal=1)[:, :, 1:] # B x D x (D-1)
    neg = torch.exp(sim_updated).sum(dim=-1)  # B x D
    neg = neg.transpose(0, 1).unsqueeze(-1) # D x B x 1
    logits = torch.cat([pos, neg], dim = -1) # D x B x 2
    logits = -torch.log(logits[:,:,0:1]/(logits.sum(dim=-1).unsqueeze(-1)))
    loss = logits.max()
    return loss


def cross_domain_loss(z1,z2):
    z1 = F.normalize(z1, p=2, dim=2)
    z2 = F.normalize(z2, p=2, dim=2)
    z_pos = z1.transpose(0, 1)  # B x D x T
    sim_pos = torch.abs(torch.matmul(z_pos, z_pos.transpose(1, 2)))  # B x D x D
    sim_updated_pos = torch.tril(sim_pos, diagonal=-1)[:, :, :-1]  # B x D x (D-1)
    sim_updated_pos += torch.triu(sim_pos, diagonal=1)[:, :, 1:]  # B x D x (D-1)
    pos = torch.exp(sim_updated_pos).sum(dim=-1)  # B x D
    pos = pos.transpose(0, 1).unsqueeze(-1)  # D x B x 1

    z_neg = z2.transpose(0, 1)  # B x D x T
    sim_neg = torch.abs(torch.matmul(z_neg, z_neg.transpose(1, 2)))  # B x D x D
    sim_updated_neg = torch.tril(sim_neg, diagonal=-1)[:, :, :-1]  # B x D x (D-1)
    sim_updated_neg += torch.triu(sim_neg, diagonal=1)[:, :, 1:]  # B x D x (D-1)
    neg = torch.exp(sim_updated_neg).sum(dim=-1)  # B x D
    neg = neg.transpose(0, 1).unsqueeze(-1)  # D x B x 1

    logits = torch.cat([pos, neg], dim=-1)  # D x B x 2
    logits = -torch.log(logits[:, :, 0:1] / (logits.sum(dim=-1).unsqueeze(-1)))
    loss = logits.mean()
    return loss


def compute_MSE(z,t):
    assert z.shape[0] == 2, "The first dimension of z_mean should be 2."
    z = z.unsqueeze(-1)
    act = nn.Sigmoid()
    z = act(z)
    glo, loc = z[0], z[1]
    ts_rec = glo + loc
    loss_fn = nn.MSELoss(reduction='none')
    loss = loss_fn(t, ts_rec)
    loss = torch.mean(loss)
    return loss
