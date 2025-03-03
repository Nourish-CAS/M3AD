import torch
import torch.nn as nn
from .DilatedConv import dilated_conv_net
import torch.nn.functional as F
from .RevIN import RevIN


class ConCrossAttn_Rec(torch.nn.Module):
    """Transformer language model.
    """

    def __init__(self, input_dims, double_receptivefield=3):
        super().__init__()

        self.input_dims = input_dims
        self.embed_dim = input_dims * 2
        self.revin = RevIN(num_features=input_dims)

        self.q_func = dilated_conv_net(in_channel=input_dims, out_channel=self.embed_dim, bottleneck=self.embed_dim // 8,
                                       double_receptivefield=double_receptivefield)
        self.k_func = dilated_conv_net(in_channel=input_dims, out_channel=self.embed_dim, bottleneck=self.embed_dim // 8,
                                       double_receptivefield=double_receptivefield)
        self.v_func = dilated_conv_net(in_channel=input_dims, out_channel=self.embed_dim, bottleneck=self.embed_dim // 8,
                                       double_receptivefield=double_receptivefield)

        # identity matrix because we are using convs for in_projections
        self.in_proj_weight = torch.concat(
            (torch.eye(self.embed_dim), torch.eye(self.embed_dim), torch.eye(self.embed_dim))).requires_grad_(False)
        self.out_proj = nn.Linear(self.embed_dim, input_dims)

    def forward(self, query_in, key_in):
        query_in = self.revin(query_in, "norm").transpose(1, 2)  # batch_size, num_features, sequence_length,

        key_in = self.revin(key_in, "norm").transpose(1, 2)

        q_out = self.q_func(query_in).permute(2, 0, 1)  # Time, Batch, Channel
        k_out = self.k_func(key_in).permute(2, 0, 1)
        v_out = self.v_func(key_in).permute(2, 0, 1)

        reconstruction, attn_weights = F.multi_head_attention_forward(
            query=q_out, key=k_out, value=v_out,
            out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias,
            in_proj_weight=self.in_proj_weight.to(q_out.device),
            need_weights=self.training,
            ### can ignore everything else, which is just default values used to make function work ###
            in_proj_bias=None, bias_k=None, bias_v=None,
            embed_dim_to_check=self.embed_dim, num_heads=1, use_separate_proj_weight=False,
            add_zero_attn=False, dropout_p=0.1, training=self.training, )

        reconstruction = self.revin(reconstruction.permute(1, 0, 2), "denorm")  # shape [batch_size, length, embed_dim]

        return reconstruction


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.15):
        super(SelfAttentionBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        input = input.permute(1, 0, 2)
        self_attn_output, _ = self.self_attention(input, input, input)
        output_att = input + self.dropout(self_attn_output)
        output_att = self.norm(output_att)
        ffn_output = self.ffn(output_att.permute(1, 0, 2))
        output = output_att.permute(1, 0, 2) + self.dropout(ffn_output)
        return output
