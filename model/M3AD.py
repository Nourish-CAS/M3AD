from .embed import *
from .RevIN import RevIN
from .Self_Attention_Family import SelfAttentionBlock,ConCrossAttn_Rec
from .gltcn import *
from einops import rearrange, repeat

def min_max_scale(signal, original_signal):
    original_min = torch.min(original_signal)
    original_max = torch.max(original_signal)
    scaled_signal = (signal - original_min) / (original_max - original_min)
    return scaled_signal

class model_step2(nn.Module):
    def __init__(self, patch_size, d_model, cross_attention_layer=1, dropout=0.15):
        super(model_step2, self).__init__()
        self.patch_size = patch_size
        self.d_model_cross = d_model
        self.projection_glo = nn.Linear(d_model, 1)
        self.projection_loc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu1 = nn.ReLU()
        self.cross_attention_layer = cross_attention_layer
        self.cross_attention = nn.ModuleList()

        for i in range(self.cross_attention_layer):
            self.cross_attention.append(ConCrossAttn_Rec(
            input_dims=self.d_model_cross,

        ))

        self.patch_weights = nn.Parameter(torch.ones(len(patch_size)), requires_grad=True)

    def forward(self, loc_results, glo_results, loc_neg=None, glo_neg=None, attn_mask=None):

        patch_weights = torch.softmax(self.patch_weights, dim=0)
        loc_glo_att_patch = 0
        glo_loc_att_patch = 0
        loc_glo_atts = []
        glo_loc_atts = []

        for patch_index, patchsize in enumerate(self.patch_size):

            if loc_neg is not None and glo_neg is not None:
                loc_neg_fusion = loc_neg[patch_index]
                glo_neg_fusion = glo_neg[patch_index]
                local_fusion = loc_results[patch_index]
                global_fusion = glo_results[patch_index]

                for attn_tri_layer in range(self.cross_attention_layer):
                    loc_glo = self.cross_attention[attn_tri_layer](local_fusion, glo_neg_fusion)
                    glo_loc = self.cross_attention[attn_tri_layer](global_fusion, loc_neg_fusion)
                    glo_neg_fusion = glo_loc
                    loc_neg_fusion = loc_glo
                    local_fusion = loc_glo
                    global_fusion = glo_loc

            else:
                local_fusion = loc_results[patch_index]
                global_fusion = glo_results[patch_index]

                for attn_tri_layer in range(self.cross_attention_layer):
                    loc_glo = self.cross_attention[attn_tri_layer](local_fusion, global_fusion)
                    glo_loc = self.cross_attention[attn_tri_layer](global_fusion, local_fusion)
                    local_fusion = loc_glo
                    global_fusion = glo_loc

            loc_glo_atts.append(local_fusion)
            glo_loc_atts.append(global_fusion)

        for patch_index, patchsize in enumerate(self.patch_size):
            loc_glo = repeat(loc_glo_atts[patch_index], 'b l n -> b (l repeat_m) n', repeat_m=self.patch_size[patch_index])
            glo_loc = repeat(glo_loc_atts[patch_index], 'b l n -> b (l repeat_m) n', repeat_m=self.patch_size[patch_index])
            loc_glo_att_patch += loc_glo * patch_weights[patch_index]
            glo_loc_att_patch += glo_loc * patch_weights[patch_index]

        loc_glo_att_patch = self.projection_loc(loc_glo_att_patch)
        glo_loc_att_patch = self.projection_glo(glo_loc_att_patch)
        repr_rec = torch.cat([loc_glo_att_patch.unsqueeze(0), glo_loc_att_patch.unsqueeze(0)], dim=0)
        repr_rec = repr_rec.squeeze(-1)  # D * B * T * 1 ——> D * B * T

        return repr_rec



class model_step1(nn.Module):
    def __init__(self, win_size, device, num_heads_step1=1, d_model=64, patch_size=[5, 10], channel=55,
                 output_attention=False):
        super(model_step1, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size
        self.channel = channel
        self.win_size = win_size
        self.d_model = d_model
        self.device = device
        self.num_heads_step1 = num_heads_step1

        # Patching List
        self.embedding_patch_size = nn.ModuleList()
        self.pos_encoder = nn.ModuleList()
        self.tcl_loc = nn.ModuleList()
        self.tcl_glo = nn.ModuleList()
        self.local_attention = nn.ModuleList()
        self.global_attention = nn.ModuleList()

        for i, patchsize in enumerate(self.patch_size):
            patch_num = self.win_size // patchsize
            self.embedding_patch_size.append(TokenEmbedding(patchsize, d_model))
            self.pos_encoder.append(PositionalEncoding(d_model, 0.1, patch_num))
            self.tcl_loc.append(Tcn_Local(d_model))
            self.tcl_glo.append(Tcn_Global(num_inputs=patch_num, num_outputs=d_model))
            self.local_attention.append(SelfAttentionBlock(
            embed_dim=d_model,
            num_heads= self.num_heads_step1
        ))
            self.global_attention.append(SelfAttentionBlock(
            embed_dim=d_model,
            num_heads=self.num_heads_step1
        ))


        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.projection = nn.Linear(d_model, 1)
        self.patch_weights = nn.Parameter(torch.ones(len(patch_size)), requires_grad=True)

    def forward(self, ts, M):
        patch_weights = torch.softmax(self.patch_weights, dim=0)
        loc_results = []
        glo_results = []
        loc_att_patch = 0
        glo_att_patch = 0
        revin_layer = RevIN(num_features=M)
        # Instance Normalization Operation
        ts = revin_layer(ts, 'norm')

        for patch_index, patchsize in enumerate(self.patch_size):
            ts_patch_size = ts
            ts_patch_size = rearrange(ts_patch_size, 'b l m -> b m l')  # Batch channel win_size
            ts_patch_size = rearrange(ts_patch_size, 'b m (n p) -> (b m) n p', p=patchsize)
            # embedding channel
            ts_patch_size = self.embedding_patch_size[patch_index](ts_patch_size)
            # loc tcn
            ts_loc = self.tcl_loc[patch_index](ts_patch_size.transpose(1, 2))
            ts_loc = ts_loc.permute(2, 0, 1) * math.sqrt(self.d_model)
            ts_loc = self.relu1(ts_loc)
            ts_loc = ts_loc.transpose(0, 1) # BS,patch num(length),dim

            # glo tcn
            ts_glo = self.tcl_glo[patch_index](ts_patch_size.transpose(1, 2))  # BS dim len
            ts_glo = ts_glo.permute(2, 0, 1) * math.sqrt(self.d_model)  # len bs dim
            ts_glo = self.relu2(ts_glo)
            ts_glo = ts_glo.transpose(0, 1) # BS,patch num(length),dim
            ts_loc = self.local_attention[patch_index](ts_loc)
            ts_glo = self.global_attention[patch_index](ts_glo)
            loc_results.append(ts_loc)
            glo_results.append(ts_glo)

        for patch_index, patchsize in enumerate(self.patch_size):
            ts_loc = repeat(loc_results[patch_index] , 'b l n -> b (l repeat_m) n', repeat_m=self.patch_size[patch_index])
            ts_glo = repeat(glo_results[patch_index], 'b l n -> b (l repeat_m) n', repeat_m=self.patch_size[patch_index])
            loc_att_patch += ts_loc * patch_weights[patch_index]
            glo_att_patch += ts_glo * patch_weights[patch_index]

        loc_att_patch = revin_layer(loc_att_patch, 'denorm')
        glo_att_patch = revin_layer(glo_att_patch, 'denorm')

        repr = torch.cat([loc_att_patch.unsqueeze(0), glo_att_patch.unsqueeze(0)], dim=0)  # Domain(D) * B * T * C
        repr = self.relu1(repr)
        repr = self.projection(repr).squeeze(-1) # D * B * T * 1 ——> D * B * T

        return loc_results, glo_results, repr



class M3AD(nn.Module):
    def __init__(self, win_size, device, num_heads_step1=1, num_heads_step2=1, cross_attention_layer=1, d_model=64, patch_size=[5, 10], channel=55,
                 output_attention=True):
        super(M3AD, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size
        self.channel = channel
        self.win_size = win_size
        self.d_model = d_model
        self.device = device
        self.num_heads_step1 = num_heads_step1
        self.num_heads_step2 = num_heads_step2
        self.cross_attention_layer = cross_attention_layer

        self.model_step1 = model_step1(self.win_size, self.device, self.num_heads_step1, self.d_model, self.patch_size, self.channel)
        self.model_step2 = model_step2(self.patch_size, self.d_model, self.cross_attention_layer)

    def freeze_step1(self):
        for param in self.model_step1.parameters():
            param.requires_grad = False

    def unfreeze_step1(self):
        for param in self.model_step1.parameters():
            param.requires_grad = True

    def freeze_step2(self):
        for param in self.model_step2.parameters():
            param.requires_grad = False

    def unfreeze_step2(self):
        for param in self.model_step2.parameters():
            param.requires_grad = True

    def forward(self, ts, loc_results=None, glo_results=None, neg_loc=None, neg_glo=None, step1=False, step2=False, all_model=False):

        B, L, M = ts.shape  # Batch win_size channel

        if step1:
            loc_results, glo_results, repr = self.model_step1(ts, M)
            return loc_results, glo_results, repr

        elif step2:
            if neg_loc is not None and neg_glo is not None:
                neg_repr_rec = self.model_step2(loc_results, glo_results, loc_neg=neg_loc, glo_neg=neg_glo)
                return neg_repr_rec

            else:
                repr_rec = self.model_step2(loc_results, glo_results)
                return repr_rec

        elif all_model:
            if loc_results is not None and glo_results is not None:
                neg_loc, neg_glo, repr_neg = self.model_step1(ts, M)
                repr_rec_neg = self.model_step2(loc_results, glo_results, neg_loc, neg_glo)
                return repr_neg, repr_rec_neg

            else:
                loc_results, glo_results, repr_pos = self.model_step1(ts, M)
                repr_rec = self.model_step2(loc_results, glo_results)
                return loc_results, glo_results, repr_pos, repr_rec

        else:
            print("Please output the correct guidance.")



