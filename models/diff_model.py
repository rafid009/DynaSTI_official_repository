import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.transformer import EncoderLayer, PositionalEncoding
import numpy as np

from torch.jit import Final
import scipy.sparse as sp
# from timm.models.layers import a # use_fused_attn
# from timm.models.vision_transformer import PatchEmbed, Attention

from models.layers import *


# torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    padding = dilation * ((kernel_size - 1)//2)
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding, bias=bias)
    # layer = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv1d_with_init_saits(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    # layer = nn.utils.weight_norm(layer)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv2d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

############################### CSDI ################################

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
    
class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x




############################### New Design ################################

# def swish(x):
#     return x * torch.sigmoid(x)


    

def Conv1d_with_init_saits_new(in_channels, out_channels, kernel_size, init_zero=False, dilation=1):
    padding = dilation * ((kernel_size - 1)//2)
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
    # layer = nn.utils.weight_norm(layer)
    if init_zero:
        nn.init.zeros_(layer.weight)
    else:
        nn.init.kaiming_normal_(layer.weight)
    return layer
    


# def conv_with_init(in_channels, out_channel, kernel_size):
#     layer = nn.Conv2d(in_channels, out_channel, kernel_size, stride=2)
#     nn.init.kaiming_normal_(layer.weight)
#     return layer



class GTA(nn.Module):
    def __init__(self, channels, d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
            diffusion_embedding_dim=128, diagonal_attention_mask=True) -> None:
        super().__init__()

        # combi 2
        self.enc_layer_1 = EncoderLayer(d_time, actual_d_feature, channels, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)
        
        self.enc_layer_2 = EncoderLayer(d_time, actual_d_feature, 2 * channels, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.init_proj = Conv1d_with_init_saits_new(d_model, channels, 1)
        self.conv_layer = Conv1d_with_init_saits_new(channels, 2 * channels, kernel_size=1)

        self.cond_proj = Conv1d_with_init_saits_new(d_model, channels, 1)
        self.conv_cond = Conv1d_with_init_saits_new(channels, 2 * channels, kernel_size=1)


        self.res_proj = Conv1d_with_init_saits_new(channels, d_model, 1)
        self.skip_proj = Conv1d_with_init_saits_new(channels, d_model, 1)


    def forward(self, x, cond, diffusion_emb):
        # x Noise
        # L -> time
        # K -> feature
        B, L, K = x.shape

        x_proj = torch.transpose(x, 1, 2) # (B, K, L)
        x_proj = self.init_proj(x_proj)

        cond = torch.transpose(cond, 1, 2) # (B, K, L)
        cond = self.cond_proj(cond)
        

        diff_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1) # 
        y = x_proj + diff_proj #+ cond

        # attn1
        y = torch.transpose(y, 1, 2) # (B, L, channels)
        y, attn_weights_1 = self.enc_layer_1(y)
        y = torch.transpose(y, 1, 2)


        y = self.conv_layer(y)
        c_y = self.conv_cond(cond)
        y = y + c_y


        y = torch.transpose(y, 1, 2) # (B, L, 2*channels)
        y, attn_weights_2 = self.enc_layer_2(y)
        y = torch.transpose(y, 1, 2)
 

        y1, y2 = torch.chunk(y, 2, dim=1)
        out = torch.sigmoid(y1) * torch.tanh(y2) # (B, channels, L)

        residual = self.res_proj(out) # (B, K, L)
        residual = torch.transpose(residual, 1, 2) # (B, L, K)

        skip = self.skip_proj(out) # (B, K, L)
        skip = torch.transpose(skip, 1, 2) # (B, L, K)


        attn_weights = (attn_weights_1 + attn_weights_2) / 2 #torch.softmax(attn_weights_1 + attn_weights_2, dim=-1)

        return (x + residual) * math.sqrt(0.5), skip, attn_weights



class SADI(nn.Module):
    def __init__(self, diff_steps, diff_emb_dim, n_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v,
            dropout, diagonal_attention_mask=True, is_simple=False, ablation_config=None):
        super().__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.is_simple = is_simple
        self.d_feature = d_feature
        channels = d_model #int(d_model / 2)
        self.ablation_config = ablation_config
        self.d_time = d_time
        self.n_head = n_head
        
        # self.spatial_context_embeddimg = SpatialDescriptor(d_time * d_feature + ablation_config['spatial_context_dim'], ablation_config['h_channels'], d_feature)

        self.layer_stack_for_first_block = nn.ModuleList([
            GTA(channels=channels, d_time=d_time, actual_d_feature=actual_d_feature, 
                        d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                        diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
            for _ in range(n_layers)
        ])
        if self.ablation_config['is_2nd_block']:
            self.layer_stack_for_second_block = nn.ModuleList([
                GTA(channels=channels, d_time=d_time, actual_d_feature=actual_d_feature, 
                            d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                            diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
                for _ in range(n_layers)
            ])

            self.embedding_2 = nn.Linear(actual_d_feature, d_model)
            self.reduce_dim_beta = nn.Linear(d_model, d_feature)
            self.reduce_dim_z = nn.Linear(d_model, d_feature)

        self.diffusion_embedding = DiffusionEmbedding(diff_steps, diff_emb_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.position_enc_cond = PositionalEncoding(d_model, n_position=d_time)
        self.position_enc_noise = PositionalEncoding(d_model, n_position=d_time)

        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.embedding_cond = nn.Linear(actual_d_feature, d_model)
        self.reduce_skip_z = nn.Linear(d_model, d_feature)
        
        if self.ablation_config['weight_combine']:
            self.weight_combine = nn.Linear(d_feature + d_time, d_feature)
        
        
        if self.ablation_config['fde-choice'] == 'fde-conv-single':
            self.mask_conv = Conv1d_with_init_saits_new(2 * self.d_feature, self.d_feature, 1)
            self.layer_stack_for_feature_weights = nn.ModuleList([
                EncoderLayer(d_feature, d_time, d_time, d_inner, 1, d_time, d_time, dropout, 0,
                            self.ablation_config['fde-diagonal'], choice='fde-conv-single')
                for _ in range(self.ablation_config['fde-layers'])
            ])
        elif self.ablation_config['fde-choice'] == 'fde-conv-multi':
            self.mask_conv = Conv1d_with_init_saits_new(2 * self.d_feature, self.d_feature, 1)
            if not self.ablation_config['is-fde-linear']:
                self.layer_stack_for_feature_weights = nn.ModuleList([
                    EncoderLayer(d_feature, d_time, d_time, d_inner, n_head, d_time, d_time, dropout, 0,
                                self.ablation_config['fde-diagonal'], choice='fde-conv-multi')
                    for _ in range(self.ablation_config['fde-layers'])
                ])
            else:
                self.layer_stack_for_feature_weights = nn.ModuleList([
                    EncoderLayer(d_feature, d_time, d_time, d_inner, n_head, 64, 64, dropout, 0,
                                self.ablation_config['fde-diagonal'], choice='fde-conv-multi', is_linear=True)
                    for _ in range(self.ablation_config['fde-layers'])
                ])
            if self.ablation_config['fde-pos-enc']:
                self.fde_pos_enc = PositionalEncoding(d_time, n_position=d_feature)

            if self.ablation_config['fde-time-pos-enc']:
                self.fde_time_pos_enc = PositionalEncoding(d_feature, n_position=d_time)
        else:
            self.mask_conv = Conv1d_with_init_saits_new(2, 1, 1)
            self.layer_stack_for_feature_weights = nn.ModuleList([
                EncoderLayer(d_feature, d_time, d_time, d_inner, n_head, d_time, d_time, dropout, 0,
                            self.ablation_config['fde-diagonal'])
                for _ in range(self.ablation_config['fde-layers'])
            ])

    # ds3
    def forward(self, inputs, diffusion_step):
        X, masks = inputs['X'], inputs['missing_mask']
        masks[:,1,:,:] = masks[:,0,:,:]
        # B, L, K -> B=batch, L=time, K=feature
        X = torch.transpose(X, 2, 3)
        masks = torch.transpose(masks, 2, 3)
        # Feature Dependency Encoder (FDE): We are trying to get a global feature time-series cross-correlation
        # between features. Each feature's time-series will get global aggregated information from other features'
        # time-series. We also get a feature attention/dependency matrix (feature attention weights) from it.
        if self.ablation_config['is_fde'] and self.ablation_config['is_first']:
            cond_X = X[:,0,:,:] + X[:,1,:,:] # (B, L, K)
            shp = cond_X.shape
            if not self.ablation_config['fde-no-mask']:
                # In one branch, we do not apply the missing mask to the inputs of FDE
                # and in the other we stack the mask with the input time-series for each feature
                # and embed them together to get a masked informed time-series data for each feature.
                cond_X = torch.stack([cond_X, masks[:,1,:,:]], dim=1) # (B, 2, L, K)
                cond_X = cond_X.permute(0, 3, 1, 2) # (B, K, 2, L)
                cond_X = cond_X.reshape(-1, 2 * self.d_feature, self.d_time) # (B, 2*K, L)
                # print(f"cond before mask: {cond_X.shape}")
                cond_X = self.mask_conv(cond_X) # (B, K, L)
                # print(f"cond before posenc: {cond_X.shape}")
                if self.ablation_config['fde-pos-enc']:
                    cond_X = self.fde_pos_enc(cond_X) # (B, K, L)

                if self.ablation_config['fde-time-pos-enc']:
                    cond_X = torch.transpose(cond_X, 1, 2) # (B, L, K)
                    cond_X = self.fde_time_pos_enc(cond_X) # (B, L, K)
                    cond_X = torch.transpose(cond_X, 1, 2) # (B, K, L)
            else:
                cond_X = torch.transpose(cond_X, 1, 2) # (B, K, L)

            for feat_enc_layer in self.layer_stack_for_feature_weights:
                cond_X, _ = feat_enc_layer(cond_X) # (B, K, L), (B, K, K)

            cond_X = torch.transpose(cond_X, 1, 2) # (B, L, K)
        else:
            cond_X = X[:,1,:,:]
        
        input_X_for_first = torch.cat([cond_X, masks[:,1,:,:]], dim=2)
        input_X_for_first = self.embedding_1(input_X_for_first)

        noise = input_X_for_first
        cond = torch.cat([X[:,0,:,:], masks[:,0,:,:]], dim=2)
        cond = self.embedding_cond(cond)

        diff_emb = self.diffusion_embedding(diffusion_step)
        pos_cond = self.position_enc_cond(cond)
        
        enc_output = self.dropout(self.position_enc_noise(noise))
        skips_tilde_1 = torch.zeros_like(enc_output)

        for encoder_layer in self.layer_stack_for_first_block:
            enc_output, skip, _ = encoder_layer(enc_output, pos_cond, diff_emb)
            skips_tilde_1 += skip

        skips_tilde_1 /= math.sqrt(len(self.layer_stack_for_first_block))
        skips_tilde_1 = self.reduce_skip_z(skips_tilde_1)

        if self.ablation_config['is_2nd_block']:
            X_tilde_1 = self.reduce_dim_z(enc_output)
            X_tilde_1 = X_tilde_1 + skips_tilde_1 + X[:, 1, :, :]

            input_X_for_second = torch.cat([X_tilde_1, masks[:,1,:,:]], dim=2)
            input_X_for_second = self.embedding_2(input_X_for_second)
            noise = input_X_for_second

            enc_output = self.position_enc_noise(noise)
            skips_tilde_2 = torch.zeros_like(enc_output)

            for encoder_layer in self.layer_stack_for_second_block:
                enc_output, skip, attn_weights = encoder_layer(enc_output, pos_cond, diff_emb)
                skips_tilde_2 += skip

            skips_tilde_2 /= math.sqrt(len(self.layer_stack_for_second_block))
            skips_tilde_2 = self.reduce_dim_beta(skips_tilde_2) 

            if self.ablation_config['weight_combine']:
                attn_weights = attn_weights.squeeze(dim=1)  
                if len(attn_weights.shape) == 4:
                    attn_weights = torch.transpose(attn_weights, 1, 3)
                    attn_weights = attn_weights.mean(dim=3)
                    attn_weights = torch.transpose(attn_weights, 1, 2)
                
                combining_weights = torch.sigmoid(
                    self.weight_combine(torch.cat([masks[:, 0, :, :], attn_weights], dim=2))
                )

                skips_tilde_3 = (1 - combining_weights) * skips_tilde_1 + combining_weights * skips_tilde_2
            else:
                skips_tilde_3 = skips_tilde_2
        if self.ablation_config['is_2nd_block']:
            if self.ablation_config['weight_combine']:
                skips_tilde_1 = torch.transpose(skips_tilde_1, 1, 2)
                skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2)
                skips_tilde_3 = torch.transpose(skips_tilde_3, 1, 2)
            else:
                skips_tilde_3 = torch.transpose(skips_tilde_3, 1, 2)
                skips_tilde_1 = None
                skips_tilde_2 = None
        else:
            skips_tilde_3 = torch.transpose(skips_tilde_1, 1, 2)
            skips_tilde_1 = None
            skips_tilde_2 = None

        return skips_tilde_1, skips_tilde_2, skips_tilde_3
    
    

################## DiT Model ####################


def modulate(x, shift, scale):
    # print(f"modulate:: x: {x.shape}, shift: {shift.shape}, scale: {scale.shape}")
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention"""

    def __init__(self, temperature, attn_dropout=0.1, sigma=2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.sigma = sigma

    def forward(self, q, k, v, attn_mask=None, is_spat=False):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


def generate_subsequent_mask(seq_length: int) -> torch.Tensor:
    """
    Generate a square mask for the sequence. The masked positions are filled with -inf.
    Unmasked positions are filled with 0. This mask is used to mask out future positions.
    
    Args:
        seq_length (int): The length of the sequence.
        
    Returns:
        torch.Tensor: A tensor of shape (seq_length, seq_length) containing the mask.
    """
    # Create a square matrix filled with 1s above the main diagonal.
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    # Convert mask values to -inf for masked positions and 0 for unmasked positions.
    # mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask

class Attn(nn.Module):
    """original Transformer multi-head attention"""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout=0.1, choice='linear', d_channel=-1, is_linear=False, kernel_size=3, dilation=1, sigma=2):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.choice = choice
        self.d_model = d_model
        self.is_linear = is_linear
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(d_k ** 0.5, attn_dropout, sigma=sigma)
        

    def forward(self, q, k, v, attn_mask=None, is_spat=False):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv

        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # attn_mask = generate_subsequent_mask(len_q)
        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, so needs broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)  # For batch and head axis broadcasting.

        v, attn_weights = self.attention(q, k, v, attn_mask, is_spat=is_spat)
        # print(f"v after attn: {v.shape}")
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        # print(f"v after attn+fc: {v.shape}")
        return v, attn_weights
    

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            is_cross: bool = False,
            dk: int = 64
    ) -> None:
        super().__init__()
        self.is_cross = is_cross
        self.num_heads = num_heads
        

        
        # if not self.is_cross:
        #     assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        #     self.head_dim = dim // num_heads
        # else:
        self.head_dim = dk
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()
        
        self.Wq = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.Wk = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.Wv = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        
        if not self.is_cross:
            q = self.Wq(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.Wk(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.Wv(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            # q, k, v = qkv.unbind(0)
        else:
            len_k = c.shape[1]
            q = self.Wq(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.Wk(c).reshape(B, len_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.Wv(c).reshape(B, len_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)

        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p=self.attn_drop.p if self.training else 0.,
        #     )
        # else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.head_dim * self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x
       
class EncoderLayer1(nn.Module):
    def __init__(self, d_time, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, attn_dropout=0.1,
                 diagonal_attention_mask=False, choice='linear', is_ffn=True, is_linear=False, feature_size=-1, kernel_size=1, dilation=1):
        super().__init__()

        self.diagonal_attention_mask = diagonal_attention_mask
        self.d_time = d_time
        # self.d_feature = d_feature

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = Attn(n_head, d_model, d_k, d_v, attn_dropout, choice=choice, d_channel=feature_size, is_linear=is_linear, kernel_size=kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.is_ffn = is_ffn
        if self.is_ffn:
            self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)
        else:
            self.pos_ffn = None

    def forward(self, enc_input, mask_time=None):
        if self.diagonal_attention_mask:
            mask_time = torch.eye(self.d_time).to(enc_input.device)
 
        residual = enc_input
        # here we apply LN before attention cal, namely Pre-LN, refer paper https://arxiv.org/abs/2002.04745
        enc_input = self.layer_norm(enc_input)
        # enc_output, attn_weights = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=mask_time)
        enc_output, attn_weights = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=mask_time)
        enc_output = self.dropout(enc_output)
        # print(f"enc_output: {enc_input.shape}")
        enc_output += residual

        if self.is_ffn:
            enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, d_k=64, d_v=64, mlp_ratio=4.0, choice='time', kernel_size=3, feature_size=-1, dilation=1, sigma=2, is_cross=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.choice = choice
        # if is_cross:
        self.attn = Attn(num_heads, hidden_size, d_k, d_v, d_channel=feature_size, choice=choice, kernel_size=kernel_size, dilation=dilation, sigma=sigma)
        # else:
        #     self.attn = MLA(hidden_size, num_heads)
        # self.attn = Attention(hidden_size, num_heads, dk=d_k, is_cross=is_cross, qkv_bias=True)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()#approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.is_cross = is_cross

    def forward(self, x, c, is_spat=False):
        # print(f"DiT block:: x: {x.shape}, c: {c.shape}")
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        # print(f"shift: {shift_msa.shape}")
        if self.is_cross:
            v = self.norm1(x)
        else:
            v = modulate(self.norm1(x), shift_msa, scale_msa)

        # print(f"v: {v.shape}")
        # x = x + gate_msa.unsqueeze(1) * self.attn(v, v, v)
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # if self.is_cross:
        #     attn, attn_weights = self.attn(v, c, c, is_spat=is_spat)
        # else:
        #     attn, attn_weights = self.attn(v, v, v, is_spat=is_spat)
        
        if self.is_cross:
            attn, attn_weights = self.attn(v, c, c)
        else:
            attn, attn_weights = self.attn(v, v, v)
            # attn, attn_weights = self.attn(v, self.choice)

            # attn_weights = None
        if self.is_cross:
            x = x + attn
            x = x + self.mlp(self.norm2(x))
        else:
            x = x + gate_msa * attn
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, attn_weights


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, d_feature, is_cross=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, d_feature, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.is_cross = is_cross

    def forward(self, x, c):
        if self.is_cross:
            x= self.norm_final(x)
        else:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
            x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class FeatureTemporalEncoder(nn.Module):
    def __init__(self, d_time, d_k, d_v, hidden_size, num_heads, mlp_ratio, dilation=1) -> None:
        super().__init__()
        # self.fde = EncoderLayer1(2 * hidden_size, d_time, d_time, num_heads, d_time, d_time, choice='fde', feature_size=2 * hidden_size, kernel_size=3, dilation=dilation)
        self.fde = DiTBlock(d_time, num_heads, d_k=d_time, d_v=d_time, feature_size=2 * hidden_size, mlp_ratio=mlp_ratio, choice='fde', kernel_size=3, dilation=dilation)
        # self.gta = EncoderLayer1(d_time, 2 * hidden_size, hidden_size, num_heads, d_k, d_v)
        self.gta = DiTBlock(2 * hidden_size, num_heads, d_k, d_v, mlp_ratio=mlp_ratio)
        self.init_proj = Conv1d_with_init(hidden_size, 2 * hidden_size, kernel_size=1)
        self.cond_proj = Conv1d_with_init(hidden_size, 2 * hidden_size, kernel_size=1)

        self.res_proj = Conv1d_with_init_saits_new(hidden_size, hidden_size, 1)
        self.skip_proj = Conv1d_with_init_saits_new(hidden_size, hidden_size, 1)

    def forward(self, x, c):
        B, L, D = x.shape

        noise = x.permute((0, 2, 1)) # B, D, L
        cond = c.permute((0, 2, 1))  # B, D, L

        noise = self.init_proj(noise) # B, 2D, L
        cond = self.cond_proj(cond) # B, 2D, L


        # noise = noise + cond
        # noise = self.fde(noise) # B, 2D, L
        noise = self.fde(noise, cond) # B, 2D, L


        noise = noise.permute((0, 2, 1)) # B, L, 2D
        cond = cond.permute((0, 2, 1)) # B, L, 2D

        noise = self.gta(noise, cond) # B, L, 2D
        # noise = noise + cond
        # noise = self.gta(noise)

        noise = noise.permute((0, 2, 1)) # B, 2D, L
        cond = cond.permute((0, 2, 1))  # B, 2D, L

        y1, y2 = torch.chunk(noise, 2, dim=1) # B, D, L  B, D, L
        out = torch.sigmoid(y1) * torch.tanh(y2) # (B, D, L)

        residual = self.res_proj(out) # (B, D, L)
        residual = torch.transpose(residual, 1, 2) # (B, L, D)

        skip = self.skip_proj(out) # (B, D, L)
        skip = torch.transpose(skip, 1, 2) # (B, L, D)

        return (x + residual) * math.sqrt(0.5), skip





class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        ablation_config,
        d_time,
        d_feature,
        n_spatial,
        # diff_steps,
        # diff_emb_dim,
        hidden_size=256,
        d_k=64,
        d_v=64,
        n_layer=4,
        n_spatial_layer=4,
        num_heads=8,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.d_time = d_time
        self.d_feature = d_feature
        self.n_spatial = n_spatial
        self.num_heads = num_heads
        self.ablation_config = ablation_config

        self.x_embedder = nn.Linear(self.d_feature * self.n_spatial * 2, hidden_size)
        # self.x_embedder = nn.Linear(self.d_feature * 2, hidden_size)

        # flattened_size = self.n_spatial * hidden_size
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_embedder = nn.Linear(self.d_feature * n_spatial * 2, hidden_size)
        # self.cond_embedder = nn.Linear(self.d_feature * 2, hidden_size)

        self.spatial_context_embeddimg = nn.Linear(self.ablation_config['spatial_context_dim'], self.ablation_config['h_channels'])  # SpatialDescriptor(self.ablation_config['spatial_context_dim'], d_time * d_feature, self.ablation_config['h_channels'], d_feature)
        
        self.noise_spat_embed = nn.Linear(self.d_feature * self.d_time + self.ablation_config['h_channels'], self.d_feature * self.d_time)
        self.cond_spat_embed = nn.Linear(self.d_feature * self.d_time + self.ablation_config['h_channels'], self.d_feature * self.d_time)

        
        self.X_pos_enc = PositionalEncoding(hidden_size, n_position=d_time)
        self.cond_pos_enc = PositionalEncoding(hidden_size, n_position=d_time)
        sigma = self.ablation_config['sigma']
        


        if ablation_config['fde']:
            self.fde_blocks = nn.ModuleList([
                DiTBlock(d_time, num_heads, d_k=d_time, d_v=d_time, feature_size=hidden_size, mlp_ratio=mlp_ratio, choice='linear', kernel_size=3, dilation=(i+1)) for i in range(n_layer)
            ])
            self.n_block_layers = len(self.fde_blocks)

        if ablation_config['gta']:
            self.blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads, d_k, d_v, mlp_ratio=mlp_ratio) for _ in range(n_layer)
            ])
            self.n_block_layers = len(self.blocks)

        if ablation_config['spatial']:

            # self.spatial_block = GraphAttentionV2Layer(self.d_time * self.d_feature + self.ablation_config['h_channels'], self.d_time * self.d_feature + self.ablation_config['h_channels'], n_heads=num_heads)

            self.spatial_blocks = nn.ModuleList([
                DiTBlock(self.d_feature * self.d_time, num_heads, d_k, d_v, mlp_ratio=mlp_ratio, sigma=sigma) for _ in range(n_spatial_layer)
            ])
            self.spatial_cond_info = nn.Linear(hidden_size, self.d_feature * self.n_spatial, bias=False)
            # self.spatial_cond_info = nn.Linear(hidden_size, self.d_feature * self.n_spatial, bias=False) 

            # self.encode_noise_info = nn.Linear(self.d_feature * self.n_spatial + , self.d_feature * self.n_spatial, bias=False)

            # self.inverse_noise_info = nn.Linear(self.d_feature * self.n_spatial, hidden_size, bias=False)

        self.final_layer = FinalLayer(hidden_size, self.d_feature * self.n_spatial)
        self.ablation_config = ablation_config
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = PositionalEncoding(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.fde.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.fde.adaLN_modulation[-1].bias, 0)
        #     nn.init.constant_(block.gta.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.gta.adaLN_modulation[-1].bias, 0)
        
        if self.ablation_config['fde']:
            for block in self.fde_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        if self.ablation_config['gta']:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if self.ablation_config['spatial']:
            for block in self.spatial_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

 

    def forward(self, inputs, t, is_spat=False):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        X, masks, spatial_info = inputs['X'], inputs['missing_mask'], inputs['spatial_context']
        B, N, K, L = X[:,1,:,:,:].shape
        # print(f"B={B}, N={N}, K={K}, L={L}")
        masks = masks.reshape(B, 2, -1, L)
        # print(f"masks: {masks.shape}")
        masks[:,1,:,:] = masks[:,0,:,:] # B, N*K, L
        
        # B, N, L, K -> B=batch, N=spatial_locations, L=time, K=feature
        
        spatial_input = spatial_info
        # print(f"spatial input: {spatial_input.shape}")

        spatial_embed = F.leaky_relu(self.spatial_context_embeddimg(spatial_input)) # B, N, -1


        
        # cond = X[:,0,:,:,:].reshape(B, N*K, L)
        
        # cond = torch.permute(cond, (0, 2, 1))
        mask = torch.permute(masks[:,1,:,:], (0,2,1)) # B, L, N*K
        # print(f"mask: {mask.shape}")
        # mask = mask.reshape(B, L, N, K).permute(0, 2, 1, 3) # B, N, L, K
        # print(f"t before embed: {t.shape}")
        t = self.t_embedder(t).unsqueeze(dim=1)                   # (B, 1, D)
        # print(f"t after embed: {t.shape}")


        cond = X[:,0,:,:,:].reshape(B, N, -1) # (B, N, K*L)
        # print(f"cond: {cond.shape}, spat emb: {spatial_embed.shape}")
        cond_x = torch.cat([cond, spatial_embed], dim=-1)
        cond = F.leaky_relu(self.cond_spat_embed(cond_x)) # (B, N, K*L)
        cond = cond.reshape(B, N, K, L).reshape(B, N*K, L)
        cond = torch.permute(cond, (0, 2, 1)) # B, L, N*K
        # cond = cond.reshape(B, N, K, L).permute(0, 1, 3, 2) # B, N, L, K
        cond = torch.cat([cond, mask], dim=-1) # B, L, 2*N*K
        # print(f"cond: {cond.shape}")
        cond = F.relu(self.cond_pos_enc(self.cond_embedder(cond))) # B, L, D  
        # print(f"cond: {cond.shape}, t: {t.shape}")
        # cond = cond.permute(0,2,1,3).reshape(B, L, -1) # B, L, N*D
        c = t + cond # B, L, D
        # skips = torch.zeros_like(noise)
        
        noise = X[:,1,:,:,:].reshape(B, N, -1) # (B, N, K*L)
        # print(f"noise: {noise.shape}\nspatial embed: {spatial_embed.shape}")
        noise = torch.cat([noise, spatial_embed], dim=-1)  # (B, N, K*L)
        noise = F.relu(self.noise_spat_embed(noise))

        # c_spat = self.spatial_cond_info(c) # B, L, N * K
        # c_spat = c_spat.reshape(B, L, N, K).permute(0,2,3,1).reshape(B, N, -1) # B, N, K*L
        
        if self.ablation_config['spatial']:
            c_spat = self.spatial_cond_info(c) # B, L, N * K
            c_spat = c_spat.reshape(B, L, N, K).permute(0,2,3,1).reshape(B, N, -1) # B, N, K*L

            for i in range(len(self.spatial_blocks)):
                # print(f"noise: {noise.shape}, c_spat: {c_spat.shape}")
                noise, attn_spat = self.spatial_blocks[i](noise, c_spat) # B, N, K*L
            # noise_x = noise + cond_x
            # noise = self.spatial_block(noise_x, inputs['adj'])
        
        noise = noise.reshape(B, N, K, L).reshape(B, N*K, L) # B, N*K, L
        noise = torch.permute(noise, (0, 2, 1)) # B, L, N*K
        noise = torch.cat([noise, mask], dim=-1) # B, L, 2*N*K
        noise = F.relu(self.X_pos_enc(self.x_embedder(noise))) # B, L, D
        
        # noise = X[:,1,:,:,:] #.reshape(B, N, -1) # (B, N, K*L)
        # skips = torch.zeros_like(noise)
        if self.ablation_config['fde'] or self.ablation_config['gta']:
            for i in range(self.n_block_layers):

                # For features
                if self.ablation_config['fde']:
                    noise = torch.transpose(noise, 1, 2)
                    # t_emb = F.relu(self.t_embeds[i](t)) # B, 1, D
                    c = torch.transpose(c, 1, 2)
                    # c = cond.transpose(1,2) + t_emb.transpose(1,2)
                    noise, attn_feats = self.fde_blocks[i](noise, c)
                    # print(f"after fde: {noise}")
                    noise = torch.transpose(noise, 1, 2)
                    c = torch.transpose(c, 1, 2)
                # For time
                if self.ablation_config['gta']:
                    noise, attn_time = self.blocks[i](noise, c)                   # B, L, D

        x = self.final_layer(noise, c)
        # print(f"after final layer: {x}")              # (N, T, patch_size ** 2 * out_channels)
        x = torch.transpose(x, 1, 2)
        return x

    # def forward_with_cfg(self, x, t, y, cfg_scale):
    #     """
    #     Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
    #     """
    #     # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    #     half = x[: len(x) // 2]
    #     combined = torch.cat([half, half], dim=0)
    #     model_out = self.forward(combined, t, y)
    #     # For exact reproducibility reasons, we apply classifier-free guidance on only
    #     # three channels by default. The standard approach to cfg applies it to all channels.
    #     # This can be done by uncommenting the following line and commenting-out the line following that.
    #     # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
    #     eps, rest = model_out[:, :3], model_out[:, 3:]
    #     cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    #     half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    #     eps = torch.cat([half_eps, half_eps], dim=0)
    #     return torch.cat([eps, rest], dim=1)
    

############################### DiT CA 2 ###################################

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

def apply_rope_x(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class MLA(torch.nn.Module):
    def __init__(self, d_model, n_heads, max_len=1024, rope_theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # print(f"d_model: {d_model}, type: {type(d_model)}")
        self.dh = d_model // n_heads
        self.q_proj_dim = d_model // 2
        self.kv_proj_dim = (2*d_model) // 3

        self.qk_nope_dim = self.dh // 2
        self.qk_rope_dim = self.dh // 2
        
        ## Q projections
        # Lora
        self.W_dq = torch.nn.Parameter(0.01*torch.randn((d_model, self.q_proj_dim)))
        self.W_uq = torch.nn.Parameter(0.01*torch.randn((self.q_proj_dim, self.d_model)))
        self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)
        
        ## KV projections
        # Lora
        self.W_dkv = torch.nn.Parameter(0.01*torch.randn((d_model, self.kv_proj_dim + self.qk_rope_dim)))
        self.W_ukv = torch.nn.Parameter(0.01*torch.randn((self.kv_proj_dim,
                                                          self.d_model + (self.n_heads * self.qk_nope_dim))))
        self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)
        
        # output projection
        self.W_o = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

        # RoPE
        self.max_seq_len = max_len
        self.rope_theta = rope_theta

        # https://github.com/lucidrains/rotary-embedding-torch/tree/main
        # visualize emb later to make sure it looks ok
        # we do self.dh here instead of self.qk_rope_dim because its better
        freqs = 1.0 / (rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))
        emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]

        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        # This is like a parameter but its a constant so we can use register_buffer
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(self, x, choice, kv_cache=None, past_length=0):
        B, S, D = x.size()

        # Q Projections
        # print(f"X shape: {x.shape}")
        compressed_q = x @ self.W_dq
        # print(f"compressed q 1: {compressed_q.shape}")
        compressed_q = self.q_layernorm(compressed_q)
        # print(f"compressed q 2: {compressed_q.shape}")
        Q = compressed_q @ self.W_uq
        # print("Q shape 1:", Q.shape, self.dh)
        Q = Q.view(B, -1, self.n_heads, self.dh).transpose(1,2)
        # print("Q shape 2:", Q.shape)
        # print("Expected split sizes:", self.qk_nope_dim, self.qk_rope_dim)
        Q, Q_for_rope = torch.split(Q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        # Q Decoupled RoPE
        cos_q = self.cos_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        sin_q = self.sin_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)

        # KV Projections
        if kv_cache is None:
            compressed_kv = x @ self.W_dkv
            KV_for_lora, K_for_rope = torch.split(compressed_kv,
                                                  [self.kv_proj_dim, self.qk_rope_dim],
                                                  dim=-1)
            KV_for_lora = self.kv_layernorm(KV_for_lora)
        else:
            new_kv = x @ self.W_dkv
            compressed_kv = torch.cat([kv_cache, new_kv], dim=1)
            new_kv, new_K_for_rope = torch.split(new_kv,
                                                 [self.kv_proj_dim, self.qk_rope_dim],
                                                 dim=-1)
            old_kv, old_K_for_rope = torch.split(kv_cache,
                                                 [self.kv_proj_dim, self.qk_rope_dim],
                                                 dim=-1)
            new_kv = self.kv_layernorm(new_kv)
            old_kv = self.kv_layernorm(old_kv)
            KV_for_lora = torch.cat([old_kv, new_kv], dim=1)
            K_for_rope = torch.cat([old_K_for_rope, new_K_for_rope], dim=1)
            

        KV = KV_for_lora @ self.W_ukv
        KV = KV.view(B, -1, self.n_heads, self.dh+self.qk_nope_dim).transpose(1,2)
        K, V = torch.split(KV, [self.qk_nope_dim, self.dh], dim=-1)
        S_full = K.size(2)        

        # K Rope
        K_for_rope = K_for_rope.view(B, -1, 1, self.qk_rope_dim).transpose(1,2)
        cos_k = self.cos_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        sin_k = self.sin_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)

        # apply position encoding to each head
        K_for_rope = K_for_rope.repeat(1, self.n_heads, 1, 1)

        # split into multiple heads
        q_heads = torch.cat([Q, Q_for_rope], dim=-1)
        k_heads = torch.cat([K, K_for_rope], dim=-1)
        v_heads = V # already reshaped before the split

        # make attention mask
        if choice == 'time':
            mask = torch.ones((S,S_full), device=x.device)
            mask = torch.tril(mask, diagonal=past_length)
            mask = mask[None, None, :, :]

            sq_mask = mask == 1

        # attention
            x = torch.nn.functional.scaled_dot_product_attention(
                q_heads, k_heads, v_heads,
                attn_mask=sq_mask
            )
        else:
            x = torch.nn.functional.scaled_dot_product_attention(
                q_heads, k_heads, v_heads
            )

        x = x.transpose(1, 2).reshape(B, S, D)

        # apply projection
        x = x @ self.W_o.T

        return x, compressed_kv
    

class DynaSTI(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        config,
        d_time,
        d_feature,
        n_spatial,
        d_k=64,
        d_v=64,
        n_layer=4,
        n_spatial_layer=4,
        num_heads=8,
        mlp_ratio=4.0,
        learn_sigma=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.d_time = d_time
        self.d_feature = d_feature
        self.n_spatial = n_spatial
        self.num_heads = num_heads
        self.config = config

        self.t_embedder_2 = TimestepEmbedder(self.config['model']['feature_embed'] + self.config['model']['h_channels'])
        

        self.cond_embedder = nn.Linear(2 * self.d_time, self.d_time)
        
        self.spatial_context_embeddimg = nn.Linear(self.config['model']['spatial_context_dim'], self.config['model']['h_channels'])  # SpatialDescriptor(self.ablation_config['spatial_context_dim'], d_time * d_feature, self.ablation_config['h_channels'], d_feature)
        self.missing_spatial_context_embedding = nn.Linear(self.config['model']['spatial_context_dim'], self.config['model']['h_channels'])

        ################# Cond Mask ##################
        self.cond_mask_embed = nn.Linear(self.d_feature * 2, self.config['model']['feature_embed']) # self.d_feature * 2)
        ################# Noise Mask ##################
        self.noise_mask_embed = nn.Linear(self.d_feature * 2, self.config['model']['feature_embed']) # self.d_feature * 2)
        
        self.cond_pos_enc_2 = PositionalEncoding(self.config['model']['feature_embed'] + self.config['model']['h_channels'], n_position=d_time)
        self.cond_pos_enc_3 = PositionalEncoding(self.config['model']['feature_embed'] + self.config['model']['h_channels'], n_position=d_time)
        

        if config['ablation']['fe']:
            self.fde_blocks = nn.ModuleList([
                DiTBlock(d_time, num_heads, d_k=d_time, d_v=d_time, feature_size=self.config['model']['feature_embed'] + self.config['model']['h_channels'], mlp_ratio=mlp_ratio, choice='feature', kernel_size=3, dilation=(i+1), is_cross=False) for i in range(n_layer)
            ])
            self.n_block_layers = len(self.fde_blocks)

        if config['ablation']['te']:

            self.blocks = nn.ModuleList([
                DiTBlock(self.config['model']['feature_embed'] + self.config['model']['h_channels'], num_heads, d_k, d_v, mlp_ratio=mlp_ratio, is_cross=False, choice='time') for _ in range(n_layer)
            ])

            
            self.n_block_layers = len(self.blocks)

        if config['ablation']['spatial']:


            self.spatial_blocks = nn.ModuleList([
                DiTBlock(self.config['model']['feature_embed'] + self.config['model']['h_channels'], num_heads, d_k, d_v, mlp_ratio=mlp_ratio, is_cross=True) for _ in range(n_spatial_layer)
            ])

            

        if config['ablation']['te'] or config['ablation']['fe']:
            self.final_layer = FinalLayer(self.config['model']['feature_embed'] + self.config['model']['h_channels'], self.d_feature, is_cross=False)
        else:
            self.final_layer = FinalLayer(self.config['model']['feature_embed'] + self.config['model']['h_channels'], self.d_feature, is_cross=True)

        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder_2.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_2.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        
        if self.config['ablation']['fe']:
            for block in self.fde_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        if self.config['ablation']['te']:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if self.config['ablation']['spatial']:
            for block in self.spatial_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

 

    def forward(self, inputs, t, is_train=1):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        X_input, X_target, masks, spatial_info, missing_location, missing_data_mask = inputs['X_input'], inputs['X_target'], inputs['missing_mask'], inputs['spatial_context'], inputs['missing_loc'], inputs['missing_data_mask']
        A_q = inputs['A_q'] if 'A_q' in inputs.keys() else None
        A_h = inputs['A_h'] if 'A_h' in inputs.keys() else None
        B, N, K, L = X_input.shape
        if self.config['is_multi']:
            _, M, _, _ = X_target.shape
        
        spatial_input = spatial_info
        

        spatial_embed = F.leaky_relu(self.spatial_context_embeddimg(spatial_input)) # B, N, 128
        missing_location_embed = F.leaky_relu(self.missing_spatial_context_embedding(missing_location)) # B, 1, 128
        print(f"missing location embed: {missing_location_embed.shape}")
        t2 = self.t_embedder_2(t).unsqueeze(dim=1).unsqueeze(1)   # (B, 1, 1, K+128)
        t2 = t2.repeat(1, L, 1, 1) # B, L, 1, K+128
                 


        ################ If only use t2 ################
        cond_x = X_input.permute(0, 3, 1, 2) # B, L, N, K
        ################ Applying Mask #################
        cond_x = torch.cat([cond_x, masks.permute(0, 3, 1, 2)], dim=-1) # B, L, N, 2K
        cond_x = self.cond_mask_embed(cond_x) # B, L, N, 2K
        
        spatial_embed = spatial_embed.reshape((B, 1, N, -1)) # B, 1, N, 128
        repeated_spatial_embed = spatial_embed.repeat(1, L, 1, 1) # B, L, N, 128
        c1 = torch.cat([cond_x, repeated_spatial_embed], dim=-1) # B, L, N, K+128
        c1 = t2 + c1 # B, L, N, K+128
        c1 = c1.permute(0, 2, 1, 3).reshape((B*N, L, -1)) # B*N, L, K+128
        c1 = c1.reshape((B, N, L, -1)).permute(0, 2, 1, 3).reshape(B*L, N, -1) # B*L, N, K+128
        ################################################


        if self.config['is_multi']:
            noise = X_target.reshape(B, M, K, L) # (B, M, K, L)
                ############## Noise mask ################
            noise = noise.permute(0, 3, 1, 2) # B, L, M, K
            # print(f"noise: {noise.shape}\nmissing mask: {missing_data_mask.shape}")
            noise = torch.cat([noise, missing_data_mask.permute(0, 3, 1, 2)], dim=-1) # B, L, M, 2K
            noise = self.noise_mask_embed(noise) # B, L, M, 2K
            
            # noise = self.noise_feature_embed(noise) # B, L, D
            repeated_missing_location_embed = missing_location_embed.repeat(1, L, 1, 1) # B, L, M, 128
        
        else:
            noise = X_target.reshape(B, K, L) # (B, K, L)

            ############## Noise mask ################
            noise = noise.permute(0, 2, 1) # B, L, K
            # print(f"noise: {noise.shape}\nmissing mask: {missing_data_mask.shape}")
            noise = torch.cat([noise, missing_data_mask.squeeze(1).permute(0, 2, 1)], dim=-1) # B, L, 2K
            noise = self.noise_mask_embed(noise) # B, L, 2K
            
            # noise = self.noise_feature_embed(noise) # B, L, D
            repeated_missing_location_embed = missing_location_embed.repeat(1, L, 1) # B, L, 128
        print(f"noise: {noise.shape}, repeated loc: {repeated_missing_location_embed.shape}")
        noise = torch.cat([noise, repeated_missing_location_embed], dim=-1)  # (B, L, K + 128)
        
        

        if self.config['ablation']['spatial']:
            if self.config['is_multi']:
                noise = noise.reshape((B*L, M, -1)) # B*L, M, K+128
            else:
                noise = noise.reshape((B*L, 1, -1)) # B*L, 1, K+128
            for i in range(len(self.spatial_blocks)):
                # print(f"noise: {noise.shape}, c_spat: {c1.shape}")
                noise, attn_spat = self.spatial_blocks[i](noise, c1) # B*L, N=1, K+128
                # print(f"spatial noise: {noise.shape}")
        

        c1 = c1.reshape((B, L, N, -1)) # B, L, N, K+128
        if self.config['is_multi']:
            noise = noise.reshape((B, L, M, -1)) # B, L, M, K+128
        else:
            noise = noise.reshape((B, L, 1, -1)) # B, L, 1, K+128

        if (self.config['ablation']['te'] or self.config['ablation']['fe']):
            c3 = torch.cat([c1, noise], dim=-2) # B, L, N+1, K+128
            c2 = c3.clone()
            if self.config['is_multi']:
                c2[:, :, -M:, :] = 0.0 # B, L, N+M, K+128
            else:
                c2[:, :, -1, :] = 0.0 # B, L, N+1, K+128
            c2 = c2 + t2 # B, L, N+1, K+128
            if self.config['is_multi']:
                c2 = c2.permute(0, 2, 1, 3).reshape((B*(N+M), L, -1)) # B * (N+M), L, K+128
            else:
                c2 = c2.permute(0, 2, 1, 3).reshape((B*(N+1), L, -1)) # B * (N+1), L, K+128
            c2 = self.cond_pos_enc_2(c2) # B * (N+1), L, K+128
            c3[:, :, :N, :] = 0.0 # B, L, N+1, K+128
            if self.config['is_multi']:
                c3 = c3.permute(0, 2, 1, 3).reshape((B * (N+M), L, -1)) # B * N+M, L, K+128
            else:
                c3 = c3.permute(0, 2, 1, 3).reshape((B * (N+1), L, -1)) # B * N+1, L, K+128
            c3 = self.cond_pos_enc_3(c3) # B * N+1, L, K+128
            noise = c3
        
            for i in range(self.n_block_layers):
                # print(f"noise: {noise.shape}, c_spat: {c2.shape}")
                if self.config['ablation']['te']:
                    noise, attn_gta = self.blocks[i](noise, c2) # B * N+1, L, K+128

                if self.config['ablation']['fe']:
                    noise = noise.permute(0, 2, 1) # B * (N+1), K+128, L
                    # c2 = c2.permute(0, 2, 1) # B * (N+1), K+128, L

                    noise, attn_fde = self.fde_blocks[i](noise, c2.permute(0, 2, 1)) # B*(N+1), K+128, L

                    noise = noise.permute(0, 2, 1) # B * (N+1), L, K+128


            if self.config['is_multi']:
                noise = noise.reshape((B, N+M, L, -1)) # B, N+M, L, K+128
            else:
                noise = noise.reshape((B, N+1, L, -1)) # B, N+1, L, K+128


            c1 = c1.permute(0, 2, 1, 3) # B, N, L, K + 128
            if self.config['is_multi']:
                zeros = torch.zeros((B, M, L, c1.shape[-1])).cuda() # B, M, L, K + 128
            else:
                zeros = torch.zeros((B, 1, L, c1.shape[-1])).cuda() # B, 1, L, K + 128
            c1 = torch.cat([c1, zeros], dim=1) # B, N+1, L, K + 128
            x = self.final_layer(noise, c1) # B, N+1, L, K

            if self.config['is_multi']:
                x = x[:, -M:, :, :] # B, M, L, K
                x = x.permute(0, 2, 1, 3).reshape(B, L, -1)
            else:
                x = x[:, -1, :, :] # B, L, K
        else:
            noise = noise.reshape((B, L, -1)) # B, L, K+128
            x = self.final_layer(noise, c1) # B, N=1, L, K
           
        x = torch.transpose(x, 1, 2) # B, K, L

        if self.config['ablation']['spatial']:
            attn_spat = attn_spat.mean(dim=1).mean(dim=0)
        else:
            attn_spat = None
        return x, attn_spat
