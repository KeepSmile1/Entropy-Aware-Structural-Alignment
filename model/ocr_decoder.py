"""
transformer decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
import numpy as np
import time

from torch.autograd import Variable
from .ocr_encoder import ResNet, BasicBlock

from scipy.spatial.distance import cdist
torch.set_printoptions(precision=None, threshold=1000000, edgeitems=None, linewidth=None, profile=None)

class Bottleneck(nn.Module):

    def __init__(self, input_dim):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, 1)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(input_dim, input_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(input_dim)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class DSAL(nn.Module):
    # attention layer for encoder
    def __init__(self, channels):
        super(DSAL, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Foreground attention
        foreground = self.maxpool(x)
        foreground = self.conv(foreground)
        foreground = self.sigmoid(foreground)

        if foreground.size() != x.size():
            foreground = F.interpolate(foreground, size=x.size()[2:], mode='bilinear', align_corners=False)
        foreground = foreground * x

        # Background attention
        background = self.avgpool(foreground)
        background = self.conv(background)
        background = self.sigmoid(background)

        if background.size() != x.size():
            background = F.interpolate(background, size=x.size()[2:], mode='bilinear', align_corners=False)
        background = background * x

        return background

class ImagePatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        输入: (B, C, H, W)
        输出: (B, num_patches, embed_dim)  其中 num_patches = (H / patch_size) * (W / patch_size)
        """
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, (H/patch_size) * (W/patch_size), embed_dim)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Attention
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed Forward Network
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_layers, num_heads, ff_dim, dropout=0.1, output_shape=(1024, 8, 8)): #输出的shape需要自己定义
        super().__init__()
        self.patch_embedding = ImagePatchEmbedding(in_channels, patch_size, embed_dim)
        self.pos_embedding = PositionalEncoding(embed_dim)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.num_patches_h = 0
        self.num_patches_w = 0
        self.embed_dim = embed_dim
        self.output_shape = output_shape

    def forward(self, x):
        """
        输入: (B, C, H, W)
        输出: (B, D, H', W')  例如 (B, 1024, 8, 8)
        """
        x = self.patch_embedding(x)  # (B, num_patches, embed_dim)
        x = self.pos_embedding(x)

        for layer in self.transformer_layers:
            x = layer(x)

        # Feature Reshaping
        num_patches = x.size(1)
        embed_dim = x.size(2)
        output_channels = self.output_shape[0]
        output_height = self.output_shape[1]
        output_width = self.output_shape[2]

        linear_projection = nn.Linear(embed_dim, output_channels).to(x.device)
        x = linear_projection(x)
        x = x.transpose(1, 2).reshape(-1, output_channels, output_height, output_width)

        return x

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        rel_pos = self.pe(positions)
        return x + rel_pos.unsqueeze(0)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.1, max_len=7000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()    
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))       
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, align=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, align=align)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        if self.compress_attention:
            batch, head, s1, s2 = attention_map.shape
            attention_map = attention_map.permute(0, 2, 3, 1).contiguous()
            attention_map = self.compress_attention_linear(attention_map).permute(0, 3, 1, 2).contiguous()

        return self.linears[-1](x), attention_map


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None, align=None):

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    else:
        pass

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.d_model)
        return embed


class StructureDecoder(nn.Module):

    def __init__(self):
        super(StructureDecoder, self).__init__()

        self.mask_multihead = MultiHeadedAttention(h=4, d_model=1024, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=1024)

        self.multihead = MultiHeadedAttention(h=4, d_model=1024, dropout=0.1, compress_attention=False)
        self.mul_layernorm2 = LayerNorm(features=1024)

        self.pff = PositionwiseFeedForward(1024, 2048)
        self.mul_layernorm3 = LayerNorm(features=1024)

        self.gate_fusion = GateFusion(
            input_size=512,
            output_size=1024,
            num_features=4
        )
        self.last_print_time = time.time() 

        self.hc_proj = nn.Linear(512, 1024)
        self.cross_attn = MultiHeadedAttention(h=4, d_model=1024, dropout=0.1)
        self.cross_pff = PositionwiseFeedForward(1024, 2048)
        self.cross_layernorm3 = LayerNorm(features=1024)
        self.cross_layernorm4 = LayerNorm(features=1024)

    def forward(self, epoch, text, conv_feature, f_code, f_depth, f_parent, f_child, pos_entropy, test=None):
        text_max_length = text.shape[1]
        mask = subsequent_mask(text_max_length).cuda() 
        result = text
        result = self.mul_layernorm1(result + self.mask_multihead(result, result, result, mask=mask)[0])

        # print(f'text is {text}, result is {result}')
        b, c, h, w = conv_feature.shape
        conv_feature = conv_feature.view(b, c, h * w).permute(0, 2, 1).contiguous() # (b, 4, 1024)

        hc_proj = self.hc_proj(f_code)  # (3755, 1024)
        hc_proj = hc_proj.unsqueeze(0).expand(b, -1, -1)

        fused_char_features = self.gate_fusion([f_parent, f_child, f_depth, pos_entropy]) 
        fused_expanded = fused_char_features.unsqueeze(0).expand(b, -1, -1)  # (b,3755,1024)
        
        hc_proj = hc_proj + fused_expanded

        attn_output, cross_att_map = self.cross_attn(query=hc_proj, key=conv_feature, value=conv_feature, mask=None)
        
        final_output = attn_output

        scores = final_output.mean(dim=-1)
        scores = F.softmax(scores, dim=-1)

        topk_scores, topk_indices = torch.topk(scores, k=5, dim=-1)  
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, final_output.size(-1)) 
        topk_attn = torch.gather(final_output, 1, topk_indices_expanded) 
        topk_scores_expanded = topk_scores.unsqueeze(-1) 
        fused = (topk_attn * topk_scores_expanded).sum(dim=1, keepdim=True) 

        zero_vectors = torch.zeros(b, 1, 1024).cuda()
        most_similar_vectors_expanded = torch.cat([fused, zero_vectors], dim=1)
        result = result + most_similar_vectors_expanded

        word_image_align, attention_map = self.multihead(result, conv_feature, conv_feature, mask=None)
        result = self.mul_layernorm2(result + word_image_align)
        result = self.mul_layernorm3(result + self.pff(result))

        return result, attention_map, topk_indices


class GateFusion(nn.Module):
    def __init__(self, input_size, output_size, num_features):
        """
        :param input_size: 每个特征的维度，例如 512
        :param output_size: 融合后输出的维度，例如 1024
        :param num_features: 特征数量，例如 4
        """
        super(GateFusion, self).__init__()
        self.gate_fc = nn.Linear(num_features * input_size, num_features)
        self.proj_fc = nn.Linear(input_size, output_size)

    def forward(self, features):
        """
        :param features: 列表，每个元素形状为 [batch, input_size] 或单个样本时 [input_size]
        :return: 融合后的特征，形状为 [batch, output_size] 或 [output_size]
        """
        concatenated = torch.cat(features, dim=-1)  # shape: [batch, num_features * input_size]
        gates = torch.sigmoid(self.gate_fc(concatenated))  # shape: [batch, num_features]
        # 对每个特征按对应门控权重融合
        fused = sum(g.unsqueeze(-1) * feat for g, feat in zip(gates.unbind(dim=-1), features))
        output = self.proj_fc(fused)
        return output


class Transformer(nn.Module):
    def __init__(self, ddcm_model=None):
        super(Transformer, self).__init__()
        self.dim = 512
        self.wordlist_len = 3756
        self.embedding_word = Embeddings(self.dim, self.wordlist_len)  
        self.pe = PositionalEncoding(d_model=self.dim, dropout=0.1, max_len=7000)
        self.visual_encoder = ResNet(num_in=3, block=BasicBlock, layers=[3,4,6,3]).cuda() 

        # if TransformerEncoder is used
        # self.visual_encoder = TransformerEncoder(
        #     in_channels=3, 
        #     patch_size=4, 
        #     embed_dim=512, 
        #     num_layers=6, 
        #     num_heads=8, 
        #     ff_dim=2048,
        #     dropout=0.1,
        #     output_shape=(1024, 8, 8)
        # ).cuda()
        
        self.structure_decoder = StructureDecoder()
        self.output_head = nn.Linear(1024, 2048)

    def forward(self, epoch, image, text_length, text_input,
                f_code, f_depth, f_parent, f_child, pos_entropy,
                conv_feature=None, test=False, att_map=None):

        if conv_feature is None:
            conv_feature = self.visual_encoder(image)

        if text_length is None:
            return {
                'conv': conv_feature,
            }

        # print(f'text input is{text_input}')
        text_embedding = self.embedding_word(text_input)
        position_embedding = self.pe(torch.zeros(text_embedding.shape).cuda()).cuda()
        text_input_with_pe = torch.cat([text_embedding, position_embedding], 2)
        batch, seq_len, _ = text_input_with_pe.shape

        text_input_with_pe, attention_map, topk_indices = self.structure_decoder(
            epoch, text_input_with_pe, conv_feature,
            f_code, f_depth, f_parent, f_child, pos_entropy, test
        )
        output_result = self.output_head(text_input_with_pe)

        # print(f'output_result result:{output_result.shape}')
        if test:
            return {
                'pred': output_result,
                'map': attention_map,
                'conv': conv_feature,
                'topk': topk_indices
            }

        else:
            total_length = torch.sum(text_length).data
            probs_res = torch.zeros(total_length, 2048).type_as(output_result.data)

            start = 0
            for index, length in enumerate(text_length):
                length = length.data
                probs_res[start:start + length, :] = output_result[index, 0:0 + length, :]
                start = start + length

            return {
                'pred': probs_res,
                'map': attention_map,
                'conv': conv_feature,
                'topk': topk_indices
            }

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


if __name__ == '__main__':
    net = ResNet(num_in=3, block=BasicBlock, layers=[3, 4, 6, 3]).cuda()
    image = torch.Tensor(8, 3, 64, 64).cuda()
    result = net(image)
    print(result.shape)
    pass
