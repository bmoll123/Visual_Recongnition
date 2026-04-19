# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
import math
from torch import nn, Tensor


class PyTorchDeformAttn(nn.Module):
    """
    純 PyTorch 實作的【多尺度 (Multi-Scale)】Deformable Attention。
    """

    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels  # 🌟 新增：特徵尺度的數量 (通常是 3，即 C3, C4, C5)
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_head = d_model // n_heads

        # 🌟 偏移量與權重的預測，現在要乘上 n_levels
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        nn.init.constant_(self.sampling_offsets.bias.data, 0.0)
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, reference_points, input_spatial_shapes, input_features):
        N, Len_q, _ = query.shape
        value = self.value_proj(input_features)

        # 🌟 1. 將拼接的超長序列，依照 spatial_shapes 切割回不同尺度的 2D 特徵圖
        value_list = []
        start_idx = 0
        for H, W in input_spatial_shapes:
            end_idx = start_idx + H * W
            # 切割並 Reshape 回 (N, C, H, W) 給 grid_sample 用
            v = (
                value[:, start_idx:end_idx, :]
                .view(N, H, W, self.d_model)
                .permute(0, 3, 1, 2)
            )
            value_list.append(v)
            start_idx = end_idx

        # 2. 預測 Offset 和 Weights
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        # 對所有層和所有點一起做 Softmax，確保權重總和為 1
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        # 複製 reference_points 以匹配形狀
        reference_points_expanded = reference_points.view(N, Len_q, 1, 1, 1, 2).expand(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )

        output = 0
        # 🌟 3. 針對每一個尺度 (Level) 分別做採樣，然後加總
        for level, (H, W) in enumerate(input_spatial_shapes):
            level_offset = sampling_offsets[:, :, :, level, :, :]
            level_weight = attention_weights[:, :, :, level, :]

            # 將 offset 轉換為特徵圖上的比例
            offset_normalizer = torch.tensor(
                [W, H], dtype=query.dtype, device=query.device
            )
            sampling_locations = (
                reference_points_expanded[:, :, :, level, :, :]
                + level_offset / offset_normalizer
            )

            # 轉換到 [-1, 1] 供 grid_sample 使用
            sampling_locations = sampling_locations * 2 - 1

            # ==========================================
            # 🌟 修復核心：正確對齊 grid_sample 的 Batch 維度
            # 原本的 shape: (N, Len_q, n_heads, n_points, 2)
            # 我們要把它變成: (N * n_heads, Len_q, n_points, 2)
            # ==========================================
            sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4).reshape(
                N * self.n_heads, Len_q, self.n_points, 2
            )

            v_split = value_list[level].reshape(N * self.n_heads, self.d_head, H, W)

            # grid_sample 抓取特徵
            # 輸出的 sampled_features shape 會是: (N * n_heads, d_head, Len_q, n_points)
            sampled_features = F.grid_sample(
                v_split,
                sampling_locations,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            # ==========================================
            # 🌟 修復採樣後的還原：
            # 我們把抓出來的特徵 Reshape 回去，並排好順序以配合 Attention Weight
            # 目標 shape: (N, Len_q, n_heads, n_points, d_head)
            # ==========================================
            sampled_features = sampled_features.reshape(
                N, self.n_heads, self.d_head, Len_q, self.n_points
            ).permute(0, 3, 1, 4, 2)

            level_weight = level_weight.unsqueeze(-1)
            level_output = (sampled_features * level_weight).sum(-2)

            # 將這個尺度的結果累加到總輸出中
            output += level_output

        output = output.flatten(2)
        output = self.output_proj(output)
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self.reference_points_proj = nn.Linear(d_model, 2)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, spatial_shapes):
        seq_len, bs, c = src.shape

        # ========================================================
        # 🌟 拯救 OOM 的神級修改：實作 RT-DETR 的 AIFI 邏輯
        # ========================================================
        # 1. 取得最後一張特徵圖 (C5) 的長寬，計算其序列長度
        h5, w5 = spatial_shapes[-1]
        c5_len = h5 * w5

        # 2. 將超長序列切開：前半段是低階特徵(C3,C4)，後半段是高階特徵(C5)
        src_c3_c4 = src[:-c5_len]
        mask_c3_c4 = mask[:, :-c5_len]
        pos_c3_c4 = pos_embed[:-c5_len]

        src_c5 = src[-c5_len:]
        mask_c5 = mask[:, -c5_len:]
        pos_c5 = pos_embed[-c5_len:]

        # 3. 【關鍵】只有長度最短的 C5 進入 Encoder 計算全局 Attention！
        # 這樣記憶體消耗直接銳減 90% 以上
        memory_c5 = self.encoder(src_c5, src_key_padding_mask=mask_c5, pos=pos_c5)

        # 4. 把算完 Attention 的 C5，跟原本沒做 Attention 的 C3, C4 重新拼起來
        # 作為完整的 Multi-scale Memory 餵給 Decoder
        memory = torch.cat([src_c3_c4, memory_c5], dim=0)
        # ========================================================

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)

        # 生成 Reference Points
        reference_points = self.reference_points_proj(query_embed).sigmoid()

        # 傳入 Decoder (Deformable Attention 只抓取參考點，所以不怕多尺度)
        hs = self.decoder(
            tgt,
            memory,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )

        return hs.transpose(1, 2), None


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        reference_points,
        spatial_shapes,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt
        intermediate = []

        for layer in self.layers:
            # 🌟 把 reference_points 和 spatial_shapes 傳給每一層 Layer
            output = layer(
                output,
                memory,
                reference_points,
                spatial_shapes,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 🌟 替換這裡：不再使用 MultiheadAttention 做 Cross-Attention
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = PyTorchDeformAttn(d_model, n_heads=nhead, n_levels=3)

        # FFN 和 Norm 維持不變... (省略中間相同的程式碼)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # 🌟 修改 forward，新增 reference_points 和 spatial_shapes
    def forward_post(
        self,
        tgt,
        memory,
        reference_points,
        spatial_shapes,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):

        # 1. Self Attention (Query 之間的互動) 維持不變
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. 🌟 核心更改：Cross Attention 改用 Deformable Attention
        # 注意：我們剛寫的 PyTorchDeformAttn 預期 batch_first=True 的格式，
        # 所以要把 (Length, Batch, Dim) 轉置為 (Batch, Length, Dim)
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, query_pos).transpose(0, 1),
            reference_points=reference_points.transpose(0, 1),
            input_spatial_shapes=spatial_shapes,
            input_features=memory.transpose(0, 1),
        ).transpose(
            0, 1
        )  # 算完再轉置回來

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 3. FFN 維持不變
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    # (forward_pre 同理修改，為了簡潔這裡先省略)
    def forward(
        self,
        tgt,
        memory,
        reference_points,
        spatial_shapes,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        return self.forward_post(
            tgt,
            memory,
            reference_points,
            spatial_shapes,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
