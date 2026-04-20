# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class CCFM(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse4 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.down4 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.fuse4_bu = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.down5 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.fuse5_bu = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, c3, c4, c5):
        p5 = c5
        p5_up = F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p4 = self.fuse4(torch.cat([c4, p5_up], dim=1))
        p4_up = F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.fuse3(torch.cat([c3, p4_up], dim=1))
        out3 = p3
        out4 = self.fuse4_bu(torch.cat([p4, self.down4(out3)], dim=1))
        out5 = self.fuse5_bu(torch.cat([p5, self.down5(out4)], dim=1))
        return out3, out4, out5


class PyTorchDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        self.d_model, self.n_levels, self.n_heads, self.n_points = (
            d_model,
            n_levels,
            n_heads,
            n_points,
        )
        self.d_head = d_model // n_heads
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
        # 🌟 核心修正：將輸入從 (Seq, Batch, C) 轉為 (Batch, Seq, C)
        query = query.transpose(0, 1)
        input_features = input_features.transpose(0, 1)

        N, Len_q, _ = query.shape  # N 現在正確對應到 Batch Size
        value = self.value_proj(input_features)
        value_list = []
        start_idx = 0
        for H, W in input_spatial_shapes:
            end_idx = start_idx + H * W
            v = (
                value[:, start_idx:end_idx, :]
                .reshape(N, H, W, self.d_model)
                .permute(0, 3, 1, 2)
            )
            value_list.append(v)
            start_idx = end_idx

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = F.softmax(
            self.attention_weights(query).view(N, Len_q, self.n_heads, -1), -1
        ).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        # 這裡 reference_points 如果是 (Batch, Seq, 2)，直接展開
        reference_points_expanded = reference_points.view(N, Len_q, 1, 1, 1, 2).expand(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )

        output = 0
        for level, (H, W) in enumerate(input_spatial_shapes):
            level_offset = sampling_offsets[:, :, :, level, :, :]
            level_weight = attention_weights[:, :, :, level, :]
            offset_normalizer = torch.tensor(
                [W, H], dtype=query.dtype, device=query.device
            )
            sampling_locations = (
                reference_points_expanded[:, :, :, level, :, :]
                + level_offset / offset_normalizer
            ) * 2 - 1
            sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4).reshape(
                N * self.n_heads, Len_q, self.n_points, 2
            )
            v_split = value_list[level].reshape(N * self.n_heads, self.d_head, H, W)
            sampled_features = F.grid_sample(
                v_split,
                sampling_locations,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled_features = sampled_features.reshape(
                N, self.n_heads, self.d_head, Len_q, self.n_points
            ).permute(0, 3, 1, 4, 2)
            output += (sampled_features * level_weight.unsqueeze(-1)).sum(-2)

        # 🌟 運算結束後轉回 (Seq, Batch, C)
        return self.output_proj(output.flatten(2)).transpose(0, 1)


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
        self.d_model, self.nhead = d_model, nhead
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            nn.LayerNorm(d_model) if normalize_before else None,
        )
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            nn.LayerNorm(d_model),
            return_intermediate=return_intermediate_dec,
        )
        self.reference_points_proj = nn.Linear(d_model, 2)
        self.ccfm = CCFM(hidden_dim=d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, spatial_shapes):
        seq_len, bs, c = src.shape
        h3, w3 = spatial_shapes[0]
        h4, w4 = spatial_shapes[1]
        h5, w5 = spatial_shapes[2]
        l3, l4, l5 = h3 * w3, h4 * w4, h5 * w5

        src_c3 = src[:l3].permute(1, 2, 0).reshape(bs, c, h3, w3)
        src_c4 = src[l3 : l3 + l4].permute(1, 2, 0).reshape(bs, c, h4, w4)
        src_c5 = src[-l5:]
        mask_c5 = mask[:, -l5:]
        pos_c5 = pos_embed[-1] if isinstance(pos_embed, list) else pos_embed[-l5:]

        memory_c5 = self.encoder(src_c5, src_key_padding_mask=mask_c5, pos=pos_c5)
        memory_c5_2d = memory_c5.permute(1, 2, 0).reshape(bs, c, h5, w5)
        out3, out4, out5 = self.ccfm(src_c3, src_c4, memory_c5_2d)

        memory = torch.cat(
            [
                out3.flatten(2).permute(2, 0, 1),
                out4.flatten(2).permute(2, 0, 1),
                out5.flatten(2).permute(2, 0, 1),
            ],
            dim=0,
        )

        if hasattr(self, "get_query_selection"):
            selected_query = self.get_query_selection(memory.transpose(0, 1))
            tgt_query = selected_query.transpose(0, 1)
        else:
            tgt_query = query_embed.unsqueeze(1).repeat(1, bs, 1)

        reference_points = self.reference_points_proj(
            tgt_query.transpose(0, 1)
        ).sigmoid()
        tgt = torch.zeros_like(tgt_query)

        hs = self.decoder(
            tgt,
            memory,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=tgt_query,
        )
        return hs.transpose(1, 2), None


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
        return self.norm(output) if self.norm is not None else output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.norm, self.return_intermediate = norm, return_intermediate

    def forward(
        self,
        tgt,
        memory,
        reference_points,
        spatial_shapes,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        output, intermediate = tgt, []
        for layer in self.layers:
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
        return (
            torch.stack(intermediate)
            if self.return_intermediate
            else output.unsqueeze(0)
        )


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
        self.linear1, self.linear2 = nn.Linear(d_model, dim_feedforward), nn.Linear(
            dim_feedforward, d_model
        )
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout, self.dropout1, self.dropout2 = (
            nn.Dropout(dropout),
            nn.Dropout(dropout),
            nn.Dropout(dropout),
        )
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


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
        self.cross_attn = PyTorchDeformAttn(d_model, n_heads=nhead, n_levels=3)
        self.linear1, self.linear2 = nn.Linear(d_model, dim_feedforward), nn.Linear(
            dim_feedforward, d_model
        )
        self.norm1, self.norm2, self.norm3 = (
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
        )
        self.dropout1, self.dropout2, self.dropout3 = (
            nn.Dropout(dropout),
            nn.Dropout(dropout),
            nn.Dropout(dropout),
        )
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        tgt,
        memory,
        reference_points,
        spatial_shapes,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        # 🌟 關鍵：這裡 DeformAttn 內部的 transpose(0,1) 會處理好維度
        tgt2 = self.cross_attn(
            query=tgt if query_pos is None else tgt + query_pos,
            reference_points=reference_points,
            input_spatial_shapes=spatial_shapes,
            input_features=memory,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(src2))


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
