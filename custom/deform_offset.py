# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
@Date    ：2021/7/12 9:40 
'''
import torch
from torch.nn import functional as F

def assign_coords(coords):
    
    # 归一化处理
    coords = (coords - coords.mean(dim=(0, 1, 2, 3))) / coords.std(dim=(0, 1, 2, 3))
    coords /= coords.abs().max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

    return coords

def batch_map_coordinates(input, coordinates):

    feats = torch.split(input, split_size_or_sections=1, dim=1)

    bs, channels, h, w = input.size()

    coordinates_h = torch.clip(coordinates[..., 0], 0., h - 1.)
    coordinates_w = torch.clip(coordinates[..., 1], 0., w - 1.)

    coordinates = torch.stack([coordinates_w, coordinates_h], dim=-1)
    
    # 形成左上、右下、右上、左下坐标
    coords_lt = torch.floor(coordinates)
    coords_rb = torch.ceil(coordinates)
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], dim=-1)
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], dim=-1)

    coords_lt = assign_coords(coords_lt)
    coords_rb = assign_coords(coords_rb)
    coords_lb = assign_coords(coords_lb)
    coords_rt = assign_coords(coords_rt)

    def get_vals_by_coords(coords):

        coords = [coord.squeeze(dim=1) for coord in
                  torch.split(coords, split_size_or_sections=1, dim=1)]

        vals = [F.grid_sample(feat, coord, align_corners=True) for feat, coord in zip(feats, coords)]
        vals = torch.cat(vals, dim=1)

        return vals

    vals_lt = get_vals_by_coords(coords_lt)
    vals_rb = get_vals_by_coords(coords_rb)
    vals_lb = get_vals_by_coords(coords_lb)
    vals_rt = get_vals_by_coords(coords_rt)
    
    # 双线性插值
    coors_offsets_lt = coordinates - torch.floor(coordinates)
    vals_t = vals_lt + (vals_rt - vals_lt) * coors_offsets_lt[..., 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coors_offsets_lt[..., 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coors_offsets_lt[..., 1]

    return mapped_vals


def batch_map_offsets(input, offsets):

    h, w = input.size()[2:]
    grids = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    grids = torch.stack(grids, axis=-1)
    grids = grids.view(1, 1, *grids.size())

    coordinates = grids + offsets
    mapped_vals = batch_map_coordinates(input, coordinates)

    return mapped_vals
