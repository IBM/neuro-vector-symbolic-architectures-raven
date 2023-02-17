#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*


import sys; sys.path.insert(0,'../')
from collections import OrderedDict, namedtuple
from itertools import product
import numpy as np
import torch
import util.utils as utils
from util.utils import normalize
SceneProb = namedtuple("SceneProb", ["position_prob", "number_prob", "type_prob", "size_prob", "color_prob"])
SceneLogProb = namedtuple("SceneLogProb", ["position_logprob", "number_logprob", "type_logprob", "size_logprob", "color_logprob"])

class SceneEngine(object):
    def __init__(self, number_slots, device):
        self.device = device
        self.num_slots = number_slots
        self.positions = list(product(range(2), repeat=self.num_slots))
        # assume nonempty
        start_index = 1
        position2number = np.sum(self.positions[start_index:], axis=1)
        # note the correspondence of positions: first digit from the left corresponds to part one
        self.positions = torch.tensor(self.positions[start_index:], dtype=torch.int).to(self.device)
        self.dim_position = self.positions.shape[0]
        self.num_pos_index_map = OrderedDict()
        for i in range(start_index, self.num_slots + 1):
            self.num_pos_index_map[i] = torch.tensor(list(filter(lambda idx: position2number[idx] == i, range(len(position2number)))), dtype=torch.long).to(self.device)

    def compute_scene_prob(self, exist_logprob, type_logprob, size_logprob, color_logprob):
        # all in log prob
        # exist: tensor of shape (batch_size, 16, slots, DIM_EXIST)
        # type: tensor of shape (batch_size, 16, slots, DIM_TYPE)
        # size: tensor of shape (batch_size, 16, slots, DIM_SIZE)
        # color: tensor of shape (batch_size, 16, slots, DIM_COLOR)
        position_prob, position_logprob = self.compute_position_prob(exist_logprob)
        number_prob = self.compute_number_prob(position_prob)
        type_prob = self.compute_type_prob(type_logprob, position_logprob)
        size_prob = self.compute_size_prob(size_logprob, position_logprob)
        color_prob = self.compute_color_prob(color_logprob, position_logprob)
        return (SceneProb(position_prob, number_prob, type_prob, size_prob, color_prob),
                SceneLogProb(position_logprob, utils.log(number_prob), utils.log(type_prob), utils.log(size_prob), utils.log(color_prob)))

    def compute_position_prob(self, exist_logprob):
        batch_size = exist_logprob.shape[0]
        exist_logprob = exist_logprob.unsqueeze(2).expand(-1, -1, self.dim_position, -1, -1)
        index = self.positions.unsqueeze(0).unsqueeze(0).expand(batch_size, 16, -1, -1).unsqueeze(-1).type(torch.long)
        position_logprob = torch.gather(exist_logprob, -1, index) # (batch_size, 16, self.dim_position, slots, 1)
        position_logprob = torch.sum(position_logprob.squeeze(-1), dim=-1) # (batch_size, 16, self.dim_position)
        position_prob = torch.exp(position_logprob)
        # assume nonempty: all zero state is filtered out
        position_prob = normalize(position_prob)[0]
        position_logprob = utils.log(position_prob)
        return position_prob, position_logprob

    def compute_number_prob(self, position_prob):
        all_num_prob = []
        for _, indices in self.num_pos_index_map.items():
            num_prob = torch.sum(position_prob[:, :, indices], dim=-1, keepdim=True)
            all_num_prob.append(num_prob)
        number_prob = torch.cat(all_num_prob, dim=-1)
        return number_prob
        
    def compute_type_prob(self, type_logprob, position_logprob):
        batch_size = type_logprob.shape[0]
        index = self.positions.unsqueeze(0).unsqueeze(0).expand(batch_size, 16, -1, -1).unsqueeze(-1).type(torch.float)
        type_logprob = type_logprob.unsqueeze(2).expand(-1, -1, self.dim_position, -1, -1)
        type_logprob = index * type_logprob # (batch_size, 16, self.dim_position, slots, DIM_TYPE)
        type_logprob = torch.sum(type_logprob, dim=3) + position_logprob.unsqueeze(-1)
        type_prob = torch.exp(type_logprob)
        type_prob = torch.sum(type_prob, dim=2)
        inconsist_prob = 1.0 - torch.clamp(torch.sum(type_prob, dim=-1, keepdim=True), max=1.0)  # clamp for numerical stability   
        type_prob = torch.cat([type_prob, inconsist_prob], dim=-1)          
        return type_prob

    def compute_size_prob(self, size_logprob, position_logprob):
        batch_size = size_logprob.shape[0]
        index = self.positions.unsqueeze(0).unsqueeze(0).expand(batch_size, 16, -1, -1).unsqueeze(-1).type(torch.float)
        size_logprob = size_logprob.unsqueeze(2).expand(-1, -1, self.dim_position, -1, -1)
        size_logprob = index * size_logprob # (batch_size, 16, self.dim_position, slots, DIM_SIZE)
        size_logprob = torch.sum(size_logprob, dim=3) + position_logprob.unsqueeze(-1)
        size_prob = torch.exp(size_logprob)
        size_prob = torch.sum(size_prob, dim=2)
        inconsist_prob = 1.0 - torch.clamp(torch.sum(size_prob, dim=-1, keepdim=True), max=1.0) # clamp for numerical stability   
        size_prob = torch.cat([size_prob, inconsist_prob], dim=-1)          
        return size_prob

    def compute_color_prob(self, color_logprob, position_logprob):
        batch_size = color_logprob.shape[0]
        index = self.positions.unsqueeze(0).unsqueeze(0).expand(batch_size, 16, -1, -1).unsqueeze(-1).type(torch.float)
        color_logprob = color_logprob.unsqueeze(2).expand(-1, -1, self.dim_position, -1, -1)
        color_logprob = index * color_logprob # (batch_size, 16, self.dim_position, slots, DIM_COLOR)
        color_logprob = torch.sum(color_logprob, dim=3) + position_logprob.unsqueeze(-1)
        color_prob = torch.exp(color_logprob)
        color_prob = torch.sum(color_prob, dim=2)
        inconsist_prob = 1.0 - torch.clamp(torch.sum(color_prob, dim=-1, keepdim=True), max=1.0) # clamp for numerical stability   
        color_prob = torch.cat([color_prob, inconsist_prob], dim=-1)        
        return color_prob
