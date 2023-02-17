#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*

import numpy as np
import torch
import util.utils as utils

def calculate_acc(output, target):
    pred = output.data.max(-1)[1]
    correct = pred.eq(target.data).cpu().sum().numpy()
    return correct * 100.0 / np.prod(target.shape)

def calculate_correct(output, target):
    pred = output.data.max(-1)[1]
    correct = pred.eq(target.data).cpu().sum().numpy()
    return correct

def JSD(p, q):
    part1 = torch.sum(p * utils.log(2.0 * p) - p * utils.log(p + q), dim=-1)
    part2 = torch.sum(q * utils.log(2.0 * q) - q * utils.log(q + p), dim=-1)
    return 0.5 * part1 + 0.5 * part2

def aux_loss_single(all_action_prob, all_action_rule):
    '''
    Compute the auxiliary loss for single subconstellations

    Parameters
    ----------
    all_action_prob     list of tensors
        The action probabilities of all attributes
    all_action_rule     list of tensors
        Ground-truth rules 

    Return
    ------
    loss                tensor 
        Main loss function (can be backproped)
    '''
    pos_num_action_prob, type_action_prob, size_action_prob, color_action_prob = all_action_prob
    pos_num_rule, type_rule, size_rule, color_rule = all_action_rule
    
    # Apply loss for position if needed (2x2, 3x3, inner grid of O-IG) 
    if pos_num_action_prob is not None : 
        loss_pos_num = -utils.log(torch.gather(pos_num_action_prob, -1, pos_num_rule.unsqueeze(-1))).mean()
    else:
        loss_pos_num = 0.

    loss_type = -utils.log(torch.gather(type_action_prob, -1, type_rule.unsqueeze(-1))).mean()
    loss_size = -utils.log(torch.gather(size_action_prob, -1, size_rule.unsqueeze(-1))).mean()

    # Compute color loss for all exept outer grid of O-IG
    if color_action_prob is not None: 
        loss_color = -utils.log(torch.gather(color_action_prob, -1, color_rule.unsqueeze(-1))).mean()
    else: 
        loss_color = 0.

    loss = (loss_pos_num + loss_type + loss_size + loss_color)
    return loss


def aux_loss_multi(all_action_prob, all_action_rule):
    '''
    Compute the auxiliary loss for one or multiple subconstellations

    Parameters
    ----------
    all_action_prob     list of tensors
        The action probabilities of all constellations and attributes
    all_action_rule     list of tensors
        Ground-truth rules 

    Return
    ------
    loss                tensor 
        Main loss function (can be backproped)
    mean_loss           float
        Mean loss function (no backprop)
    '''
    # Constellation with one grid (C,3x3, 2x2)
    if len(all_action_prob)==4:
        loss =  aux_loss_single(all_action_prob,all_action_rule[0])
    # Cases with two constellations (LR,UD,O-IC, O-IG)
    else:  
        loss = aux_loss_single(all_action_prob[0],all_action_rule[0]) + aux_loss_single(all_action_prob[1],all_action_rule[1])

    return loss, loss.mean().item() 