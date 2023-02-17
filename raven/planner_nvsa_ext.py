#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*

import sys; sys.path.insert(0,'../')
import numpy as np
import torch
from scipy.special import comb
import util.utils as utils
from util.utils import count_1, left_rotate, normalize, right_rotate

# NVSA import
from nvsa.reasoning.vsa_backend_raven import vsa_rule_detector_extended

class GeneralPlanner(object):
    def __init__(self, scene_dim, device, inconsistency_state, action_set=None,**kwargs):
        self.scene_dim = scene_dim
        self.inconsistency_state = inconsistency_state
        self.offset = 1 if self.inconsistency_state else 0
        self.action_set = action_set
        self.device = device
        
        self.vsa_rule_detector = vsa_rule_detector_extended(**kwargs)
        
        # constant
        self.valid_length_constant = self.scene_dim
        # progression one
        self.valid_length_progression_one = self.scene_dim - self.offset - 2
        # progression two
        self.valid_length_progression_two = self.scene_dim - self.offset - 4
        # progression mone
        self.valid_length_progression_mone = self.scene_dim - self.offset -2
        # progression mtwo
        self.valid_length_progression_mtwo = self.scene_dim - self.offset -4

        # arithmetic plus
        self.valid_length_arithmetic_plus = 0
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                k = (i + 1) + (j + 1) - 1
                if k <= self.scene_dim - self.offset - 1:
                    self.valid_length_arithmetic_plus += 1

        # arithmetic minus
        self.valid_length_arithmetic_minus = 0
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                k = (i + 1) - (j + 1) - 1
                if k >= 0:
                    self.valid_length_arithmetic_minus += 1
        
        # distribute three left
        self.valid_length_distribute_three = 0
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                for k in range(self.scene_dim - self.offset):
                    if i != j and j != k and i != k:
                        self.valid_length_distribute_three += 1
    
    
    def unit_action_prob(self, all_states, name,p_vsa_d=None, p_vsa_c=None,vsa=False):
        # all_states: prob of shape (batch, 16, scene_dim)
        if name.startswith("distribute_three_left") and vsa:
            prob,_ = self.vsa_rule_detector.distribute_three(p_vsa_d)  
            return prob, (self.valid_length_distribute_three)        
        if name.startswith("distribute_three_right") and vsa:
            prob,_ = self.vsa_rule_detector.distribute_three(p_vsa_d)  
            prob[:]=1e-15
            return prob, (self.valid_length_distribute_three)        
        elif name.startswith("constant") :
            prob,_ =  self.vsa_rule_detector.constant(p_vsa_d)
            return prob, (self.valid_length_constant**3)        
        elif name.startswith("arithmetic_plus"):
            prob,_ = self.vsa_rule_detector.arithmetic_plus(p_vsa_c,self.vsa_cb_cont_a) 
            return prob, (self.valid_length_arithmetic_plus ** 3)        
        elif name.startswith("arithmetic_minus"): 
            prob,_= self.vsa_rule_detector.arithmetic_minus(p_vsa_c,self.vsa_cb_cont_a)
            return prob, (self.valid_length_arithmetic_minus ** 3)        
        elif name.startswith("progression_one"): 
            prob,_ = self.vsa_rule_detector.progression_plus(p_vsa_c,self.vsa_cb_cont_1)
            return prob, (self.valid_length_progression_one ** 3)        
        elif name.startswith("progression_two"): 
            prob,_ = self.vsa_rule_detector.progression_plus(p_vsa_c,self.vsa_cb_cont_2)
            return prob, (self.valid_length_progression_two ** 3)        
        elif name.startswith("progression_mone"): 
            prob,_ = self.vsa_rule_detector.progression_minus(p_vsa_c,self.vsa_cb_cont_1)
            return prob, (self.valid_length_progression_mone ** 3)        
        elif name.startswith("progression_mtwo"): 
            prob,_ = self.vsa_rule_detector.progression_minus(p_vsa_c,self.vsa_cb_cont_2)
            return prob, (self.valid_length_progression_mtwo ** 3)        

    def action_prob(self, all_states,vsa=False):
        prob = []
        all_states_prob = torch.exp(all_states)
        p_vsa_d = self.vsa_rule_detector.pmf2vec(self.vsa_cb_discrete_c,all_states_prob[:,:8])
        p_vsa_c = self.vsa_rule_detector.pmf2vec(self.vsa_cb_cont_c[:self.scene_dim-self.offset],all_states_prob[:,:8,:self.scene_dim-self.offset])
        for action_name in self.action_set:
            # pdb.set_trace()
            uni_action_prob, norm_constant = self.unit_action_prob(all_states, action_name, p_vsa_d, p_vsa_c,vsa)
            prob.append(uni_action_prob.unsqueeze(-1) / norm_constant)
        all_prob = torch.cat(prob, dim=-1)
        # if all_prob.sum() == 0:
        #     print('abc')
        #     temp = torch.ones_like(all_prob) / all_prob.shape[1]
        #     temp = temp[0]
        #     return temp.reshape(1,temp.shape[0])
        return normalize(all_prob)[0]

class PositionPlanner(GeneralPlanner):
    def __init__(self, scene_dim, device, vsa_cb_discrete_a, vsa_cb_discrete_c, vsa_cb_cont_a, vsa_cb_cont_c, 
                action_set=["constant", "progression_one", "progression_mone", "arithmetic_plus", "arithmetic_minus", "distribute_three_left","distribute_three_right"], **kwargs):
        self.vsa_cb_discrete_a, self.vsa_cb_discrete_c = vsa_cb_discrete_a[:scene_dim], vsa_cb_discrete_c[:scene_dim]
        self.vsa_cb_cont_a, self.vsa_cb_cont_c = vsa_cb_cont_a[:scene_dim], vsa_cb_cont_c[:scene_dim] # no zero value for number

        self.vsa_rule_detector = vsa_rule_detector_extended(**kwargs)
        self.scene_dim = scene_dim
        self.inconsistency_state = False
        self.device = device
        self.offset = 1 if self.inconsistency_state else 0 
        self.action_set = action_set
        self.num_slots = int(np.log2(scene_dim + 1))
        # constant
        self.valid_length_constant = self.scene_dim
        # progression one
        self.progression_one_tri_valid = []
        self.progression_one_bi_valid = []
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 2:
                self.progression_one_tri_valid.append([k, right_rotate(k + 1, 1, self.num_slots) - 1, right_rotate(k + 1, 2, self.num_slots) - 1])
                self.progression_one_bi_valid.append([k, right_rotate(k + 1, 1, self.num_slots) - 1])
        self.progression_one_tri_valid = torch.tensor(self.progression_one_tri_valid).to(device)
        self.progression_one_bi_valid = torch.tensor(self.progression_one_bi_valid).to(device)
        # progression two
        self.progression_two_tri_valid = []
        self.progression_two_bi_valid = []
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 4:
                self.progression_two_tri_valid.append([k, right_rotate(k + 1, 2, self.num_slots) - 1, right_rotate(k + 1, 4, self.num_slots) - 1])
                self.progression_two_bi_valid.append([k, right_rotate(k + 1, 2, self.num_slots) - 1])
        self.progression_two_tri_valid = torch.tensor(self.progression_two_tri_valid).to(device)
        self.progression_two_bi_valid = torch.tensor(self.progression_two_bi_valid).to(device)
        # progression mone
        self.progression_mone_tri_valid = []
        self.progression_mone_bi_valid = []
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 2:
                self.progression_mone_tri_valid.append([k, left_rotate(k + 1, 1, self.num_slots) - 1, left_rotate(k + 1, 2, self.num_slots) - 1])
                self.progression_mone_bi_valid.append([k, left_rotate(k + 1, 1, self.num_slots) - 1])
        self.progression_mone_tri_valid = torch.tensor(self.progression_mone_tri_valid).to(device)
        self.progression_mone_bi_valid = torch.tensor(self.progression_mone_bi_valid).to(device)
        # progression mtwo
        self.progression_mtwo_tri_valid = []
        self.progression_mtwo_bi_valid = []
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 4:
                self.progression_mtwo_tri_valid.append([k, left_rotate(k + 1, 2, self.num_slots) - 1, left_rotate(k + 1, 4, self.num_slots) - 1])
                self.progression_mtwo_bi_valid.append([k, left_rotate(k + 1, 2, self.num_slots) - 1])
        self.progression_mtwo_tri_valid = torch.tensor(self.progression_mtwo_tri_valid).to(device)
        self.progression_mtwo_bi_valid = torch.tensor(self.progression_mtwo_bi_valid).to(device)
        # arithmetic plus
        self.arithmetic_plus_tri_valid = []
        self.arithmetic_plus_bi_valid = []
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                if i != j:
                    k = ((i + 1) | (j + 1)) - 1
                    self.arithmetic_plus_tri_valid.append([i, j, k])
                    self.arithmetic_plus_bi_valid.append([i, j])
        self.arithmetic_plus_tri_valid = torch.tensor(self.arithmetic_plus_tri_valid).to(device)
        self.arithmetic_plus_bi_valid = torch.tensor(self.arithmetic_plus_bi_valid).to(device)
        # arithmetic minus
        self.arithmetic_minus_tri_valid = []
        self.arithmetic_minus_bi_valid = []
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                if i != j:
                    k = ((i + 1) & (~(j + 1))) - 1
                    if k >= 0:
                        self.arithmetic_minus_tri_valid.append([i, j, k])
                        self.arithmetic_minus_bi_valid.append([i, j])
        self.arithmetic_minus_tri_valid = torch.tensor(self.arithmetic_minus_tri_valid).to(device)
        self.arithmetic_minus_bi_valid = torch.tensor(self.arithmetic_minus_bi_valid).to(device)
    
        # distribute three left and right
        self.valid_length_distribute_three = 0 
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                for k in range(self.scene_dim - self.offset):
                    if (i != j and j != k and i != k) and (count_1(i) == count_1(j) and count_1(j) == count_1(k) and count_1(i) == count_1(k)):
                        self.valid_length_distribute_three += 1



    def action_prob(self, all_states,no_distthree_pos=False,vsa=False):
        unnorm_prob, prob = [], []
        all_states_prob = torch.exp(all_states)
        p_vsa_d = self.vsa_rule_detector.pmf2vec(self.vsa_cb_discrete_c,all_states_prob[:,:8])
        for action_name in self.action_set:
            unit_action_prob, norm_constant = self.unit_action_prob_pos(all_states, action_name,p_vsa_d)
            unnorm_prob.append(unit_action_prob.unsqueeze(-1))
            prob.append(unit_action_prob.unsqueeze(-1) / norm_constant)
        all_prob = torch.cat(prob, dim=-1)

        # if all_prob.sum() == 0:
        #     print('abc')
        #     temp = torch.ones_like(all_prob) / all_prob.shape[1]
        #     temp = temp[0]

        #     unnorm_prob = []
        #     a = torch.ones(1)

        #     for action_name in self.action_set:
        #         unnorm_prob.append(a.unsqueeze(-1))

        #     return temp.reshape(1,temp.shape[0]), unnorm_prob
        return normalize(all_prob)[0], unnorm_prob
    
    def unit_action_prob_pos(self, all_states, name,p_vsa_d):
        batch_size = all_states.shape[0]
        if name.startswith("distribute_three_left") :
            prob,_ = self.vsa_rule_detector.distribute_three(p_vsa_d)  
            return prob, (self.valid_length_distribute_three)        
        if name.startswith("distribute_three_right") :
            prob,_ = self.vsa_rule_detector.distribute_three(p_vsa_d)  
            prob[:]=1e-15
            return prob, (self.valid_length_distribute_three)        
        if name.startswith("constant") :
            prob,_ = self.vsa_rule_detector.constant(p_vsa_d)  
            return prob, (self.valid_length_constant**3)        
        else:
            tri_valid_indices = getattr(self, name + "_tri_valid").unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1).type(torch.long)
            bi_valid_indices = getattr(self, name + "_bi_valid").unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1).type(torch.long)
            valid_length = tri_valid_indices.shape[1]
            # first row
            first_row = all_states[:, :3, :].unsqueeze(1).expand(-1, valid_length, -1, -1)
            first_row_logprob = utils.log(torch.sum(torch.exp(torch.sum(torch.gather(first_row, -1, tri_valid_indices).squeeze(-1), dim=-1)), dim=-1))
            # second row
            second_row = all_states[:, 3:6, :].unsqueeze(1).expand(-1, valid_length, -1, -1)
            second_row_logprob = utils.log(torch.sum(torch.exp(torch.sum(torch.gather(second_row, -1, tri_valid_indices).squeeze(-1), dim=-1)), dim=-1))
            # third row
            third_row = all_states[:, 6:8, :].unsqueeze(1).expand(-1, valid_length, -1, -1)
            third_row_logprob = utils.log(torch.sum(torch.exp(torch.sum(torch.gather(third_row, -1, bi_valid_indices).squeeze(-1), dim=-1)), dim=-1))
            return torch.exp((first_row_logprob + second_row_logprob + third_row_logprob)), (valid_length ** 3)

class NumberPlanner(GeneralPlanner):
    
    def __init__(self, scene_dim, device, vsa_cb_discrete_a, vsa_cb_discrete_c, vsa_cb_cont_a, vsa_cb_cont_c, 
                action_set=["progression_one", "progression_mone", "arithmetic_plus", "arithmetic_minus", "distribute_three_left","distribute_three_right"], **kwargs):
        super(NumberPlanner, self).__init__(scene_dim, device, False, action_set, **kwargs)
        self.normalization_num = [comb(self.scene_dim, i + 1) for i in range(self.scene_dim)]
        self.normalization_num = torch.tensor(self.normalization_num).to(device)
        self.vsa_cb_discrete_a, self.vsa_cb_discrete_c = vsa_cb_discrete_a[:scene_dim], vsa_cb_discrete_c[:scene_dim]
        self.vsa_cb_cont_a, self.vsa_cb_cont_c = vsa_cb_cont_a[1:scene_dim+1], vsa_cb_cont_c[1:scene_dim+1] # no zero value for number
        self.vsa_cb_cont_1 = vsa_cb_cont_a[1]
        self.vsa_cb_cont_2 = vsa_cb_cont_a[2]

    def action_prob(self, all_states,vsa):
        unnorm_prob, prob = [], []
        all_states_prob = torch.exp(all_states)
        p_vsa_d = self.vsa_rule_detector.pmf2vec(self.vsa_cb_discrete_c,all_states_prob[:,:8])
        p_vsa_c = self.vsa_rule_detector.pmf2vec(self.vsa_cb_cont_c, all_states_prob[:,:8])
        for action_name in self.action_set:
            unit_action_prob, norm_constant = self.unit_action_prob(all_states, action_name,p_vsa_d,p_vsa_c,vsa)
            unnorm_prob.append(unit_action_prob.unsqueeze(-1))
            prob.append(unit_action_prob.unsqueeze(-1) / norm_constant)
        all_prob = torch.cat(prob, dim=-1)
        
        # if all_prob.sum() == 0:
        #     #pdb.set_trace()
        #     print('abc')
        #     #pdb.set_trace()
        #     #print(all_prob.shape)
        #     temp = torch.ones_like(all_prob) / all_prob.shape[1]
        #     temp = temp[0]
        #     #pdb.set_trace()
        #     #print((temp.reshape(1,temp.shape[0])).shape)
        #     unnorm_prob = []
        #     a = torch.ones(1)

        #     for action_name in self.action_set:
        #         unnorm_prob.append(a.unsqueeze(-1))



        #     return temp.reshape(1,temp.shape[0]),unnorm_prob
        return normalize(all_prob)[0], unnorm_prob

class TypePlanner(GeneralPlanner):

    def __init__(self, scene_dim, device, vsa_cb_discrete_a, vsa_cb_discrete_c, vsa_cb_cont_a, vsa_cb_cont_c, 
                action_set=["constant", "progression_one", "progression_mone", "progression_two", "progression_mtwo", "distribute_three_left","distribute_three_right"],**kwargs):
        super(TypePlanner, self).__init__(scene_dim, device, True, action_set,**kwargs)
        self.vsa_cb_discrete_a, self.vsa_cb_discrete_c = vsa_cb_discrete_a[:scene_dim], vsa_cb_discrete_c[:scene_dim]
        self.vsa_cb_cont_a, self.vsa_cb_cont_c = vsa_cb_cont_a[1:scene_dim+1], vsa_cb_cont_c[1:scene_dim+1] # no zero value for type
        self.vsa_cb_cont_1, self.vsa_cb_cont_2 = vsa_cb_cont_a[1],  vsa_cb_cont_a[2]

class SizePlanner(GeneralPlanner):

    def __init__(self, scene_dim, device, vsa_cb_discrete_a, vsa_cb_discrete_c, vsa_cb_cont_a, vsa_cb_cont_c, 
                action_set=["constant", "progression_one", "progression_mone", "progression_two", "progression_mtwo", "arithmetic_plus", "arithmetic_minus", "distribute_three_left","distribute_three_right"], **kwargs):
        super(SizePlanner, self).__init__(scene_dim, device, True, action_set, **kwargs)
        self.vsa_cb_discrete_a, self.vsa_cb_discrete_c = vsa_cb_discrete_a[:scene_dim], vsa_cb_discrete_c[:scene_dim]
        self.vsa_cb_cont_a, self.vsa_cb_cont_c = vsa_cb_cont_a[1:scene_dim+1], vsa_cb_cont_c[1:scene_dim+1] # no zero value for size
        self.vsa_cb_cont_1, self.vsa_cb_cont_2 = vsa_cb_cont_a[1],  vsa_cb_cont_a[2]


class ColorPlanner(GeneralPlanner):

    def __init__(self, scene_dim, device, vsa_cb_discrete_a, vsa_cb_discrete_c, vsa_cb_cont_a, vsa_cb_cont_c, 
                action_set=["constant", "progression_one", "progression_mone", "progression_two", "progression_mtwo", "arithmetic_plus", "arithmetic_minus", "distribute_three_left","distribute_three_right"], **kwargs):
        super(ColorPlanner, self).__init__(scene_dim, device, True, action_set,**kwargs)  
        self.vsa_cb_discrete_a, self.vsa_cb_discrete_c = vsa_cb_discrete_a[:scene_dim], vsa_cb_discrete_c[:scene_dim]
        self.vsa_cb_cont_a, self.vsa_cb_cont_c = vsa_cb_cont_a[:scene_dim], vsa_cb_cont_c[:scene_dim] # no zero value for size
        self.vsa_cb_cont_1, self.vsa_cb_cont_2 = vsa_cb_cont_a[1],  vsa_cb_cont_a[2]