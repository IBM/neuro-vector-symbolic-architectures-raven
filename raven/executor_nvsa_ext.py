#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*

import sys; sys.path.insert(0,'../')
import numpy as np
import torch
from util.utils import count_1, left_rotate, normalize, right_rotate
# NVSA import
from nvsa.reasoning.vsa_backend_raven import vsa_rule_executor_extended


class GeneralExecutor(object):
    def __init__(self, scene_dim, device, inconsistency_state, action_set=None,**kwargs):
        self.scene_dim = scene_dim
        self.scene_dim_2 = self.scene_dim ** 2
        self.inconsistency_state = inconsistency_state
        self.device = device
        self.action_set = action_set 

        self.vsa_rule_executor = vsa_rule_executor_extended(**kwargs)
        self.offset = 1 if self.inconsistency_state else 0

    def apply(self, all_states, action):
        
        bs = all_states.shape[0]
        result = all_states[:,0,:].clone()
        for b in range(bs): 
            if "distribute_three" in self.action_set[action[b]]:
                p_vsa_d = self.vsa_rule_executor.pmf2vec(self.vsa_cb_discrete_c,all_states[b,:8])
                result[b] = self.vsa_rule_executor.distribute_three(self.vsa_cb_discrete_a, p_vsa_d)
            elif "constant" in self.action_set[action[b]]: 
                pred_const= all_states[b,6,:] + all_states[b,7,:] 
                result[b] = normalize(pred_const)[0]
            elif "arithmetic_plus" in self.action_set[action[b]]:
                p_vsa_c = self.vsa_rule_executor.pmf2vec(self.vsa_cb_cont_c,all_states[b,:8])
                result[b] = self.vsa_rule_executor.arithmetic_plus(self.vsa_cb_cont_a, p_vsa_c)
            elif "arithmetic_minus" in self.action_set[action[b]]:
                p_vsa_c = self.vsa_rule_executor.pmf2vec(self.vsa_cb_cont_c,all_states[b,:8])
                result[b] = self.vsa_rule_executor.arithmetic_minus(self.vsa_cb_cont_a,p_vsa_c)
            elif "progression_one" in self.action_set[action[b]]: 
                p_vsa_c = self.vsa_rule_executor.pmf2vec(self.vsa_cb_cont_c,all_states[b,:8])
                result[b] = self.vsa_rule_executor.progression_plus(self.vsa_cb_cont_a, p_vsa_c,self.vsa_cb_cont_1)
            elif "progression_two" in self.action_set[action[b]]: 
                p_vsa_c = self.vsa_rule_executor.pmf2vec(self.vsa_cb_cont_c,all_states[b,:8])
                result[b] = self.vsa_rule_executor.progression_plus(self.vsa_cb_cont_a, p_vsa_c,self.vsa_cb_cont_2)
            elif "progression_mone" in self.action_set[action[b]]: 
                p_vsa_c = self.vsa_rule_executor.pmf2vec(self.vsa_cb_cont_c,all_states[b,:8])
                result[b] = self.vsa_rule_executor.progression_minus(self.vsa_cb_cont_a, p_vsa_c,self.vsa_cb_cont_1)
            elif "progression_mtwo" in self.action_set[action[b]]: 
                p_vsa_c = self.vsa_rule_executor.pmf2vec(self.vsa_cb_cont_c,all_states[b,:8])
                result[b] = self.vsa_rule_executor.progression_minus(self.vsa_cb_cont_a, p_vsa_c,self.vsa_cb_cont_2)
        return result

    
class PositionExecutor(GeneralExecutor):
    def __init__(self, scene_dim, device, vsa_cb_discrete_a, vsa_cb_discrete_c, vsa_cb_cont_a, vsa_cb_cont_c, 
                action_set=["constant", "progression_one", "progression_mone", "arithmetic_plus", "arithmetic_minus", "distribute_three_left","distribute_three_right"], **kwargs):
        self.vsa_cb_discrete_a, self.vsa_cb_discrete_c = vsa_cb_discrete_a[:scene_dim], vsa_cb_discrete_c[:scene_dim]
        self.vsa_cb_cont_a, self.vsa_cb_cont_c = vsa_cb_cont_a[:scene_dim], vsa_cb_cont_c[:scene_dim] # no zero value for number
        self.vsa_rule_executor = vsa_rule_executor_extended(**kwargs)
        self.scene_dim = scene_dim
        self.scene_dim_2 = self.scene_dim ** 2
        self.inconsistency_state = False
        self.device = device
        self.offset = 1 if self.inconsistency_state else 0 
        self.action_set = action_set
      
        self.constant_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim):
            self.constant_trans[:, k, k] = 1.0
        self.constant_trans = self.constant_trans.view(self.scene_dim_2, self.scene_dim).to(device)
        self.num_slots = int(np.log2(scene_dim + 1))

        self.progression_one_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 2:
                self.progression_one_trans[:, right_rotate(k + 1, 1, self.num_slots) - 1, right_rotate(k + 1, 2, self.num_slots) - 1] = 1.0
        self.progression_one_trans = self.progression_one_trans.view(self.scene_dim_2, self.scene_dim).to(device)
        
        self.progression_two_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 4:
                self.progression_two_trans[:, right_rotate(k + 1, 2, self.num_slots) - 1, right_rotate(k + 1, 4, self.num_slots) - 1] = 1.0
        self.progression_two_trans = self.progression_two_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.progression_mone_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 2:
                self.progression_mone_trans[:, left_rotate(k + 1, 1, self.num_slots) - 1, left_rotate(k + 1, 2, self.num_slots) - 1] = 1.0
        self.progression_mone_trans = self.progression_mone_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.progression_mtwo_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset):
            if count_1(k + 1) <= self.num_slots - 4:
                self.progression_mtwo_trans[:, left_rotate(k + 1, 2, self.num_slots) - 1, left_rotate(k + 1, 4, self.num_slots) - 1] = 1.0
        self.progression_mtwo_trans = self.progression_mtwo_trans.view(self.scene_dim_2, self.scene_dim).to(device)
        
        self.arithmetic_plus_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                if i != j:
                    k = ((i + 1) | (j + 1)) - 1
                    self.arithmetic_plus_trans[i, j, k] = 1.0
        self.arithmetic_plus_trans = self.arithmetic_plus_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.arithmetic_minus_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for i in range(self.scene_dim - self.offset):
            for j in range(self.scene_dim - self.offset):
                k = ((i + 1) & (~(j + 1))) - 1
                if k >= 0:
                    self.arithmetic_minus_trans[i, j, k] = 1.0
        self.arithmetic_minus_trans = self.arithmetic_minus_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        # distribute three left
        self.distribute_three_left_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset):
            self.distribute_three_left_trans[:, k, k] = 1.0
        self.distribute_three_left_trans = self.distribute_three_left_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        # distribute three right
        self.distribute_three_right_trans = torch.zeros((self.scene_dim, self.scene_dim, self.scene_dim))
        for k in range(self.scene_dim - self.offset):
            self.distribute_three_right_trans[k, :, k] = 1.0
        self.distribute_three_right_trans = self.distribute_three_right_trans.view(self.scene_dim_2, self.scene_dim).to(device)

        self.gather_trans()

    def gather_trans(self):
        self.trans = torch.cat([getattr(self, trans_matrix_name + "_trans").unsqueeze(0) for trans_matrix_name in self.action_set], dim=0)

    def apply_position(self, all_states, action): 
        bs = all_states.shape[0]
        result = all_states[:,0,:].clone()
        for b in range(bs):
            if "distribute_three" in self.action_set[action[b]]:
                p_vsa_d = self.vsa_rule_executor.pmf2vec(self.vsa_cb_discrete_c,all_states[b,:8])
                result[b] = self.vsa_rule_executor.distribute_three(self.vsa_cb_discrete_a, p_vsa_d)
            else: 
                third_joint = torch.bmm(all_states[b:b+1, 6, :].unsqueeze(-1), all_states[b:b+1, 7, :].unsqueeze(1))
                joint = third_joint
                trans = self.trans[action[b:b+1]]
                pred = torch.bmm(joint.view(-1, 1, self.scene_dim_2), trans).squeeze(1)
                result[b] = normalize(pred)[0]
        return result

class NumberExecutor(GeneralExecutor):
    def __init__(self, scene_dim, device, vsa_cb_discrete_a, vsa_cb_discrete_c, vsa_cb_cont_a, vsa_cb_cont_c,
                action_set=["progression_one", "progression_mone", "arithmetic_plus", "arithmetic_minus", "distribute_three_left","distribute_three_right"], **kwargs):
        super(NumberExecutor, self).__init__(scene_dim, device, False, action_set,**kwargs)
        self.vsa_cb_discrete_a, self.vsa_cb_discrete_c = vsa_cb_discrete_a[:scene_dim], vsa_cb_discrete_c[:scene_dim]
        self.vsa_cb_cont_a, self.vsa_cb_cont_c = vsa_cb_cont_a[1:scene_dim+1], vsa_cb_cont_c[1:scene_dim+1] # no zero value for number
        self.vsa_cb_cont_1, self.vsa_cb_cont_2 = vsa_cb_cont_a[1], vsa_cb_cont_a[2]

class TypeExecutor(GeneralExecutor):
    def __init__(self, scene_dim, device, vsa_cb_discrete_a, vsa_cb_discrete_c, vsa_cb_cont_a, vsa_cb_cont_c, 
                action_set=["constant", "progression_one", "progression_mone", "progression_two", "progression_mtwo", "distribute_three_left","distribute_three_right"], **kwargs):
        super(TypeExecutor, self).__init__(scene_dim, device, True, action_set,**kwargs)
        self.vsa_cb_discrete_a, self.vsa_cb_discrete_c = vsa_cb_discrete_a[:scene_dim], vsa_cb_discrete_c[:scene_dim]
        self.vsa_cb_cont_a, self.vsa_cb_cont_c = vsa_cb_cont_a[1:scene_dim+1], vsa_cb_cont_c[1:scene_dim+1] # no zero value for type
        self.vsa_cb_cont_1, self.vsa_cb_cont_2 = vsa_cb_cont_a[1], vsa_cb_cont_a[2]

class SizeExecutor(GeneralExecutor):
    def __init__(self, scene_dim, device, vsa_cb_discrete_a, vsa_cb_discrete_c, vsa_cb_cont_a, vsa_cb_cont_c, 
                action_set=["constant", "progression_one", "progression_mone", "progression_two", "progression_mtwo",
                "arithmetic_plus", "arithmetic_minus", "distribute_three_left","distribute_three_right"],
                **kwargs):
        super(SizeExecutor, self).__init__(scene_dim, device, True, action_set,**kwargs)
        self.vsa_cb_discrete_a, self.vsa_cb_discrete_c = vsa_cb_discrete_a[:scene_dim], vsa_cb_discrete_c[:scene_dim]
        self.vsa_cb_cont_a, self.vsa_cb_cont_c = vsa_cb_cont_a[1:scene_dim+1], vsa_cb_cont_c[1:scene_dim+1] # no zero value for size
        self.vsa_cb_cont_1, self.vsa_cb_cont_2 = vsa_cb_cont_a[1], vsa_cb_cont_a[2]

class ColorExecutor(GeneralExecutor):
    def __init__(self, scene_dim, device, vsa_cb_discrete_a, vsa_cb_discrete_c, vsa_cb_cont_a, vsa_cb_cont_c, 
                action_set=["constant", "progression_one", "progression_mone", "progression_two", "progression_mtwo",
                            "arithmetic_plus", "arithmetic_minus", "distribute_three_left", "distribute_three_right"],
                **kwargs):
        super(ColorExecutor, self).__init__(scene_dim, device, True, action_set,**kwargs)
        self.vsa_cb_discrete_a, self.vsa_cb_discrete_c = vsa_cb_discrete_a[:scene_dim], vsa_cb_discrete_c[:scene_dim]
        self.vsa_cb_cont_a, self.vsa_cb_cont_c = vsa_cb_cont_a[:scene_dim], vsa_cb_cont_c[:scene_dim] # no zero value for size
        self.vsa_cb_cont_1, self.vsa_cb_cont_2 = vsa_cb_cont_a[1], vsa_cb_cont_a[2]
