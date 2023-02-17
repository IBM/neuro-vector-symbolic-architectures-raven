#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*

import sys; sys.path.insert(0,'../')
import torch
from const import DIM_COLOR, DIM_SIZE, DIM_TYPE
from executor_nvsa_ext import (ColorExecutor, NumberExecutor, PositionExecutor, SizeExecutor, TypeExecutor)
from planner_nvsa_ext import (ColorPlanner, NumberPlanner, PositionPlanner, SizePlanner, TypePlanner)
from scene import SceneEngine
from util.utils import normalize, sample_action

def get_env(env_name, device,**kwargs):
    if env_name == "center_single":
        return CenterSingle(device,**kwargs)
    if env_name == "distribute_four":
        return DistributeFour(device,**kwargs)
    if env_name == "distribute_nine":
        return DistributeNine(device,**kwargs)
    if env_name == "in_center_single_out_center_single":
        return InCenterSingleOutCenterSingle(device,**kwargs)
    if env_name == "in_distribute_four_out_center_single":
        return InDistributeFourOutCenterSingle(device,**kwargs)
    if env_name == "left_center_single_right_center_single":
        return LeftCenterSingleRightCenterSingle(device,**kwargs)
    if env_name == "up_center_single_down_center_single":
        return UpCenterSingleDownCenterSingle(device,**kwargs)
    return None

class GeneralEnv(object):
    def __init__(self, num_slots, device,vsa_cb_discrete, vsa_cb_cont, **kwargs):
        self.num_slots = num_slots
        self.device = device
        self.vsa_cb_discrete_a, self.vsa_cb_discrete_c = vsa_cb_discrete, vsa_cb_discrete
        self.vsa_cb_cont_a, self.vsa_cb_cont_c = vsa_cb_cont, vsa_cb_cont 
        self.scene_engine = SceneEngine(self.num_slots, device)
        if "pos_action_set" in kwargs: 
            pos_action_set = kwargs["pos_action_set"]
            self.pos_planner = PositionPlanner(2 ** self.num_slots - 1, device, self.vsa_cb_discrete_a, self.vsa_cb_discrete_c, self.vsa_cb_cont_a, self.vsa_cb_cont_c, pos_action_set,**kwargs)
            self.pos_executor = PositionExecutor(2 ** self.num_slots - 1, device, self.vsa_cb_discrete_a, self.vsa_cb_discrete_c, self.vsa_cb_cont_a, self.vsa_cb_cont_c,pos_action_set,**kwargs)
        else:
            self.pos_planner = PositionPlanner(2 ** self.num_slots - 1, device, self.vsa_cb_discrete_a, self.vsa_cb_discrete_c, self.vsa_cb_cont_a, self.vsa_cb_cont_c,**kwargs)
            self.pos_executor = PositionExecutor(2 ** self.num_slots - 1, device, self.vsa_cb_discrete_a, self.vsa_cb_discrete_c, self.vsa_cb_cont_a, self.vsa_cb_cont_c,**kwargs)
        if "num_action_set" in kwargs:
            num_action_set = kwargs["num_action_set"]
            self.num_planner = NumberPlanner(self.num_slots, device, self.vsa_cb_discrete_a, self.vsa_cb_discrete_c, self.vsa_cb_cont_a, self.vsa_cb_cont_c,num_action_set,**kwargs)
            self.num_executor = NumberExecutor(self.num_slots, device, self.vsa_cb_discrete_a, self.vsa_cb_discrete_c, self.vsa_cb_cont_a, self.vsa_cb_cont_c,num_action_set,**kwargs)
        else:
            self.num_planner = NumberPlanner(self.num_slots, device, self.vsa_cb_discrete_a, self.vsa_cb_discrete_c, self.vsa_cb_cont_a, self.vsa_cb_cont_c,**kwargs)
            self.num_executor = NumberExecutor(self.num_slots, device,self.vsa_cb_discrete_a, self.vsa_cb_discrete_c, self.vsa_cb_cont_a, self.vsa_cb_cont_c,**kwargs)
        self.type_planner = TypePlanner(DIM_TYPE + 1, device,self.vsa_cb_discrete_a, self.vsa_cb_discrete_c, self.vsa_cb_cont_a, self.vsa_cb_cont_c,**kwargs)
        self.type_executor = TypeExecutor(DIM_TYPE + 1, device,self.vsa_cb_discrete_a, self.vsa_cb_discrete_c, self.vsa_cb_cont_a, self.vsa_cb_cont_c,**kwargs)
        self.size_planner = SizePlanner(DIM_SIZE + 1, device,self.vsa_cb_discrete_a, self.vsa_cb_discrete_c, self.vsa_cb_cont_a, self.vsa_cb_cont_c,**kwargs)
        self.size_executor = SizeExecutor(DIM_SIZE + 1, device,self.vsa_cb_discrete_a, self.vsa_cb_discrete_c, self.vsa_cb_cont_a, self.vsa_cb_cont_c,**kwargs)
        self.color_planner = ColorPlanner(DIM_COLOR + 1, device,self.vsa_cb_discrete_a, self.vsa_cb_discrete_c, self.vsa_cb_cont_a, self.vsa_cb_cont_c,**kwargs)
        self.color_executor = ColorExecutor(DIM_COLOR + 1, device,self.vsa_cb_discrete_a, self.vsa_cb_discrete_c, self.vsa_cb_cont_a, self.vsa_cb_cont_c,**kwargs)
   
    def prepare(self, model_output):
        return self.scene_engine.compute_scene_prob(*model_output)
   
    def action_prob(self, scene_logprob,vsa):
        pos_action_prob, pos_unnorm = self.pos_planner.action_prob(scene_logprob.position_logprob,vsa)
        num_action_prob, num_unnorm = self.num_planner.action_prob(scene_logprob.number_logprob,vsa)
        type_action_prob = self.type_planner.action_prob(scene_logprob.type_logprob,vsa)
        size_action_prob = self.size_planner.action_prob(scene_logprob.size_logprob,vsa)
        color_action_prob = self.color_planner.action_prob(scene_logprob.color_logprob,vsa)
        # relations on number and position are exclusive
        pos_num_select_prob = normalize(torch.cat([sum(pos_unnorm), sum(num_unnorm)], dim=-1))[0]
        pos_action_prob = pos_action_prob * pos_num_select_prob[:, 0].unsqueeze(-1)
        num_action_prob = num_action_prob * pos_num_select_prob[:, 1].unsqueeze(-1)
        pos_num_action_prob = torch.cat([pos_action_prob, num_action_prob], dim=-1)
        return pos_num_action_prob, type_action_prob, size_action_prob, color_action_prob
 
    def action(self, scene_logprob, sample=True, vsa=True):
        pos_action_prob, pos_unnorm = self.pos_planner.action_prob(scene_logprob.position_logprob,vsa)
        num_action_prob, num_unnorm = self.num_planner.action_prob(scene_logprob.number_logprob,vsa)
        # pdb.set_trace()
        type_action_prob = self.type_planner.action_prob(scene_logprob.type_logprob,vsa)
        # pdb.set_trace()
        size_action_prob = self.size_planner.action_prob(scene_logprob.size_logprob,vsa)
        color_action_prob = self.color_planner.action_prob(scene_logprob.color_logprob,vsa)
        pos_num_select_prob = normalize(torch.cat([sum(pos_unnorm), sum(num_unnorm)], dim=-1))[0]
        pos_num_action_prob = torch.cat([pos_action_prob * pos_num_select_prob[:, 0].unsqueeze(-1), 
                                         num_action_prob * pos_num_select_prob[:, 1].unsqueeze(-1)], dim=-1)
        # get action
        pos_num_select, pos_num_select_logprob = sample_action(pos_num_select_prob, sample=sample)
        pos_action, pos_action_logprob = sample_action(pos_action_prob, sample=sample)
        num_action, num_action_logprob = sample_action(num_action_prob, sample=sample)
        type_action, type_action_logprob = sample_action(type_action_prob, sample=sample)
        size_action, size_action_logprob = sample_action(size_action_prob, sample=sample)
        color_action, color_action_logprob = sample_action(color_action_prob, sample=sample)
        action = (pos_num_select, pos_action, num_action, type_action, size_action, color_action)
        action_logprob = (pos_num_select_logprob, pos_action_logprob, num_action_logprob, type_action_logprob, size_action_logprob, color_action_logprob)
        return action, action_logprob, (pos_num_action_prob, type_action_prob, size_action_prob, color_action_prob)

    def step(self, scene_prob, action):
        pos_action, num_action, type_action, size_action, color_action = action[1:]
        pos_pred = self.pos_executor.apply_position(scene_prob.position_prob, pos_action)
        num_pred = self.num_executor.apply(scene_prob.number_prob, num_action)
        type_pred = self.type_executor.apply(scene_prob.type_prob, type_action)
        size_pred = self.size_executor.apply(scene_prob.size_prob, size_action)
        color_pred = self.color_executor.apply(scene_prob.color_prob, color_action)
        pred = (pos_pred, num_pred, type_pred, size_pred, color_pred)
        return pred

    def score(self, select_action, pred, obs, error_function):
        pos_pred, num_pred, type_pred, size_pred, color_pred = pred
        pos_error = error_function(pos_pred.unsqueeze(1).expand(-1, 8, -1), obs.position_prob[:, 8:, :])
        num_error = error_function(num_pred.unsqueeze(1).expand(-1, 8, -1), obs.number_prob[:, 8:, :])
        type_error = error_function(type_pred.unsqueeze(1).expand(-1, 8, -1), obs.type_prob[:, 8:, :])
        size_error = error_function(size_pred.unsqueeze(1).expand(-1, 8, -1), obs.size_prob[:, 8:, :])
        color_error = error_function(color_pred.unsqueeze(1).expand(-1, 8, -1), obs.color_prob[:, 8:, :])
        pos_num_select = select_action.type(torch.float)
        pos_num_select_expanded = pos_num_select.unsqueeze(-1).expand(-1, 8)
        scores = -((1.0 - pos_num_select_expanded) * pos_error + pos_num_select_expanded * num_error + type_error + size_error + color_error)
        return scores

    def loss(self, select_action, logprob, pred, obs, targets, error_function):
        pos_pred, num_pred, type_pred, size_pred, color_pred = pred
        pos_num_select_logprob, pos_action_logprob, num_action_logprob, type_action_logprob, size_action_logprob, color_action_logprob = logprob
        pos_error = error_function(pos_pred.unsqueeze(1).expand(-1, 8, -1), obs.position_prob[:, 8:, :])
        num_error = error_function(num_pred.unsqueeze(1).expand(-1, 8, -1), obs.number_prob[:, 8:, :])
        type_error = error_function(type_pred.unsqueeze(1).expand(-1, 8, -1), obs.type_prob[:, 8:, :])
        size_error = error_function(size_pred.unsqueeze(1).expand(-1, 8, -1), obs.size_prob[:, 8:, :])
        color_error = error_function(color_pred.unsqueeze(1).expand(-1, 8, -1), obs.color_prob[:, 8:, :])

        pos_num_select = select_action.type(torch.float)
        pos_num_select_expanded = pos_num_select.unsqueeze(-1).expand(-1, 8)
        scores = -((1.0 - pos_num_select_expanded) * pos_error + pos_num_select_expanded * num_error + type_error + size_error + color_error)
        negative_reward = torch.nn.functional.cross_entropy(scores, targets, reduction="none")
        log = pos_num_select_logprob + (1.0 - pos_num_select) * pos_action_logprob + pos_num_select * num_action_logprob + \
              type_action_logprob + size_action_logprob + color_action_logprob
        loss = torch.mean(negative_reward + negative_reward.detach() * log)    
        return loss, scores, negative_reward.mean().item()


class CenterSingle(GeneralEnv):
    def __init__(self, device,**kwargs):
        super(CenterSingle, self).__init__(1, device,**kwargs)
   
    def action(self, scene_logprob, sample=True,vsa=True):
        # action prob
        type_action_prob = self.type_planner.action_prob(scene_logprob.type_logprob,vsa)
        size_action_prob = self.size_planner.action_prob(scene_logprob.size_logprob,vsa)
        color_action_prob = self.color_planner.action_prob(scene_logprob.color_logprob,vsa)
        # get action
        type_action, type_action_logprob = sample_action(type_action_prob, sample=sample)
        size_action, size_action_logprob = sample_action(size_action_prob, sample=sample)
        color_action, color_action_logprob = sample_action(color_action_prob, sample=sample)
        action = (None, None, None, type_action, size_action, color_action)
        action_logprob = (None, None, None, type_action_logprob, size_action_logprob, color_action_logprob)
        return action, action_logprob, (None, type_action_prob, size_action_prob, color_action_prob)
    
    def step(self, scene_prob, action):
        _, _, type_action, size_action, color_action = action[1:]
        type_pred = self.type_executor.apply(scene_prob.type_prob, type_action)
        size_pred = self.size_executor.apply(scene_prob.size_prob, size_action)
        color_pred = self.color_executor.apply(scene_prob.color_prob, color_action)
        pred = (None, None, type_pred, size_pred, color_pred)
        return pred
    
    def score(self, select_action, pred, obs, error_function):
        _, _, type_pred, size_pred, color_pred = pred
        type_error = error_function(type_pred.unsqueeze(1).expand(-1, 8, -1), obs.type_prob[:, 8:, :])
        size_error = error_function(size_pred.unsqueeze(1).expand(-1, 8, -1), obs.size_prob[:, 8:, :])
        color_error = error_function(color_pred.unsqueeze(1).expand(-1, 8, -1), obs.color_prob[:, 8:, :])
        scores = -(type_error + size_error + color_error)
        return scores

    def loss(self, select_action, logprob, pred, obs, targets, error_function):
        _, _, type_pred, size_pred, color_pred = pred
        _, _, _, type_action_logprob, size_action_logprob, color_action_logprob = logprob
        type_error = error_function(type_pred.unsqueeze(1).expand(-1, 8, -1), obs.type_prob[:, 8:, :])
        size_error = error_function(size_pred.unsqueeze(1).expand(-1, 8, -1), obs.size_prob[:, 8:, :])
        color_error = error_function(color_pred.unsqueeze(1).expand(-1, 8, -1), obs.color_prob[:, 8:, :])

        scores = -(type_error + size_error + color_error)
        negative_reward = torch.nn.functional.cross_entropy(scores, targets, reduction="none")
        log = type_action_logprob + size_action_logprob + color_action_logprob
        loss = torch.mean(negative_reward + negative_reward.detach() * log)    
        return loss, scores, negative_reward.mean().item()

class DistributeFour(GeneralEnv):
    def __init__(self, device,**kwargs):
        super(DistributeFour, self).__init__(4, device,**kwargs)

class DistributeNine(GeneralEnv):
    def __init__(self, device,**kwargs):
        super(DistributeNine, self).__init__(9, device, pos_action_set=["constant","progression_one", "progression_mone", "progression_two", "progression_mtwo",
                                                                        "arithmetic_plus", "arithmetic_minus","distribute_three_left", "distribute_three_right"], 
                                                        num_action_set=["progression_one", "progression_mone", "progression_two", "progression_mtwo",
                                                                        "arithmetic_plus", "arithmetic_minus", "distribute_three_left", "distribute_three_right"],**kwargs) 

class OutCenterSingle(GeneralEnv):

    def __init__(self, device,**kwargs):
        super(OutCenterSingle, self).__init__(1, device,**kwargs)

    def action(self, scene_logprob, sample=True, vsa=True):
        # action prob
        type_action_prob = self.type_planner.action_prob(scene_logprob.type_logprob,vsa)
        size_action_prob = self.size_planner.action_prob(scene_logprob.size_logprob,vsa)
        # get action
        type_action, type_action_logprob = sample_action(type_action_prob, sample=sample)
        size_action, size_action_logprob = sample_action(size_action_prob, sample=sample)
        action = (None, None, None, type_action, size_action, None)
        action_logprob = (None, None, None, type_action_logprob, size_action_logprob, None)
        return action, action_logprob, (None, type_action_prob, size_action_prob, None)

    def step(self, scene_prob, action):
        _, _, type_action, size_action, _ = action[1:]
        type_pred = self.type_executor.apply(scene_prob.type_prob, type_action)
        size_pred = self.size_executor.apply(scene_prob.size_prob, size_action)
        pred = (None, None, type_pred, size_pred, None)
        return pred

    def score(self, select_action, pred, obs, error_function):
        _, _, type_pred, size_pred, _ = pred
        type_error = error_function(type_pred.unsqueeze(1).expand(-1, 8, -1), obs.type_prob[:, 8:, :])
        size_error = error_function(size_pred.unsqueeze(1).expand(-1, 8, -1), obs.size_prob[:, 8:, :])
        scores = -(type_error + size_error)
        return scores

    def loss(self, select_action, logprob, pred, obs, targets, error_function):
        _, _, type_pred, size_pred, _ = pred
        _, _, _, type_action_logprob, size_action_logprob, _ = logprob
        type_error = error_function(type_pred.unsqueeze(1).expand(-1, 8, -1), obs.type_prob[:, 8:, :])
        size_error = error_function(size_pred.unsqueeze(1).expand(-1, 8, -1), obs.size_prob[:, 8:, :])

        scores = -(type_error + size_error)
        negative_reward = torch.nn.functional.cross_entropy(scores, targets, reduction="none")
        log = type_action_logprob + size_action_logprob
        loss = torch.mean(negative_reward + negative_reward.detach() * log)    
        return loss, scores, negative_reward.mean().item()

class InCenterSingleOutCenterSingle(object):

    def __init__(self, device,**kwargs):
        self.in_center_single = CenterSingle(device,**kwargs)
        self.out_center_single = OutCenterSingle(device,**kwargs)

    def prepare(self, model_output):
        in_component = []
        out_component = []
        for element in model_output:
            in_component.append(element[:, :, 1:, :])
            out_component.append(element[:, :, :1, :])
        in_scene_prob, in_scene_logprob = self.in_center_single.prepare(in_component)
        out_scene_prob, out_scene_logprob = self.out_center_single.prepare(out_component)
        return (in_scene_prob, out_scene_prob), (in_scene_logprob, out_scene_logprob)

    def action(self, scene_logprob, sample=True,vsa=True):
        in_action, in_action_logprob, in_action_prob = self.in_center_single.action(scene_logprob[0], sample,vsa)
        out_action, out_action_logprob, out_action_prob = self.out_center_single.action(scene_logprob[1], sample,vsa)
        return (in_action, out_action), (in_action_logprob, out_action_logprob), (in_action_prob, out_action_prob)

    def step(self, scene_prob, action):
        in_pred = self.in_center_single.step(scene_prob[0], action[0])
        out_pred = self.out_center_single.step(scene_prob[1], action[1])
        return (in_pred, out_pred)

    def score(self, select_action, pred, obs, error_function):
        _, _, in_type_pred, in_size_pred, in_color_pred = pred[0]
        in_type_error = error_function(in_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].type_prob[:, 8:, :])
        in_size_error = error_function(in_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].size_prob[:, 8:, :])
        in_color_error = error_function(in_color_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].color_prob[:, 8:, :])
        _, _, out_type_pred, out_size_pred, _ = pred[1]
        out_type_error = error_function(out_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].type_prob[:, 8:, :])
        out_size_error = error_function(out_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].size_prob[:, 8:, :])
        scores = -(in_type_error + in_size_error + in_color_error + out_type_error + out_size_error)
        return scores

    def loss(self, select_action, logprob, pred, obs, targets, error_function):
        _, _, in_type_pred, in_size_pred, in_color_pred = pred[0]
        _, _, _, in_type_action_logprob, in_size_action_logprob, in_color_action_logprob = logprob[0]
        in_type_error = error_function(in_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].type_prob[:, 8:, :])
        in_size_error = error_function(in_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].size_prob[:, 8:, :])
        in_color_error = error_function(in_color_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].color_prob[:, 8:, :])

        _, _, out_type_pred, out_size_pred, _ = pred[1]
        _, _, _, out_type_action_logprob, out_size_action_logprob, _ = logprob[1]
        out_type_error = error_function(out_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].type_prob[:, 8:, :])
        out_size_error = error_function(out_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].size_prob[:, 8:, :])

        scores = -(in_type_error + in_size_error + in_color_error + out_type_error + out_size_error)
        negative_reward = torch.nn.functional.cross_entropy(scores, targets, reduction="none")
        log = in_type_action_logprob + in_size_action_logprob + in_color_action_logprob + out_type_action_logprob + out_size_action_logprob
        loss = torch.mean(negative_reward + negative_reward.detach() * log)
        return loss, scores, negative_reward.mean().item()


class InDistributeFourOutCenterSingle(object):

    def __init__(self, device,**kwargs):
        self.in_distribute_four = DistributeFour(device,**kwargs)
        self.out_center_single = OutCenterSingle(device,**kwargs)

    def prepare(self, model_output):
        in_component = []
        out_component = []
        for element in model_output:
            in_component.append(element[:, :, 1:5, :])
            out_component.append(element[:, :, :1, :])
        in_scene_prob, in_scene_logprob = self.in_distribute_four.prepare(in_component)
        out_scene_prob, out_scene_logprob = self.out_center_single.prepare(out_component)
        return (in_scene_prob, out_scene_prob), (in_scene_logprob, out_scene_logprob)

    def action(self, scene_logprob, sample=True,vsa=True):
        in_action, in_action_logprob, in_action_prob = self.in_distribute_four.action(scene_logprob[0], sample, vsa)
        out_action, out_action_logprob, out_action_prob = self.out_center_single.action(scene_logprob[1], sample, vsa)
        return (in_action, out_action), (in_action_logprob, out_action_logprob), (in_action_prob, out_action_prob)

    def step(self, scene_prob, action):
        in_pred = self.in_distribute_four.step(scene_prob[0], action[0])
        out_pred = self.out_center_single.step(scene_prob[1], action[1])
        return (in_pred, out_pred)

    def score(self, select_action, pred, obs, error_function):
        in_pos_pred, in_num_pred, in_type_pred, in_size_pred, in_color_pred = pred[0]
        in_pos_error = error_function(in_pos_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].position_prob[:, 8:, :])
        in_num_error = error_function(in_num_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].number_prob[:, 8:, :])
        in_type_error = error_function(in_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].type_prob[:, 8:, :])
        in_size_error = error_function(in_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].size_prob[:, 8:, :])
        in_color_error = error_function(in_color_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].color_prob[:, 8:, :])
        in_pos_num_select = select_action[0].type(torch.float)
        in_pos_num_select_expanded = in_pos_num_select.unsqueeze(-1).expand(-1, 8)
        _, _, out_type_pred, out_size_pred, _ = pred[1]
        out_type_error = error_function(out_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].type_prob[:, 8:, :])
        out_size_error = error_function(out_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].size_prob[:, 8:, :])
        scores = -((1.0 - in_pos_num_select_expanded) * in_pos_error + in_pos_num_select_expanded * in_num_error + 
                   in_type_error + in_size_error + in_color_error + out_type_error + out_size_error)
        return scores

    def loss(self, select_action, logprob, pred, obs, targets, error_function):
        in_pos_pred, in_num_pred, in_type_pred, in_size_pred, in_color_pred = pred[0]
        in_pos_num_select_logprob, in_pos_action_logprob, in_num_action_logprob, in_type_action_logprob, in_size_action_logprob, in_color_action_logprob = logprob[0]
        in_pos_error = error_function(in_pos_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].position_prob[:, 8:, :])
        in_num_error = error_function(in_num_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].number_prob[:, 8:, :])
        in_type_error = error_function(in_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].type_prob[:, 8:, :])
        in_size_error = error_function(in_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].size_prob[:, 8:, :])
        in_color_error = error_function(in_color_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].color_prob[:, 8:, :])

        in_pos_num_select = select_action[0].type(torch.float)
        in_pos_num_select_expanded = in_pos_num_select.unsqueeze(-1).expand(-1, 8)

        _, _, out_type_pred, out_size_pred, _ = pred[1]
        _, _, _, out_type_action_logprob, out_size_action_logprob, _ = logprob[1]
        out_type_error = error_function(out_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].type_prob[:, 8:, :])
        out_size_error = error_function(out_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].size_prob[:, 8:, :])

        scores = -((1.0 - in_pos_num_select_expanded) * in_pos_error + in_pos_num_select_expanded * in_num_error + 
                   in_type_error + in_size_error + in_color_error + out_type_error + out_size_error)

        negative_reward = torch.nn.functional.cross_entropy(scores, targets, reduction="none")
        log = in_pos_num_select_logprob + (1.0 - in_pos_num_select) * in_pos_action_logprob + in_pos_num_select * in_num_action_logprob + \
              in_type_action_logprob + in_size_action_logprob + in_color_action_logprob + out_type_action_logprob + out_size_action_logprob
        loss = torch.mean(negative_reward + negative_reward.detach() * log)
        return loss, scores, negative_reward.mean().item()

class LeftCenterSingleRightCenterSingle(object):
    def __init__(self, device,**kwargs):
        self.left_center_single = CenterSingle(device,**kwargs)
        self.right_center_single = CenterSingle(device,**kwargs)

    def prepare(self, model_output):
        left_component = []
        right_component = []
        for element in model_output:
            left_component.append(element[:, :, :1, :])
            right_component.append(element[:, :, 1:, :])
        left_scene_prob, left_scene_logprob = self.left_center_single.prepare(left_component)
        right_scene_prob, right_scene_logprob = self.right_center_single.prepare(right_component)
        return (left_scene_prob, right_scene_prob), (left_scene_logprob, right_scene_logprob)

    def action(self, scene_logprob, sample=True, vsa=True):
        left_action, left_action_logprob, left_action_prob = self.left_center_single.action(scene_logprob[0], sample,vsa)
        right_action, right_action_logprob, right_action_prob = self.right_center_single.action(scene_logprob[1], sample,vsa)
        return (left_action, right_action), (left_action_logprob, right_action_logprob), (left_action_prob, right_action_prob)

    def step(self, scene_prob, action):
        left_pred = self.left_center_single.step(scene_prob[0], action[0])
        right_pred = self.right_center_single.step(scene_prob[1], action[1])
        return (left_pred, right_pred)

    def score(self, select_action, pred, obs, error_function):
        _, _, left_type_pred, left_size_pred, left_color_pred = pred[0]
        left_type_error = error_function(left_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].type_prob[:, 8:, :])
        left_size_error = error_function(left_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].size_prob[:, 8:, :])
        left_color_error = error_function(left_color_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].color_prob[:, 8:, :])
        _, _, right_type_pred, right_size_pred, right_color_pred = pred[1]
        right_type_error = error_function(right_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].type_prob[:, 8:, :])
        right_size_error = error_function(right_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].size_prob[:, 8:, :])
        right_color_error = error_function(right_color_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].color_prob[:, 8:, :])
        scores = -(left_type_error + left_size_error + left_color_error + right_type_error + right_size_error + right_color_error)
        return scores

    def loss(self, select_action, logprob, pred, obs, targets, error_function):
        _, _, left_type_pred, left_size_pred, left_color_pred = pred[0]
        _, _, _, left_type_action_logprob, left_size_action_logprob, left_color_action_logprob = logprob[0]
        left_type_error = error_function(left_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].type_prob[:, 8:, :])
        left_size_error = error_function(left_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].size_prob[:, 8:, :])
        left_color_error = error_function(left_color_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].color_prob[:, 8:, :])

        _, _, right_type_pred, right_size_pred, right_color_pred = pred[1]
        _, _, _, right_type_action_logprob, right_size_action_logprob, right_color_action_logprob = logprob[1]
        right_type_error = error_function(right_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].type_prob[:, 8:, :])
        right_size_error = error_function(right_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].size_prob[:, 8:, :])
        right_color_error = error_function(right_color_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].color_prob[:, 8:, :])

        scores = -(left_type_error + left_size_error + left_color_error + right_type_error + right_size_error + right_color_error)
        negative_reward = torch.nn.functional.cross_entropy(scores, targets, reduction="none")
        log = right_type_action_logprob + right_size_action_logprob + right_color_action_logprob + left_type_action_logprob + \
              left_size_action_logprob + left_color_action_logprob
        loss = torch.mean(negative_reward + negative_reward.detach() * log)
        return loss, scores, negative_reward.mean().item()

class UpCenterSingleDownCenterSingle(object):
    def __init__(self, device,**kwargs):
        self.up_center_single = CenterSingle(device,**kwargs)
        self.down_center_single = CenterSingle(device,**kwargs)

    def prepare(self, model_output):
        up_component = []
        down_component = []
        for element in model_output:
            up_component.append(element[:, :, :1, :])
            down_component.append(element[:, :, 1:, :])
        up_scene_prob, up_scene_logprob = self.up_center_single.prepare(up_component)
        down_scene_prob, down_scene_logprob = self.down_center_single.prepare(down_component)
        return (up_scene_prob, down_scene_prob), (up_scene_logprob, down_scene_logprob)

    def action(self, scene_logprob, sample=True,vsa=True):
        up_action, up_action_logprob, up_action_prob = self.up_center_single.action(scene_logprob[0], sample,vsa)
        down_action, down_action_logprob, down_action_prob = self.down_center_single.action(scene_logprob[1], sample,vsa)
        return (up_action, down_action), (up_action_logprob, down_action_logprob), (up_action_prob, down_action_prob)

    def step(self, scene_prob, action):
        up_pred = self.up_center_single.step(scene_prob[0], action[0])
        down_pred = self.down_center_single.step(scene_prob[1], action[1])
        return (up_pred, down_pred)

    def score(self, select_action, pred, obs, error_function):
        _, _, up_type_pred, up_size_pred, up_color_pred = pred[0]
        up_type_error = error_function(up_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].type_prob[:, 8:, :])
        up_size_error = error_function(up_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].size_prob[:, 8:, :])
        up_color_error = error_function(up_color_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].color_prob[:, 8:, :])
        _, _, down_type_pred, down_size_pred, down_color_pred = pred[1]
        down_type_error = error_function(down_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].type_prob[:, 8:, :])
        down_size_error = error_function(down_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].size_prob[:, 8:, :])
        down_color_error = error_function(down_color_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].color_prob[:, 8:, :])
        scores = -(up_type_error + up_size_error + up_color_error + down_type_error + down_size_error + down_color_error)
        return scores

    def loss(self, select_action, logprob, pred, obs, targets, error_function):
        _, _, up_type_pred, up_size_pred, up_color_pred = pred[0]
        _, _, _, up_type_action_logprob, up_size_action_logprob, up_color_action_logprob = logprob[0]
        up_type_error = error_function(up_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].type_prob[:, 8:, :])
        up_size_error = error_function(up_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].size_prob[:, 8:, :])
        up_color_error = error_function(up_color_pred.unsqueeze(1).expand(-1, 8, -1), obs[0].color_prob[:, 8:, :])

        _, _, down_type_pred, down_size_pred, down_color_pred = pred[1]
        _, _, _, down_type_action_logprob, down_size_action_logprob, down_color_action_logprob = logprob[1]
        down_type_error = error_function(down_type_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].type_prob[:, 8:, :])
        down_size_error = error_function(down_size_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].size_prob[:, 8:, :])
        down_color_error = error_function(down_color_pred.unsqueeze(1).expand(-1, 8, -1), obs[1].color_prob[:, 8:, :])

        scores = -(up_type_error + up_size_error + up_color_error + down_type_error + down_size_error + down_color_error)
        negative_reward = torch.nn.functional.cross_entropy(scores, targets, reduction="none")
        log = up_type_action_logprob + up_size_action_logprob + up_color_action_logprob + down_type_action_logprob + \
              down_size_action_logprob + down_color_action_logprob
        loss = torch.mean(negative_reward + negative_reward.detach() * log)
        return loss, scores, negative_reward.mean().item()