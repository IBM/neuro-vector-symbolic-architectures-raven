#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*


# -*- code: utf-8 -*-
import glob
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


# This file maps the rules of PrAE to NVSA

# the rule/component idx: for configs with 2 components, run twice with different comp_idx
# left_right: 0 for left, 1 for right
# up_down: 0 for up, 1 for down
# in_out: 0 for out, 1 for in

################# DEFINE RULE MAP PrAE -> NVSA ###############################################

# for distribute_four (2x2)
# pos_num_rule_idx_map_four = {"Constant": 0, "Progression_One_Pos": 1, "Progression_Mone_Pos": 2, "Arithmetic_Plus_Pos": 3, "Arithmetic_Minus_Pos": 4, "Distribute_Three_Left_Pos": 5, "Distribute_Three_Right_Pos": 6,
                            #  "Progression_One_Num": 7, "Progression_Mone_Num": 8, "Arithmetic_Plus_Num": 9, "Arithmetic_Minus_Num": 10, "Distribute_Three_Left_Num": 11, "Distribute_Three_Right_Num": 12}
nvsa_ext_num_rule_idx_map_four = {0:0, 1:1, 2:2,3:3,4:4,5:5,6:5,7:7,8:8,9:9,10:10,11:11,12:11}

# for distribute_nine (3x3)
# pos_num_rule_idx_map_nine = {"Constant": 0, "Progression_One_Pos": 1, "Progression_Mone_Pos": 2, "Progression_Two_Pos": 3, "Progression_Mtwo_Pos": 4, "Arithmetic_Plus_Pos": 5, "Arithmetic_Minus_Pos": 6, "Distribute_Three_Left_Pos": 7,
                            #  "Distribute_Three_Right_Pos": 8, "Progression_One_Num": 9, "Progression_Mone_Num": 10, "Progression_Two_Num": 11, "Progression_Mtwo_Num": 12, "Arithmetic_Plus_Num": 13, "Arithmetic_Minus_Num": 14,
                            #  "Distribute_Three_Left_Num": 15, "Distribute_Three_Right_Num": 16}
nvsa_ext_num_rule_idx_map_nine = {0:0, 1:1, 2:2,3:3,4:4,5:5,6:6,7:7,8:7,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:15}

# for other configs, pos_num_rule can only be 0 (Constant)
# type_rule_idx_map = {"Constant": 0, "Progression_One": 1, "Progression_Mone": 2, "Progression_Two": 3, "Progression_Mtwo": 4, "Distribute_Three_Left": 5, "Distribute_Three_Right": 6}
nvsa_ext_type_rule_idx_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:5}

# size_rule_idx_map = {"Constant": 0, "Progression_One": 1, "Progression_Mone": 2, "Progression_Two": 3, "Progression_Mtwo": 4, "Arithmetic_Plus": 5, "Arithmetic_Minus": 6, "Distribute_Three_Left": 7, "Distribute_Three_Right": 8}
nvsa_ext_size_rule_idx_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:7}

# color_rule_idx_map = {"Constant": 0, "Progression_One": 1, "Progression_Mone": 2, "Progression_Two": 3, "Progression_Mtwo": 4, "Arithmetic_Plus": 5, "Arithmetic_Minus": 6, "Distribute_Three_Left": 7, "Distribute_Three_Right": 8}
nvsa_ext_color_rule_idx_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:7}
##################################################################################################



def merge_rules_nvsa_ext(all_action_rule,config):
    '''
    Merge the rules (mainly dist3) for NVSA

    Parameters
    ----------
    all_action_rule     list of tensors
        Ground-truth rules 
    config              str
        Constellation

    Return
    ------
    all_action_rule     list of tensor 

    '''

    if config =="center_single":
        all_action_rule = all_action_rule[0]
        bs = all_action_rule[0].shape[0]

        for b in range(bs):
            # type
            all_action_rule[1][b] = nvsa_ext_type_rule_idx_map[all_action_rule[1][b].item()]
            # size
            all_action_rule[2][b] = nvsa_ext_size_rule_idx_map[all_action_rule[2][b].item()]
            # color
            all_action_rule[3][b] = nvsa_ext_color_rule_idx_map[all_action_rule[3][b].item()]
        return [all_action_rule]
    elif config =="distribute_four": 
        all_action_rule = all_action_rule[0]
        bs = all_action_rule[0].shape[0]

        for b in range(bs):
            all_action_rule[0][b] = nvsa_ext_num_rule_idx_map_four[all_action_rule[0][b].item()] 
            # type
            all_action_rule[1][b] = nvsa_ext_type_rule_idx_map[all_action_rule[1][b].item()]
            # size
            all_action_rule[2][b] = nvsa_ext_size_rule_idx_map[all_action_rule[2][b].item()]
            # color
            all_action_rule[3][b] = nvsa_ext_color_rule_idx_map[all_action_rule[3][b].item()]
        return [all_action_rule]

    elif config =="distribute_nine": 
        all_action_rule = all_action_rule[0]
        bs = all_action_rule[0].shape[0]

        for b in range(bs):
            all_action_rule[0][b] = nvsa_ext_num_rule_idx_map_nine[all_action_rule[0][b].item()] 
            # type
            all_action_rule[1][b] = nvsa_ext_type_rule_idx_map[all_action_rule[1][b].item()]
            # size
            all_action_rule[2][b] = nvsa_ext_size_rule_idx_map[all_action_rule[2][b].item()]
            # color
            all_action_rule[3][b] = nvsa_ext_color_rule_idx_map[all_action_rule[3][b].item()]
        return [all_action_rule]
    if config =="up_center_single_down_center_single" or config == "left_center_single_right_center_single" or config == "in_center_single_out_center_single":

        # all_action_rule = all_action_rule[0]
        bs = all_action_rule[0][0].shape[0]
        # position no merge for now
        for i in range(2):
            for b in range(bs):
                # type
                all_action_rule[i][1][b] = nvsa_ext_type_rule_idx_map[all_action_rule[i][1][b].item()]
                # size
                all_action_rule[i][2][b] = nvsa_ext_size_rule_idx_map[all_action_rule[i][2][b].item()]
                # color
                all_action_rule[i][3][b] = nvsa_ext_color_rule_idx_map[all_action_rule[i][3][b].item()]
        return all_action_rule
    if  config == "in_distribute_four_out_center_single":

        # all_action_rule = all_action_rule[0]
        bs = all_action_rule[0][0].shape[0]
        # position no merge for now
        for b in range(bs):
            all_action_rule[0][0][b] = nvsa_ext_num_rule_idx_map_four[all_action_rule[0][0][b].item()] 
            # type
            all_action_rule[0][1][b] = nvsa_ext_type_rule_idx_map[all_action_rule[0][1][b].item()]
            # size
            all_action_rule[0][2][b] = nvsa_ext_size_rule_idx_map[all_action_rule[0][2][b].item()]
            # color
            all_action_rule[0][3][b] = nvsa_ext_color_rule_idx_map[all_action_rule[0][3][b].item()]

            # type
            all_action_rule[1][1][b] = nvsa_ext_type_rule_idx_map[all_action_rule[1][1][b].item()]
            # size
            all_action_rule[1][2][b] = nvsa_ext_size_rule_idx_map[all_action_rule[1][2][b].item()]
            # color
            all_action_rule[1][3][b] = nvsa_ext_color_rule_idx_map[all_action_rule[1][3][b].item()]
        return all_action_rule

def rule_todevice(all_action_rule, device): 
    '''
    Maps the each element in the rule list to 
    a specified device (cpu/gpu)

    Parameters
    ----------
    all_action_rule     list of tensors
        Ground-truth rules 
    device              str
        
    Return
    ------
    all_action_rule     list of tensor 

    '''
    all_action_rule_device = []
    for action_rule_const in all_action_rule:
        this_action_rule = []
        for action_rule in action_rule_const:
            action_rule = action_rule.to(device)
            this_action_rule.append(action_rule)
        all_action_rule_device.append(this_action_rule)
    return all_action_rule_device

def generate_rule_dataset(path, config): 
    '''
    Generate .txt files for OOD generalization for rule-attribute pairs. 
    (See Supplementary Note 4 of the paper)
    A separate dataset is generated for each combination of 
    attribute [Type, Size, Color] and rule [Constant, Progr, Dist3]

    Parameters
    ----------
    path     str
        Path to the RAVEN/I-RAVEN dataset 
    config   str
        Constellation 
    '''
    
    rule_idx = {"Type":1, "Size":2, "Color":3}
    
    for rule in ['Constant', 'Progression', 'Distribute_Three']: 
        for attribute in ['Type', 'Size', 'Color']:
            print(rule,attribute) 
            for split in ["train","val","test"]: 
                files = glob.glob(os.path.join(path, config, "*{:}.xml".format(split)))
                valid_filelist = []
                for file in tqdm(files): 
                    xml_tree = ET.parse(file)
                    xml_tree_root = xml_tree.getroot()
                    xml_rules = xml_tree_root[1]

                    is_target_rule = (xml_rules[0][rule_idx[attribute]].attrib["name"].casefold() == rule.casefold())
                    if config =="left_center_single_right_center_single": 
                        is_target_rule = is_target_rule or  (xml_rules[1][rule_idx[attribute]].attrib["name"].casefold() == rule.casefold())

                    if is_target_rule and split=="test": 
                        valid_filelist.append(file)  
                    elif (not is_target_rule) and not(split == "test"):
                        valid_filelist.append(file)  
                print(len(valid_filelist))

                file_name = os.path.join(path, config, "generalization_{:}_{:}_{:}.txt".format(rule,attribute,split))
                with open(file_name,'w') as fp: 
                    for item in valid_filelist: 
                        fp.write("{}\n".format(item))
    

if __name__ == '__main__':
    
    # path = "/dccstor/saentis/data/RAVEN/RAVEN-10000/"
    path = "/dccstor/saentis/data/I-RAVEN/"
    config = 'left_center_single_right_center_single'
    # config = 'center_single'
    generate_rule_dataset(path,config=config)