#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*

import glob, os
import numpy as np
import torch as t

NCONST={"center_single":1, "distribute_four": 1, "distribute_nine":1,  "in_center_single_out_center_single":2, 
        "in_distribute_four_out_center_single":2, "left_center_single_right_center_single": 2, "up_center_single_down_center_single" : 2 }

class Dataset(t.utils.data.Dataset):
    def __init__(self, dataset_path, dataset_type, config, 
                test=False, gen_attribute="",gen_rule=""):
        '''
        Dataloader for the RAVEN/I-RAVEN dataset. In order to use the dataloader, make sure to first 
        run the prepocessing provided at https://github.com/WellyZhang/PrAE/blob/main/src/auxiliary/preprocess_rule.py

        
        Parameters
        ----------
        dataset_path:   str
            Path to the dataset 
        dataset_type:   str
            Choose one of the splits {train, val, test}
        config:         str
            Constellation (see NCONST keys)
        test:           boolean
            Dataloader returns no attribute rules if activated
        gen_attribute:  str
            If you want to do OOD generalization experiments, select the attribute here: {'Type', 'Size', 'Color'} 
        gen_rule:       str
            If you want to do OOD generalization experiments, select the rule here {'Constant', 'Progression', 'Distribute_Three'}
        
        '''

        self.dataset_path = dataset_path
        if gen_attribute !="": 
            # load only 
            loadfile_txt = os.path.join(dataset_path, config, "generalization_{:}_{:}_{:}.txt".format(gen_rule,gen_attribute,dataset_type))
            with open(loadfile_txt,'r') as f: 
                self.file_names = [line.replace('.xml\n','.npz') for line in f.readlines()]
        else: 
            # default load all data
            self.file_names = [f for f in glob.glob(os.path.join(self.dataset_path, config, "*.npz")) \
                           if dataset_type in f and "rule" not in f]
        self.config = config
        self.test = test

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)

        image = t.tensor(np.expand_dims(data["image"],1), dtype=t.float)
        target = t.tensor(data["target"], dtype=t.long)

        if self.test:
            rule_ret = 0
        else:
            rule_ret=[]

            for i in range(NCONST[self.config]):  
                rule_gt = np.load(data_path.replace(".npz", "_rule_comp{:}.npz".format(i)))
                pos_num_rule = t.tensor(rule_gt["pos_num_rule"], dtype=t.long)
                type_rule = t.tensor(rule_gt["type_rule"], dtype=t.long)
                size_rule = t.tensor(rule_gt["size_rule"], dtype=t.long)
                color_rule = t.tensor(rule_gt["color_rule"], dtype=t.long)
                rule = [pos_num_rule, type_rule, size_rule, color_rule]
                rule_ret.append(rule)

            rule_ret.reverse()

        return image, target, rule_ret