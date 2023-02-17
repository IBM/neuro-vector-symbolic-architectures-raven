#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*

import os, sys, time, torch, shutil

def check_paths(args):
    try:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        new_log_dir = os.path.join(args.log_dir, time.ctime().replace(" ", "-"))
        args.log_dir = new_log_dir
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',savedir=''):
    save_name = os.path.join(savedir, filename)
    torch.save(state, save_name)
    if is_best:
        save_name = os.path.join(savedir, 'model_best.pth.tar')
        shutil.copyfile(savedir+filename, save_name)

def save_codebooks(args,perception_cb, perception_imdict, backend_cb_cont, backend_cb_discrete): 
    dic = { "perception_cb": perception_cb, 
            "perception_imdict": perception_imdict,
            "backend_cb_cont" : backend_cb_cont,
            "backend_cb_discrete" : backend_cb_discrete
    }
    torch.save(dic,os.path.join(args.checkpoint_dir,"codebooks.pt")) 