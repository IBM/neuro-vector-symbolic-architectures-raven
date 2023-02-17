#*----------------------------------------------------------------------------*
#* Copyright (C) 2023 IBM Inc. All rights reserved                            *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*----------------------------------------------------------------------------*


import argparse, os, sys; sys.path.insert(0,'../'); sys.path.insert(0,'.')
import numpy as np
import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.averagemeter import AverageMeter
from rulemap import rule_todevice, merge_rules_nvsa_ext
import criteria
from torch.utils.tensorboard import SummaryWriter
from const import NUMPOS, DIM_COLOR, DIM_SIZE, DIM_TYPE
from dataset_nvsa import Dataset
import env_nvsa_ext as env_nvsa_ext
from util.checkpath import check_paths, save_checkpoint, save_codebooks

# NVSA imports
from nvsa.perception.resnet import resnet18
from nvsa.perception.metrics import  hd_mult_frontend, generate_IM
from nvsa.reasoning.vsa_block_utils import block_discrete_codebook, block_continuous_codebook


def generate_nvsa_codebooks(args,rng): 
    '''
    Generate the codebooks for NVSA frontend and backend. 
    The codebook can also be loaded if it is stored under args.resume/
    '''
    imfilename = os.path.join(args.resume,"codebooks.pt")
    if os.path.isfile(imfilename):
        print("Load predefined NVSA codebooks")  
        data= torch.load(imfilename, map_location='cpu')
        perception_cb = data["perception_cb"]
        perception_imdict = data["perception_imdict"]
        backend_cb_cont = data['backend_cb_cont'].to(args.device)
        backend_cb_discrete = data['backend_cb_discrete'].to(args.device)
    else: 
        print("Generate new NVSA codebooks")
        backend_cb_cont, _ = block_continuous_codebook(device=args.device,scene_dim=12, d=args.nvsa_backend_d,k=args.nvsa_backend_k, rng=rng)  
        backend_cb_discrete, _ = block_discrete_codebook(device=args.device, d=args.nvsa_backend_d,k=args.nvsa_backend_k, rng=rng)  
        perception_cb, perception_imdict = generate_IM(args.d, (DIM_TYPE, DIM_SIZE, DIM_COLOR, args.num_pos) ,rng) 
    
    return perception_cb, perception_imdict, backend_cb_cont, backend_cb_discrete

def train(args, env,perception_cb, device):
    '''
    End-to-end training and validation of NVSA
    '''
    def train_epoch(epoch):

        model.train()

        # Define tracking meters
        xe_loss_avg = AverageMeter('XE Loss', ':.3f')
        aux_loss_avg = AverageMeter('Aux Loss', ':.3f')
        acc_avg = AverageMeter('Acccuracy', ':.3f')

        for counter, (images, targets, all_action_rule) in enumerate(tqdm(train_loader)):
            images, targets = images.to(device), targets.to(device)
            # Merge auxiliary labels (attribute rules) and map them to device (GPU/CPU)
            all_action_rule = merge_rules_nvsa_ext(all_action_rule,args.config)
            all_action_rule_device = rule_todevice(all_action_rule,device)
            [B,N,_, H,W] = images.shape

            # Pass images through trainable ResNet18
            model_output = model(images.view(B*N,1,H,W))
            # Compare query with codebook and compute probabilities
            marg_prob = metric_fc(model_output,B,N)
            # Infer the scene probabilities
            scene_prob, scene_logprob = env.prepare(marg_prob)
            # Rule probability computation 
            action, action_logprob, all_action_prob = env.action(scene_logprob)  
            # Rule execution
            pred = env.step(scene_prob, action)
            # Compute loss (JSD) and scores
            loss, scores, xe_loss_item = env.loss(action[0], action_logprob, pred, scene_prob, targets, criteria.JSD)
            # Compute the auxiliary loss (with attribute rules)
            aux_loss, aux_loss_item = criteria.aux_loss_multi(all_action_prob, all_action_rule_device) 
            # Final loss 
            final_loss = args.main_loss * loss + args.aux * aux_loss

            acc = criteria.calculate_acc(scores, targets)
            xe_loss_avg.update(xe_loss_item,images.size(0))
            aux_loss_avg.update(aux_loss_item,images.size(0))
            acc_avg.update(acc.item(),images.size(0))

            optimizer.zero_grad()
            final_loss.backward()

            # Clip gradients with l2 normalization
            if args.clip: 
                val = torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.clip, norm_type=2.0)

            optimizer.step()

        print("Epoch {}, Total Iter: {}, Train Avg XE: {:.6f}".format(epoch, counter, xe_loss_avg.avg))
        writer.add_scalar('loss/training_XE',xe_loss_avg.avg,epoch)
        writer.add_scalar('loss/training_AUX',aux_loss_avg.avg,epoch)
        writer.add_scalar('acc/train',acc_avg.avg,epoch)
        metric_fc.store_s(writer,epoch) # Track the (trainable) s value

        return 

    def validate_epoch(epoch):
        model.eval()
        xe_loss_avg = AverageMeter('XE Loss', ':.3f')
        acc_avg = AverageMeter('Acccuracy', ':.3f')
        for counter, (images, targets, all_action_rule) in enumerate(tqdm(val_loader)):
            images, targets = images.to(device), targets.to(device)
            [B,N,_, H,W] = images.shape
        
            # Pass images through trainable ResNet18
            model_output = model(images.view(B*N,1,H,W))
            # Compare query with codebook and compute probabilities
            marg_prob = metric_fc(model_output,B,N)
            # Infer the scene probabilities
            scene_prob, scene_logprob = env.prepare(marg_prob)
            # Rule probability computation 
            action, action_logprob, _ = env.action(scene_logprob, sample=False)
            # Rule execution
            pred = env.step(scene_prob, action)
            # Compute loss (JSD) and scores
            loss, scores, xe_loss_item = env.loss(action[0], action_logprob, pred, scene_prob, targets, criteria.JSD)
            xe_loss_avg.update(loss.item(),images.size(0))
            acc = criteria.calculate_acc(scores, targets)
            acc_avg.update(acc.item(),images.size(0))

        print("Epoch {}, Valid Avg XE: {:.6f}, Valid Avg Acc: {:.4f}".format(epoch, xe_loss_avg.avg, acc_avg.avg))
        writer.add_scalar('loss/val_XE',xe_loss_avg.avg,epoch)
        writer.add_scalar('acc/val',acc_avg.avg,epoch)
        return acc_avg.avg
  
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    writer = SummaryWriter(args.log_dir)

    # Init ResNet18 model  
    model = resnet18(pretrained = args.pretrained, progress = True, num_classes=args.d, no_maxpool=args.no_maxpool)
    
    # Init VSA readout + marginalization
    metric_fc = hd_mult_frontend(num_features = args.d, num_classes = 5, mat=perception_cb,s= args.s,trainable_s=args.trainable_s,
                            marg_m = args.m, marg_in_act=args.in_act,num_pos=args.num_pos, fixed_weights=1)
    # Put models to devices
    model.to(device)
    metric_fc.to(device)

    # Init optimizer
    train_param = [param for param in model.parameters() if param.requires_grad]
    train_param += [param for param in metric_fc.parameters() if param.requires_grad]
    optimizer = optim.Adam(train_param, args.lr, weight_decay=args.weight_decay)

    # Load all checkpoint 
    model_path = os.path.join(args.resume,"checkpoint.pth.tar")
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict_model'])
        metric_fc.load_state_dict(checkpoint['state_dict_metric_fc'])
        start_epoch=checkpoint['epoch']
        best_acc=checkpoint["best_acc"]
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' at Epoch {:.3f}".format(model_path,checkpoint["epoch"]))
    else:
        best_acc = 0
        start_epoch = 0
   
    # dataset loader
    train_set = Dataset(args.dataset, "train", args.config, gen_attribute=args.gen_attribute, gen_rule=args.gen_rule)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_set = Dataset(args.dataset, "val", args.config, test=True, gen_attribute=args.gen_attribute, gen_rule=args.gen_rule)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)

    # training loop starts
    for epoch in range(start_epoch,args.epochs):
        train_epoch(epoch)
        with torch.no_grad():
            acc = validate_epoch(epoch)

        # store model(s)
        is_best = acc > best_acc
        best_acc = max(acc,best_acc)
        save_checkpoint({'epoch': epoch + 1, 'controller': "resnet18", 'state_dict_model': model.state_dict(),
                'state_dict_metric_fc': metric_fc.state_dict(), 'best_acc': best_acc, 'optimizer' : optimizer.state_dict()}, is_best,savedir=args.checkpoint_dir)

    return writer

def test(args, env, device, perception_cb, writer = None, dset="RAVEN"):
    '''
    End-to-end testing NVSA
    '''
    def test_epoch():
        model.eval()
        xe_loss_avg = AverageMeter('XE Loss', ':.3f')
        acc_avg = AverageMeter('Acccuracy', ':.3f')
        for counter, (images, targets, all_action_rule) in enumerate(tqdm(test_loader)):
            images, targets = images.to(device), targets.to(device)
            [B,N,_, H,W] = images.shape

            # Pass images through ResNet18
            model_output = model(images.view(B*N,1,H,W))
            # Compare query with codebook and compute probabilities
            marg_prob = metric_fc(model_output,B,N)
            # Infer the scene probabilities
            scene_prob, scene_logprob = env.prepare(marg_prob)
            # Rule probability computation 
            action, action_logprob, all_action_prob = env.action(scene_logprob, sample=False)
            # Rule execution
            pred = env.step(scene_prob, action)
            # Compute loss (JSD) and scores
            loss, scores, xe_loss_item = env.loss(action[0], action_logprob, pred, scene_prob, targets, criteria.JSD)
            xe_loss_avg.update(loss.item(),images.size(0))
            acc = criteria.calculate_acc(scores, targets)
            acc_avg.update(acc.item(),images.size(0))

        # Save final result as npz (and potentially in Tensorboard) 
        if not (writer is None):  
            writer.add_scalar("acc/test-{}".format(dset), acc_avg.avg, 0)
            np.savez(args.save_dir+"result_{:}.npz".format(dset),acc = acc_avg.avg )
        else:
            args.save_dir = args.resume.replace("ckpt/","save/")
            np.savez(args.save_dir+"result_{:}.npz".format(dset),acc = acc_avg.avg )

        print("Test Avg Acc: {:.4f}".format(acc_avg.avg))

    
    # Load all checkpoint 
    model_path = os.path.join(args.resume,"model_best.pth.tar")
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        print("=> loaded checkpoint '{}' with acc {:.3f}".format(model_path,checkpoint["best_acc"]))
    else:
        raise ValueError("No checkpoint found at {:}".format(model_path)) 
   
    # Load ResNet18 model  
    model = resnet18(pretrained = args.pretrained, progress = True, num_classes=args.d, no_maxpool=args.no_maxpool)
    model.to(device)
    model.load_state_dict(checkpoint['state_dict_model'])

    # Init VSA readout + marginalization
    metric_fc = hd_mult_frontend(num_features = args.d, num_classes = 5, mat=perception_cb, marg_m = args.m, marg_in_act=args.in_act,num_pos=args.num_pos)
    metric_fc.to(device)
    metric_fc.load_state_dict(checkpoint['state_dict_metric_fc'])
    
    # dataset loader
    test_set = Dataset(args.dataset, "test", args.config, test=False, gen_attribute=args.gen_attribute, gen_rule=args.gen_rule)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)

    print("Evaluating on {}".format(args.config))
    with torch.no_grad():
        test_epoch()
    return writer

def main(): 
    arg_parser = argparse.ArgumentParser(description='NVSA training and evaluation on RAVEN/I-RAVEN')

    arg_parser.add_argument("--mode", type=str, default="train", help="train/test")
    arg_parser.add_argument("--exp_dir", type=str, default ="./" )
    arg_parser.add_argument("--resume", type=str, default='', help="resume from a initialized model")
    arg_parser.add_argument("--seed", type=int, default=1234, help="random number seed")
    arg_parser.add_argument("--run", type=int, default=0, help="Run id ")

    # Dataset 
    arg_parser.add_argument("--dataset", type=str, default="./", help="dataset path to RAVEN")
    arg_parser.add_argument("--dataset-i-raven", type=str, default="", help="dataset path to I-RAVEN")
    arg_parser.add_argument("--config", type=str, default="center_single", help="the configuration used for training")
    arg_parser.add_argument('--gen-attribute',type=str, default="", help="Generalization experiment [Type, Size, Color]")
    arg_parser.add_argument('--gen-rule',type=str, default="", help="Generalization experiment [Arithmetic, Constant, Progression, Distribute_Three]")

    # Training hyperparameters
    arg_parser.add_argument("--epochs", type=int, default=100, help="the number of training epochs")
    arg_parser.add_argument("--batch-size", type=int, default=16, help="size of batch")
    arg_parser.add_argument("--device", type=int, default=0, help="device index for GPU; if GPU unavailable, leave it as default")
    arg_parser.add_argument("--lr", type=float, default=0.95e-4, help="learning rate")
    arg_parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay of optimizer, same as l2 reg")
    arg_parser.add_argument("--num-workers", type=int, default=8, help="number of workers for data loader")
    arg_parser.add_argument("--aux", type=float, default=1.0, help="weight of auxiliary loss (attribute-rules)")
    arg_parser.add_argument("--main-loss", type=float, default=1.0, help="weight of the main loss")
    arg_parser.add_argument("--clip", type=float, default=10, help="Max value/norm in gradient clipping (now l2 norm)")
    
    # ResNet18 settings
    arg_parser.add_argument("--pretrained", type=int, default=1, help="Use pretrained ResNet (ImageNet-1k)")
    arg_parser.add_argument("--no-maxpool", type=int, default=1, help="Use no maxpool and different stride in input")

    # NVSA frontend settings 
    arg_parser.add_argument('--d', type=int, default=512, help="Frontend VSA dimension (output of ResNet18)")
    arg_parser.add_argument('--s', type=float, default=7., help="Inverse softmax temperature in marginalization")
    arg_parser.add_argument('--trainable-s', action="store_true", help="Trainable inverse softmax temperature in marginalization")
    arg_parser.add_argument('--m', type=float, default=0., help="Potential threshold in marginalization")
    arg_parser.add_argument('--in-act', type=str, default="ReLU",help="Activation function in marginalization for all attributes except exist")
    arg_parser.add_argument('--exist-act', type=str, default="Identity",help="Activation function in marginalization for exist")

    # NVSA backend settings
    arg_parser.add_argument('--nvsa-backend-d', type=int, default=1024, help="VSA dimension in backend" )
    arg_parser.add_argument('--nvsa-backend-k', type=int, default=4, help="Number of blocks in VSA vectors")
    arg_parser.add_argument('--detector-act', type=str, default="threshold", help="Activation on rule probabilities")
    arg_parser.add_argument('--detector-m', type=float, default=0.0, help="Shift of rule probabilities")
    arg_parser.add_argument('--executor-act', type=str, default="ReLU", help="Activation on predicted scene probabilities" )
    arg_parser.add_argument('--executor-s', type=float, default=1, help="Saling of scene probabilities")
    arg_parser.add_argument('--executor-cos2pmf-act', type=str, default="Identity", help="Function for mapping similarities to probability")


    args = arg_parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    # Number of positions depends on the constellation
    args.num_pos = NUMPOS[args.config]
    
    # Use a rng for reproducible results   
    rng = np.random.default_rng(seed=args.seed)

    # Load or define new codebooks
    perception_cb, perception_imdict, backend_cb_cont, backend_cb_discrete = generate_nvsa_codebooks(args,rng)

    # backend for training/testing
    env = env_nvsa_ext.get_env(args.config,device, vsa_cb_discrete=backend_cb_discrete, vsa_cb_cont = backend_cb_cont,
                    detector_m=args.detector_m, detector_s = 1, detector_act=args.detector_act, executor_m=0, 
                    executor_s= args.executor_s, executor_act=args.executor_act,executor_cos2pmf_act=args.executor_cos2pmf_act)
    if args.mode == "train":
        # Define the training folders
        args.exp_dir = args.exp_dir+"{:}_pret_{:}_bs_{:}_lr_{:}_wd_{:}_s_{:}_ts_{:}_m_{:}_clip_{:}_main_loss_{:}_aux_{:}_d_{:}_k_{:}_exe_s_{:}_exe_cos2pmf_{:}_det_act{:}_det_m{:.3f}{:}{:}/{:}/".format(
                                                    args.config, args.pretrained, args.batch_size,args.lr, args.weight_decay, args.s, args.trainable_s, args.m,args.clip, args.main_loss, args.aux,args.nvsa_backend_d,args.nvsa_backend_k,
                                                    args.executor_s, args.executor_cos2pmf_act,args.detector_act, args.detector_m,args.gen_attribute, args.gen_rule,args.run)
        args.checkpoint_dir = args.exp_dir+"ckpt/"
        args.save_dir = args.exp_dir+"save/"
        args.log_dir = args.exp_dir+"log/"
        check_paths(args)
        save_codebooks(args,perception_cb, perception_imdict, backend_cb_cont, backend_cb_discrete)
        
        # Run the actual training 
        writer = train(args, env, perception_cb, device)

        # do final testing
        args.resume = args.checkpoint_dir
        writer = test(args,env,device,perception_cb,writer,"RAVEN")
        # do final testing of I-RAVEN if wanted
        if args.dataset_i_raven != "":
            args.dataset = args.dataset_i_raven
            writer = test(args,env,device,perception_cb,writer,"I-RAVEN")
        writer.close()
    elif args.mode == "test":
        # Test on RAVEN
        test(args, env, device, perception_cb, dset="RAVEN")
        # Test on I-RAVEN if set
        if args.dataset_i_raven != "":
            args.dataset = args.dataset_i_raven
            test(args,env, device, perception_cb, dset="I-RAVEN")

if __name__ == "__main__":
    main()
