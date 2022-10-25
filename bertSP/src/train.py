#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
from others.logging import init_logger
from train_baseline import validate_abs, train_abs, test_abs #, test_text_abs

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-test_split", default='test', help='Used for test (i.e., inference) mode. Expect valid or test.')
    parser.add_argument("-data_path", default='dataset/spice')
    parser.add_argument("-bert_data_path", default='dataset/spice')
    parser.add_argument("-model_path", default='models/baseline')
    parser.add_argument("-result_path", default='results/baseline')
    parser.add_argument("-tgt_dict", default='')
    parser.add_argument("-temp_dir", default='temp')

    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)

    # These two are not in use
    parser.add_argument("-input_syntax_voc", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-predict_qtype", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    #parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    parser.add_argument("-label_smoothing", default=0.0, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=1, type=float)
    parser.add_argument("-beam_size", default=1, type=int)
    parser.add_argument("-min_length", default=10, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=400, type=int)
    parser.add_argument("-dosubset", default='', type=str, help='used to select a subset of test/valid '
                                                                'files to run on. Should be a regular expression to '
                                                                'cover the desired shard numbers., e.g. 2[0-9] for'
                                                                'all 20, 21, ..., 29 shard files.')

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)


    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='logs/log.log')
    parser.add_argument('-seed', default=222, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='', type=str)
    parser.add_argument("-test_start_from", default=-1, type=int)
    parser.add_argument("-valid_from", default='', type=str)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-ban_unk_token", type=str2bool, nargs='?', const=True, default=False)


    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if (args.mode == 'train'):
        train_abs(args, device_id)
    elif (args.mode == 'validate'):
        validate_abs(args, device_id)
    elif (args.mode == 'test'):
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
        test_abs(args, device_id, cp, step)