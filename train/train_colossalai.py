import argparse
from email.policy import default
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
# from torch import distributed as dist
from torch.cuda.amp import autocast
from tqdm import tqdm, trange
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"  
from transformers import (AutoConfig,
                          get_linear_schedule_with_warmup,
                          AdamW)
from model_scripts.albert_models import AlbertForClozeTest
from model_scripts.bert_models import BertForClozeTest
from model_scripts.roberta_models import RobertaForClozeTest
from tools.adv_tools import FGM
from tools.data_tools import ELEDataset
from tools.path_tools import get_files
from tools.eda_tools import gen_eda_data
from preprocess_v2 import  multi_preprocess_data, Sample
from config import ROOT, SUPER_ROOT

import colossalai
import torch
import torch.nn as nn
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import CPUAdam
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import TensorShardStrategy
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
from transformers import GPT2Config, GPT2LMHeadModel

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, setType='train', kFold_num=None, kFold_idx=None):
    # if args.local_rank not in [-1, 0] and setType == 'train':
    #     dist.barrier()  # 确保多卡情况下只有第一张卡读取数据

    # preprocess raw data if pt data not prepared
    if 'albert' in args.model_name_or_path:
        model_name = 'albert'
    elif 'roberta' in args.model_name_or_path:
        model_name = 'roberta'
    elif 'bert' in args.model_name_or_path:
        model_name = 'bert'

    # gen eda data if need to use and not already done
    if setType=='train' and bool(args.use_eda_data) and not os.path.exists(ROOT+"/data/pt/train_eda-{}.pt".format(model_name)) and not os.path.exists(ROOT+"/data/raw/train_eda/"):
        gen_eda_data(ROOT+"/data/raw/train")
    if setType == 'train' and bool(args.use_eda_data):
        setType = 'train_eda'
    # preprocess data if not already done
    args.data_dir = ROOT+'/data/raw/{}'.format(setType)
    args.data_out_path = ROOT+'/data/pt/{}-{}.pt'.format(setType, model_name)

    if not os.path.exists(args.data_out_path):
        multi_preprocess_data(args)
        
    # load data
    dataset = ELEDataset(ROOT+'/data/pt/{}-{}.pt'.format(setType, model_name), kFold_num=kFold_num, kFold_idx=kFold_idx, debug=args.debug)
    
    return dataset

def train(args, model, kFold_num, kFold_idx):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = load_and_cache_examples(args, 'train', kFold_num=kFold_num, kFold_idx=kFold_idx)
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    zero = dict(
    model_config=dict(
        shard_strategy=TensorShardStrategy(),
        reduce_scatter_bucket_size_mb=25,
        fp32_reduce_scatter=False,
        tensor_placement_policy="cuda",
        gradient_predivide_factor=1.0,
        reuse_fp16_shard=False
    ),
    optimizer_config=dict(
        gpu_margin_mem_ratio=0.8,
        initial_scale=2**5,
        min_scale=1,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=1000,
        hysteresis=2,
        max_scale=2**32
    )
)
    colossalai.launch_from_torch(config=zero)
    shard_strategy = TensorShardStrategy()
    
    with ZeroInitContext(target_device=args.local_rank, shard_strategy=shard_strategy, shard_param=True):
        model = AlbertForClozeTest(args.model_name_or_path)

    
    model = ShardedModelV2(model, shard_strategy, tensor_placement_policy='cpu', reuse_fp16_shard=True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) //args.gradient_accumulation_steps * args.num_train_epochs

    eval_dataset = None
    if args.do_eval_during_train:
        eval_dataset = load_and_cache_examples(args, 'dev')

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    no_decay_params_id = [id(p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    decay_params_id = [id(p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {'params': [p for p in model.parameters() if id(p) in decay_params_id], 'weight_decay': args.weight_decay},
        {'params': [p for p in model.parameters() if id(p) in no_decay_params_id], 'weight_decay': 0.0}
    ]

    args.warmup_steps = int(t_total * args.warmup_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = ShardedOptimizerV2(model, optimizer, initial_scale=2**5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16))
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
    #                                                       output_device=args.local_rank,)
    # Train!
    logger.info("***** 训练阶段 *****")
    logger.info("  训练样本数 = %d", len(train_dataset))
    logger.info("  训练轮次 = %d", args.num_train_epochs)
    logger.info("  每GPU训练批次大小 = %d", args.per_gpu_train_batch_size)
    logger.info("  总训练批次大小 (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps  )
    logger.info("  梯度累积步数 = %d", args.gradient_accumulation_steps)
    logger.info("  总训练步数 = %d", t_total)

    fgm = None
    if args.adv_type == 'fgm':
        fgm = FGM(model)
    current_acc = 0.01
    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    best_acc = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    skip_scheduler = False
    for _ in train_iterator: # 遍历每个训练轮次
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator): # 遍历训练集
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            
            with autocast(enabled=bool(args.fp16)):
                article, option, answer, article_mask, option_mask, mask, blank_pos, sample_name = batch
                batch_size, option_num, out = model(batch)
            
                target = answer.view(-1, )
                # calculate loss
                loss = loss_func(out, target)
                loss = loss.view(batch_size, option_num) * mask
                # replace nan to 0
                loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss)
                loss = loss.sum() / (mask.sum() if not mask.sum() == 0 else 1)

                if args.n_gpu > 1:
                   loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                   loss = loss / args.gradient_accumulation_steps

                scale_before_step = scaler.get_scale()
                scaler.scale(loss).backward()

                if args.adv_type == 'fgm':
                    fgm.attack()  # 对抗训练
                    out = model(batch)[2]
                    adv_loss = loss_func(out, target)
                    adv_loss = adv_loss.view(batch_size, option_num) * mask
                    adv_loss = torch.where(torch.isnan(adv_loss), torch.full_like(adv_loss, 0), adv_loss)
                    adv_loss = adv_loss.sum() / (mask.sum() if not mask.sum() == 0 else 1)
                    adv_loss.backward()
                    fgm.restore()

            tr_loss += loss.item()
            epoch_iterator.set_description("loss {}".format(round(loss.item()*args.gradient_accumulation_steps, 4)))
            if (step + 1) % args.gradient_accumulation_steps == 0: # 每满梯度累积步数则进行一次更新
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                if bool(args.fp16):
                    skip_scheduler = scaler.get_scale() != scale_before_step
                if not skip_scheduler:
                    scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # 训练中验证
                if True and args.logging_steps > 0 and args.do_eval_during_train and (
                        global_step % args.logging_steps == 0 or (global_step + 1) == t_total) and eval_dataset:
                    eval_result = evaluate(args, model, eval_dataset)
                    output_dir = args.output_dir
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    current_acc = eval_result['eval_accuracy']

                    logger.info("  最佳准确率 : {}".format(best_acc))
                    logger.info("  当前准确率 : {}".format(current_acc))
                    logger.info("  当前步数 : {}".format(global_step))
                    logger.info("  ")
                    for k in eval_result.keys():
                        logger.info("  {} : {}".format(k, eval_result[k]))
                    if current_acc > best_acc:
                        best_acc = current_acc
                        if not os.path.exists(args.output_dir) and True:
                            os.makedirs(args.output_dir)
                        logger.info("保存模型到： %s", args.output_dir)
                        model.save_pretrained(args.output_dir)
                        output_eval_file = os.path.join(args.output_dir, "eval_results_during_train.txt")
                        with open(output_eval_file, "w") as writer:
                            for k, v in eval_result.items():
                                writer.write("{} : {}\n".format(k, v))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    return current_acc

def evaluate(args, model, eval_dataset=None):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if not eval_dataset:
        eval_dataset = load_and_cache_examples(args, setType='dev')
    if not eval_dataset:
        raise ValueError('验证和测试数据集不能为空')

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    # multi-gpu eval
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** 验证/测试阶段 *****")
    logger.info("  样本数 = %d", len(eval_dataset))
    logger.info("  批次大小 = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    val_acc_num = 0
    val_que_num = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        article, option, answer, article_mask, option_mask, mask, blank_pos, sample_name = batch
    
        target = answer.view(-1, )
        with torch.no_grad():
            batch_size, option_num, out = model(batch)
            # calculate loss
            loss = loss_func(out, target)
            loss = loss.view(batch_size, option_num) * mask
            # replace nan to 0
            loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss)
            loss = loss.sum() / (mask.sum() if not mask.sum() == 0 else 1)

            eval_loss += loss.mean().item()
        nb_eval_steps += 1
        # accuracy number
        acc_num = (torch.argmax(out, -1) == target).float()
        acc_num = acc_num.view(batch_size, option_num) * mask
        acc_num = acc_num.sum(-1)
        val_acc_num += acc_num.sum().item()
        val_que_num += mask.sum().item()
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = val_acc_num/val_que_num * 100
    result = {'eval_loss': round(eval_loss,4), 'eval_accuracy': round(eval_accuracy,6)}
    return result


def main(args):

    

    # args.output_dir = os.path.join(args.output_dir, args.task_name)
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "训练结果输出目录 ({}) 已存在且不为空。请使用 --overwrite_output_dir命令进行覆盖。".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if False or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda",args.local_rank)
        # dist.init_process_group(backend="nccl")
        args.n_gpu = 1
        
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if True else logging.WARN,
    )
    logger.info(
        "当前节点号: %s, 使用设备: %s, GPU数目: %s, 是否采用分布式训练: %s, 是否采用半精度训练: %s",
        0,
        device,
        args.n_gpu,
        bool(True),
        bool(args.fp16),
    )
    if args.debug:
        logger.info("当前为debug模式")
    # Set seed
    set_seed(args)

    # Load pretrained model
    # if args.local_rank not in [-1, 0]:
    #     dist.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    output_dir_root = args.output_dir
    for kFold_idx in range(args.k_folds_num):
        if True:
            logger.info("***** 训练{}/{}折 *****".format(kFold_idx+1, args.k_folds_num))
        args.output_dir = output_dir_root + "/fold{}".format(kFold_idx)
        if args.do_train:
            if "albert" in args.model_name_or_path:
                model = AlbertForClozeTest(args.model_name_or_path)
            elif "roberta" in args.model_name_or_path:
                model = RobertaForClozeTest(args.model_name_or_path)
            elif "bert" in args.model_name_or_path:
                model = BertForClozeTest(args.model_name_or_path)
            else:
                raise ValueError("model not supported")

            # if args.local_rank == 0:
            #     dist.barrier()  # Make sure only the first process in distributed training will download model & vocab
            # args.device = torch.device("cuda",torch.cuda.current_device())
            model.to(args.device)
            current_acc = train(args, model, args.k_folds_num, kFold_idx)
            if not os.path.exists(args.output_dir) and True:
                os.makedirs(args.output_dir)

            # Save the trained model and the tokenizer
            if (True ) and (
                    not args.do_eval_during_train):
                logger.info("保存模型到： %s", args.output_dir)
                model.save_pretrained(args.output_dir)

        if args.do_eval and True:
            if "albert" in args.output_dir:
                model = AlbertForClozeTest(args.output_dir)
            elif "roberta" in args.output_dir:
                model = RobertaForClozeTest(args.output_dir)
            elif "bert" in args.output_dir:
                model = BertForClozeTest(args.output_dir)
            else:
                raise ValueError("model not supported")
            # args.device = torch.device("cuda",torch.cuda.current_device())
            model.to(args.device)
            result = evaluate(args, model)
            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** 测试结果 *****")
                for k, v in result.items():
                    logger.info("  {} : {}".format(k, v))
                    writer.write("{} : {}\n".format(k, v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    # parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')



    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='随机数种子')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0)
    parser.add_argument('--task_name',
                        type=str,
                        default='ClozeTest',
                        help='任务类型')
    parser.add_argument('--adv_type',
                        default='fgm',
                        type=str,
                        choices=['fgm', 'none'])
    parser.add_argument('--debug',default=True, action='store_true')

    parser.add_argument(
        "--model_name_or_path",
        default=ROOT+'/pretrained_models/albert-xxlarge-v2/',
        type=str,
        help="Path to pre-trained model ",
    )
    parser.add_argument(
        "--output_dir",
        default=SUPER_ROOT+'/model/',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument("--do_train",default=True,action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_during_train", action="store_true", help="Run evaluation during training at each logging step.",)
    parser.add_argument("--use_eda_data", action="store_true", help="using eda extra data",)
    parser.add_argument("--k_folds_num", type=int, default=5, help="k forlds trainning")

    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int, help="Batch size per GPU/CPU for training.",)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,default=16,help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
        default=-1,type=int,help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_rate", default=0.1, type=int,help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")

    parser.add_argument("--no_cuda", type=bool, default=False, help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", type=bool, default=True, help="Overwrite the content of the output directory",)

    parser.add_argument("--fp16",
	    type=int,default=1,help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    
    args = parser.parse_args()

    

    if args.debug:
        args.num_train_epochs = 1
    
    main(args)
