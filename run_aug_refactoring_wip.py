import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from utils import set_seed, collate_fn, collate_fn2, tsne, analyze_result
from datasets import load_metric
from model_refactoring_wip import *
from torch.utils.data import TensorDataset, Dataset, DataLoader
from dataset import PairDataset
from evaluation import *
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
import wandb
import pickle
import json
import csv
import hydra
from omegaconf import DictConfig
import warnings
from data import load, LACL_load, LACL_load_aug, write_aug_data
import transformers
from loss import *
import pdb
import os
from torch.optim import SGD, Adam, AdamW
warnings.filterwarnings("ignore")


task_to_labels = {
    'sst2': 2,
    'imdb': 2,
    '20ng': 20,
    'trec': 6,
    'clinc150':150,
    'banking77':77,
    'snips':7,
}


task_to_metric = {
    'sst2': 'sst2',
    'imdb': 'sst2',
    '20ng': 'mnli',
    'trec': 'mnli',
}


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(args, model, tokenizer):

    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.eps)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    # --------------------------------------------------------------- #
    if args.classifier_type == 'softmax':
        if args.label_smoothing == True:
            # if args.task_name == 'snips'
            #     args.smoothing_ratio = args.smoothing_ratio / 5
            criterion = LabelSmoothLoss(args.smoothing_ratio)
        else:
            criterion = nn.CrossEntropyLoss()
    elif args.classifier_type == 'sigmoid':
        criterion = nn.BCEWithLogitsLoss()
    # --------------------------------------------------------------- #
    

    # EARLY STOP 
    EARLY_STOPPING,stop_cnt = 10,0

    # Log, CKPT
    # performance ={'best_auroc_acc':0,'best_maha_auroc_acc':0,'best_auroc_result':None,'best_auroc_epoch':None,'best_maha_auroc_result':None,'best_maha_auroc_result':None}
    best_auroc_acc, best_maha_auroc_acc=0, 0
    best_auroc_result,best_maha_auroc_result,best_maha_auroc_result, best_auroc_epoch=  None,None,None,None
    
    for epoch in range(int(args.num_train_epochs)):
        # EARLY STOP 
        stop_cnt += 1
        
        # dataset load & log
        benchmarks = ()
        train_aug1_dataset, train_aug2_dataset, test_ind_dataset, test_ood_dataset, train_raw_dataset = LACL_load_aug(args, tokenizer)
        benchmarks = (('ood_'+ args.task_name, test_ood_dataset),) + benchmarks
        comb_dset = PairDataset(train_aug1_dataset, train_aug2_dataset)
        comb_train_dataloader = DataLoader(comb_dset, shuffle=True, drop_last=True, collate_fn=collate_fn2, batch_size=args.train_batch, num_workers=args.num_workers)
        # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)

        # train_dataloader = DataLoader(train_raw_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
        train_dataloader = DataLoader(train_raw_dataset, batch_size=100, collate_fn=collate_fn, shuffle=True, drop_last=True)

        # scheduler
        total_steps = int(len(comb_train_dataloader) * args.num_train_epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max = total_steps, eta_min = args.learning_rate * 1e-3)
        model.zero_grad()
        
        # initialize
        train_bar = tqdm(comb_train_dataloader)
        total_cl1_loss= total_cl2_loss = total_ce_loss = total_reg_loss = total_proxy_loss = 0


        # reset proxy
        # model.reset_proxy(train_dataloader)
        # proxy_anchor_loss = ProxyAnchorLoss()


        # train code
        for idx, (batch1,batch2) in enumerate(train_bar):
            
            
            # batch initialize
            model.train()
            batch1 = {key: value.to(DEVICE) for key, value in batch1.items()}
            batch2 = {key: value.to(DEVICE) for key, value in batch2.items()}
            if args.classifier_type == 'sigmoid':
                intent_labels = smoothed_labels(intent_labels, args.n_classes, smoothing = args.smoothing_ratio) if args.label_smoothing == True else smoothed_labels(intent_labels, args.n_classes)
            
            # Feed batch into model
            b1_dict = model(**batch1)
            b2_dict = model(**batch2)
            

            total_loss = 0

            # Cross-Entropy loss
            if args.loss_ce==True:
                loss_ce = criterion(b1_dict['logits'], batch1['labels']) 
                # loss_ce += criterion(b2_dict['logits'], batch1['labels'])
                
                total_ce_loss += loss_ce.item()
                total_loss += loss_ce


            # Contrastive Learning loss (GP)
            label = batch1['labels'].cuda()
            if args.cl1 == True:
                if args.cl1_scl ==True:
                    cont = torch.cat([b1_dict['global_projection'],b2_dict['global_projection']], dim=0)
                    sim_mat = get_sim_mat(cont)
                    loss_cl_gp = Supervised_NT_xent(sim_mat, labels=label, temperature=args.temperature)
                    total_cl1_loss += loss_cl_gp.item()
                    total_loss += loss_cl_gp * args.cl_weight
                else:
                    cont = torch.cat([b1_dict['global_projection'],b2_dict['global_projection']], dim=-1).view(-1, b1_dict['global_projection'].shape[-1])
                    loss_cl_gp = nt_xent(cont, args.temperature)
                    total_cl1_loss += loss_cl_gp.item()
                    total_loss += loss_cl_gp * args.cl_weight

            # Contrastive Learning loss (Last CLS)
            if args.cl2 == True:
                if args.cl2_scl ==True:
                    cont2 = torch.cat([b1_dict['cls_projection'],b2_dict['cls_projection']], dim=0)
                    sim_mat2 = get_sim_mat(cont2)
                    loss_cl_cls = Supervised_NT_xent(sim_mat2, labels=label, temperature=args.temperature)
                    total_cl2_loss += loss_cl_cls.item()
                    total_loss += loss_cl_cls * args.cl_weight
                else:
                    cont2 = torch.cat([b1_dict['cls_projection'],b2_dict['cls_projection']], dim=-1).view(-1, b1_dict['cls_projection'].shape[-1])
                    loss_cl_cls = nt_xent(cont2, args.temperature)
                    total_cl2_loss += loss_cl_cls.item()
                    total_loss += loss_cl_cls * args.cl_weight
            
            
            # Correlation Regularization loss
            if args.reg_loss == True:
                tok_embeddings = [b1_dict['lw_mean_embedding'], b2_dict['lw_mean_embedding']]
                loss_reg = reg_loss(tok_embeddings,args)
                loss_cr = loss_reg
                total_reg_loss += loss_reg.item()
                total_loss += loss_cr * args.reg_loss_weight

            # Train_bar description
            train_bar.set_description(f'Train epoch {epoch}, CE Loss: {(total_ce_loss)/(idx+1):.2f}, GP_CL Loss: {(total_cl1_loss)/(idx+1):.2f}, CLS_CL Loss: {(total_cl2_loss)/(idx+1):.2f}, Reg Loss {(total_reg_loss)/(idx+1):.2f}:')

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()


        # Test accuracy evaluation
        test_acc = evaluate_acc(args, model, test_ind_dataset, tag="test") if args.loss_ce==True else 0

        if epoch > 0 and epoch % args.log_interval == 0:
            # OOD performance evaluation
            if args.final_rep == 'all':
                if args.cl1 == True:
                    keys = [1,2,3,4,5,6,7,8,9,10,11,12,'global_projection']
                    # keys = [1,2,3,4,5,6,7,8,9,10,11,12,'global_projection']
                elif args.cl2 == True:
                    keys = [1,2,3,4,5,6,7,8,9,10,11,12,'cls_projection']
                    # keys = [12,'cls_projection']
                else:
                    keys = [1,2,3,4,5,6,7,8,9,10,11,12]
                    # keys = [12]

                correct_90_index_list, wrong_90_index_list = [],[]
                correct_95_index_list, wrong_95_index_list = [],[]
                cosine_score_list, maha_score_list, maha_label_list = [], [], []
                

                for key in keys:
                    model.prepare_ood(key, train_dataloader)
                    for tag, ood_features in benchmarks:
                        base_results, a, b, c, d, maha_s, cosine_s, maha_l = evaluate_ood(args, model, test_ind_dataset, ood_features, tag, final_rep = key)
                        
                        # cosine maha layer-wise eval
                        maha_score_list.append(maha_s)
                        cosine_score_list.append(cosine_s)
                        maha_label_list.append(maha_l)


                        print(base_results)
                        correct_90_index_list.append(a)
                        wrong_90_index_list.append(b)
                        correct_95_index_list.append(c)
                        wrong_95_index_list.append(d)

                    write_log2(args, args.model_name, epoch, test_acc, base_results)
                
                # maha layer-wise eval
                sum_maha_score = sum(maha_score_list[:12])
                maha_label = maha_label_list[0]
                maha_sum_auroc = get_auroc(maha_label, sum_maha_score)
                maha_sum_fpr_90 = get_fpr_95(maha_label, sum_maha_score, recall_level=0.9, return_indices=False)
                maha_sum_fpr_95 = get_fpr_95(maha_label, sum_maha_score, recall_level=0.95, return_indices=False)

                # cosine layer-wise eval (cls)
                sum_cosine_score = sum(cosine_score_list[:12])
                maha_label = maha_label_list[0]
                cosine1_sum_auroc = get_auroc(maha_label, sum_cosine_score)
                cosine1_sum_fpr_90 = get_fpr_95(maha_label, sum_cosine_score, recall_level=0.9, return_indices=False)
                cosine1_sum_fpr_95 = get_fpr_95(maha_label, sum_cosine_score, recall_level=0.95, return_indices=False)

                # cosine layer-wise eval (z)
                sum_cosine_score = sum(cosine_score_list)
                maha_label = maha_label_list[0]
                cosine2_sum_auroc = get_auroc(maha_label, sum_cosine_score)
                cosine2_sum_fpr_90 = get_fpr_95(maha_label, sum_cosine_score, recall_level=0.9, return_indices=False)
                cosine2_sum_fpr_95 = get_fpr_95(maha_label, sum_cosine_score, recall_level=0.95, return_indices=False)


                lw_maha = [maha_sum_auroc,maha_sum_fpr_90, maha_sum_fpr_95,cosine1_sum_auroc, cosine1_sum_fpr_90, cosine1_sum_fpr_95,cosine2_sum_auroc, cosine2_sum_fpr_90, cosine2_sum_fpr_95]


                if args.cl1==True or args. cl2 == True:
                    # IPD_list, last_cls_AIA_list, GP_AIA_list, last_cls_pcr_list = cal_new_metric1(correct_index_list, wrong_index_list, GP = True)
                    IPD_list, last_cls_AIA_list, GP_AIA_list, last_cls_pcr_list = cal_new_metric2(correct_90_index_list, wrong_90_index_list, GP = True)
                    write_log3(args, args.model_name, epoch, test_acc, IPD_list, last_cls_AIA_list, lw_maha, last_cls_pcr_list, GP_AIA = GP_AIA_list, precision = 0.9)
                    IPD_list, last_cls_AIA_list, GP_AIA_list, last_cls_pcr_list = cal_new_metric2(correct_95_index_list, wrong_95_index_list, GP = True)
                    write_log3(args, args.model_name, epoch, test_acc, IPD_list, last_cls_AIA_list, lw_maha, last_cls_pcr_list, GP_AIA = GP_AIA_list, precision = 0.95)
                else:
                    # IPD_list, last_cls_AIA_list, last_cls_pcr_list = cal_new_metric1(correct_index_list, wrong_index_list, GP = False)
                    IPD_list, last_cls_AIA_list, last_cls_pcr_list = cal_new_metric2(correct_90_index_list, wrong_90_index_list, GP = False)
                    write_log3(args, args.model_name, epoch, test_acc, IPD_list, last_cls_AIA_list, lw_maha, last_cls_pcr_list,  precision = 0.9)
                    IPD_list, last_cls_AIA_list, last_cls_pcr_list = cal_new_metric2(correct_95_index_list, wrong_95_index_list, GP = False)
                    write_log3(args, args.model_name, epoch, test_acc, IPD_list, last_cls_AIA_list, lw_maha, last_cls_pcr_list, precision = 0.95)
                    
            else:
                model.prepare_ood(args.final_rep, train_dataloader)
                for tag, ood_features in benchmarks:
                    base_results = evaluate_ood(args, model, test_ind_dataset, ood_features, tag, final_rep=args.final_rep)
                    print(base_results)
                
                if best_auroc_result == None:
                        best_auroc_result = base_results
                        best_auroc_epoch = epoch
                        best_auroc_acc = test_acc
                else:
                    for k, v in best_auroc_result.items():
                        if k.endswith('softmax_auroc') and v < base_results[k]:
                            best_auroc_result = base_results
                            best_auroc_epoch = epoch
                            best_auroc_acc = test_acc
                            stop_cnt = 0
                if best_maha_auroc_result == None:
                    best_maha_auroc_result = base_results
                    best_maha_auroc_epoch = epoch
                    best_maha_auroc_acc = test_acc
                else:
                    for k, v in best_maha_auroc_result.items():
                        if k.endswith('maha_auroc') and v < base_results[k]:
                            best_maha_auroc_result = base_results
                            best_maha_auroc_epoch = epoch
                            best_maha_auroc_acc = test_acc
                            stop_cnt = 0

                    write_log2(args, args.model_name, epoch, test_acc, base_results)
                    
                    # if EARLY_STOPPING <= stop_cnt:
                    #     break

def write_log3(args, model_name, epoch, test_acc, IPD, cls_AIA, lw_maha, last_cls_pcr_list, GP_AIA=None, precision = 0.95):
    if args.split:
        dir = os.path.join(args.log_dir, f"{args.task_name}-{args.split_ratio}-seed{args.seed}", model_name, 'eval_result')
    else:
        dir = os.path.join(args.log_dir,  f"{args.task_name}-no_split-seed{args.seed}", model_name, 'eval_result')
    
    os.makedirs(os.path.join(dir), exist_ok = True)
    log_fname = f'Diversity_{precision}.csv'
    with open(os.path.join(dir,log_fname), 'a') as f:
        for i in IPD:
            f.write(f'{i*100:.3f},')
        f.write('\n')

    log_fname = f'cls_ensemble_{precision}.csv'
    with open(os.path.join(dir,log_fname), 'a') as f:
        for i in cls_AIA:
            f.write(f'{i*100:.3f},')
        f.write('\n')

    log_fname = f'explicit_ensemble_{precision}.csv'
    with open(os.path.join(dir,log_fname), 'a') as f:
        for i in lw_maha:
            f.write(f'{i*100:.3f},')
        f.write('\n')
    
    log_fname = f'cls_pcr_{precision}.csv'
    with open(os.path.join(dir,log_fname), 'a') as f:
        for i in last_cls_pcr_list:
            f.write(f'{i*100:.3f},')
        f.write('\n')

    if GP_AIA != None:
        log_fname = f'GP_ensemble_{precision}.csv'
        with open(os.path.join(dir,log_fname), 'a') as f:
            for i in GP_AIA:
                f.write(f'{i*100:.3f},')
            f.write('\n')

def cal_new_metric1(correct_lists, wrong_lists, GP = True):

    num_layers = 12
   
    # IPD rate (diversity)
    IPD_list = []
    for i in range(num_layers - 1):
        query_correct_list, query_wrong_list = set(correct_lists[i]), wrong_lists[i]
        next_correct_list, next_wrong_list = set(correct_lists[i+1]), wrong_lists[i+1]

        pos_complementary = len(next_correct_list.intersection(query_wrong_list))
        neg_complementary = len(query_correct_list.intersection(next_wrong_list))
        
        IPD_list.append((pos_complementary+neg_complementary)/(len(correct_lists[i])+len(wrong_lists[i])))
    
    IPD_list.append(sum(IPD_list)/len(IPD_list))


    # CLS AIA (Ensemble absorbption)
    last_cls_AIA_list = []
    last_cls_pcr_list = []
    last_cls_correct = set(correct_lists[11])
    last_cls_wrong = wrong_lists[11]
    for i in range(num_layers-1):
        query_correct_list, query_wrong_list = set(correct_lists[i]), wrong_lists[i]

        pos_complementary = len(query_correct_list.intersection(last_cls_wrong))
        neg_complementary = len(last_cls_correct.intersection(query_wrong_list))

        # last_cls_pcr_list.append((pos_complementary+neg_complementary)/(len(query_correct_list)+len(query_wrong_list)))
        last_cls_pcr_list.append((pos_complementary)/(len(last_cls_wrong)))

        last_cls_AIA_list.append(neg_complementary/(pos_complementary+neg_complementary))


    
    last_cls_AIA_list.append(sum(last_cls_AIA_list)/len(last_cls_AIA_list))

    # GP AIA (Ensemble absorbption)
    if GP == True:
        GP_correct = set(correct_lists[-1])
        GP_wrong = wrong_lists[-1]
        GP_AIA_list = []

        for i in range(num_layers):
            query_correct_list, query_wrong_list = set(correct_lists[i]), wrong_lists[i]
            pos_complementary = len(query_correct_list.intersection(GP_wrong))
            neg_complementary = len(GP_correct.intersection(query_wrong_list))

            GP_AIA_list.append(neg_complementary/(pos_complementary+neg_complementary))
        
        GP_AIA_list.append(sum(GP_AIA_list)/len(GP_AIA_list))            

        return IPD_list, last_cls_AIA_list, GP_AIA_list, last_cls_pcr_list
    
    return IPD_list, last_cls_AIA_list, last_cls_pcr_list


def cal_new_metric2(correct_lists, wrong_lists, GP = True):

    num_layers = 12
   
    # IPD rate (diversity)
    IPD_list = []
    for i in range(num_layers - 1):
        query_correct_list, query_wrong_list = set(correct_lists[i]), wrong_lists[i]
        next_correct_list, next_wrong_list = set(correct_lists[i+1]), wrong_lists[i+1]

        pos_complementary = len(next_correct_list.intersection(query_wrong_list))
        neg_complementary = len(query_correct_list.intersection(next_wrong_list))
        
        IPD_list.append((pos_complementary+neg_complementary)/(len(correct_lists[i])+len(wrong_lists[i])))
    
    IPD_list.append(sum(IPD_list)/len(IPD_list))


    # CLS AIA (Ensemble absorbption)
    last_cls_AIA_list = []
    last_cls_pcr_list = []
    last_cls_correct = set(correct_lists[11])
    last_cls_wrong = wrong_lists[11]
    for i in range(num_layers-1):
        query_correct_list, query_wrong_list = set(correct_lists[i]), wrong_lists[i]

        pos_complementary = len(query_correct_list.intersection(last_cls_wrong))
        neg_complementary = len(last_cls_correct.intersection(query_wrong_list))

        # last_cls_pcr_list.append((pos_complementary+neg_complementary)/(len(query_correct_list)+len(query_wrong_list)))
        last_cls_pcr_list.append((pos_complementary)/(len(last_cls_wrong)))

        last_cls_AIA_list.append(neg_complementary/(pos_complementary+neg_complementary))


    
    last_cls_AIA_list.append(sum(last_cls_AIA_list)/len(last_cls_AIA_list))

    # GP AIA (Ensemble absorbption)
    if GP == True:
        GP_correct = set(correct_lists[-1])
        GP_wrong = wrong_lists[-1]
        GP_AIA_list = []

        for i in range(num_layers):
            query_correct_list, query_wrong_list = set(correct_lists[i]), wrong_lists[i]
            pos_complementary = len(query_correct_list.intersection(GP_wrong))
            neg_complementary = len(GP_correct.intersection(query_wrong_list))

            GP_AIA_list.append(neg_complementary/(pos_complementary+neg_complementary))
        
        GP_AIA_list.append(sum(GP_AIA_list)/len(GP_AIA_list))            

        return IPD_list, last_cls_AIA_list, GP_AIA_list, last_cls_pcr_list
    
    return IPD_list, last_cls_AIA_list, last_cls_pcr_list



# def cal_pcr(correct_lists, wrong_lists):  

#     anchor_correct_list = correct_lists[-1]
#     anchor_wrong_list = wrong_lists[-1]

#     anchor_correct = set(anchor_correct_list)
#     anchor_wrong = set(anchor_wrong_list)

#     pcr_list = []
#     pccr_list = []
#     for idx in range(len(wrong_lists)-1):
#         target_wrong = wrong_lists[idx]
#         target_correct = correct_lists[idx]

#         pcr_list.append(len(anchor_wrong.intersection(target_correct)) / len(anchor_wrong))
#         pccr_list.append((len(anchor_correct.intersection(target_correct))+len(anchor_wrong.intersection(target_wrong)))/(len(anchor_wrong)+len(anchor_correct)))
    
#     print(pcr_list)
#     print(pccr_list)
    
#     return pcr_list, pccr_list



def write_log2(args, model_name, epoch, test_acc, dic):
    if args.split:
        dir = os.path.join(args.log_dir, f"{args.task_name}-{args.split_ratio}-seed{args.seed}", model_name, 'eval_result')
    else:
        dir = os.path.join(args.log_dir,  f"{args.task_name}-no_split-seed{args.seed}", model_name, 'eval_result')
    
    os.makedirs(os.path.join(dir), exist_ok = True)
    log_fname = 'evaluation_log.csv'
    with open(os.path.join(dir,log_fname), 'a') as f:
        f.write(f'{epoch},{test_acc:.3f},')
        for key in dic.keys():
            f.write(f'{dic[key]*100:.3f},')
        f.write('\n')

@hydra.main(config_path=".", config_name="model_config.yaml")
def main(args: DictConfig) -> None:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.device_count())
    args.n_gpu = torch.cuda.device_count()
    DEVICE = device
    dataset = args.task_name
    set_seed(args)

    if args.task_name == 'clinc150':
        if args.split_ratio == 0:
            num_labels = 150
        else:
            num_labels = round(15*args.split_ratio) * 10
    elif args.task_name == 'banking77':
        # assert args.split == False
        num_labels = round(77*args.split_ratio)
    elif args.task_name == 'snips':
        # assert args.split == False
        num_labels = round(7*args.split_ratio)
        # num_labels = 7
    elif args.task_name.startswith('mix'):
        mix_task_name = args.task_name[4:]
        if mix_task_name == 'snips':
            num_labels = 150 - 7
        if mix_task_name == 'banking' or  mix_task_name == 'banking77':
            mix_task_name = 'banking'
            num_labels = 150 - 30

    if args.model_name_or_path.startswith('roberta'):
        config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.gradient_checkpointing = True
        config.alpha = args.alpha
        config.loss = args.loss
        
        # !!
        config.output_attentions = True
        config.output_hidden_states = True
        if args.no_dropout:
            config.attention_probs_dropout_prob = 0.0
            config.hidden_dropout_prob = 0.0
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        # model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, last_cls = args.last_cls)
        model = LaCL_roberta.from_pretrained(args.model_name_or_path, config=config, args=args)
        model.to(0)
    elif args.model_name_or_path.startswith('bert'):
        config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.gradient_checkpointing = True
        config.alpha = args.alpha
        config.loss = args.loss
        if args.no_dropout:
            config.attention_probs_dropout_prob = 0.0
            config.hidden_dropout_prob = 0.0
        # !!
        config.output_attentions = True
        config.output_hidden_states = True

        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

        # model = LACL_SA.from_pretrained(args.model_name_or_path, config=config,args=args)            
        # model = Baselines.from_pretrained(args.model_name_or_path, config=config,args=args)            
        model = LACL_fixed.from_pretrained(args.model_name_or_path, config=config,args=args)            
        model.to(0)

    print(args)
    train(args, model, tokenizer)


if __name__ == "__main__":
    main()

