import torch
import random
import pdb
import matplotlib
import csv
from tqdm import tqdm
matplotlib.use('agg')
from matplotlib import pyplot as plt  
import numpy as np
import os
import json
import torch.nn.functional as F
from statistics import pstdev
from typing import List
import hydra
from sklearn.manifold import TSNE


import os
import json

def write_log(args, epoch, test_acc, dic,f_name):
    dir = os.path.join('./logs',args.model_name)
    os.makedirs(os.path.join(dir), exist_ok = True)
    if os.path.exists(os.path.join(dir,f_name)) == False:
        write_meta = True
    else: 
        write_meta = False
    with open(os.path.join(dir,f_name), 'a') as f:
        if write_meta == True:
            f.write('epoch,test_acc,')
            for key in dic.keys():
                f.write(f'{key},')
            f.write('\n')
            f.write(f'{epoch},{test_acc:.3f},')
            for key in dic.keys():
                f.write(f'{dic[key]*100:.3f},')
            f.write('\n')
        else:
            f.write(f'{epoch},{test_acc:.3f},')
            for key in dic.keys():
                f.write(f'{dic[key]*100:.3f},')
            f.write('\n')
            




def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    
    if 'indices' in batch[0]:
        indices = torch.LongTensor([f["indices"] for f in batch])
        outputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": labels,
            "indices": indices,
        }
    else:
        outputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": labels,
        }
    return outputs


def collate_fn2(batch):
    
    batch1 = [i[0] for i in batch]
    batch2 = [i[1] for i in batch]

    max_len1 = max([len(f["input_ids"]) for f in batch1])
    max_len2 = max([len(f["input_ids"]) for f in batch2])

    max_len = max(max_len1,max_len2)

    input_ids1 = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch1]
    input_ids2 = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch2]

    input_mask1 = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch1]
    input_mask2 = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch2]

    labels1 = [f["labels"] for f in batch1]
    labels2 = [f["labels"] for f in batch2]
    
    input_ids1 = torch.tensor(input_ids1, dtype=torch.long)
    input_ids2 = torch.tensor(input_ids2, dtype=torch.long)
    input_mask1 = torch.tensor(input_mask1, dtype=torch.float)
    input_mask2 = torch.tensor(input_mask2, dtype=torch.float)
    labels1 = torch.tensor(labels1, dtype=torch.long)
    labels2 = torch.tensor(labels2, dtype=torch.long)
    
    outputs1 = {
        "input_ids": input_ids1,
        "attention_mask": input_mask1,
        "labels": labels1,
    }
    
    outputs2 = {
        "input_ids": input_ids2,
        "attention_mask": input_mask2,
        "labels": labels2,
    }
    return [outputs1, outputs2]

def collate_fn_ref1(batch):
    # tensor shape [1, batch size] -> [batch size]
    # pdb.set_trace()
    max_len = max([len(f["input_ids"].squeeze()) for f in batch])
    input_ids = [f["input_ids"].squeeze().tolist() + [0] * (max_len - len(f["input_ids"].squeeze())) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"].squeeze()) + [0.0] * (max_len - len(f["input_ids"].squeeze())) for f in batch]
    labels = [f["labels"] for f in batch]
    indices = [f["indices"] for f in batch]
    input_ids = torch.LongTensor(input_ids)
    input_mask = torch.FloatTensor(input_mask)
    labels = torch.LongTensor(labels)
    indices = torch.LongTensor(indices)
    outputs = {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "labels": labels,
        "indices": indices,
    }
    return outputs

def log_pred_results(f: float,
                     acc: float,
                     f_seen: float,
                     acc_in: float,
                     p_seen: float,
                     r_seen: float,
                     f_unseen: float,
                     acc_ood: float,
                     p_unseen: float,
                     r_unseen: float,
                     classes: List[str],
                     output_dir: str,
                     confusion_matrix: np.ndarray,
                     auroc: float = None,
                     fpr95: float = None,
                     threshold: float = None):
    with open(os.path.join(output_dir, "results.txt"), "a") as f_out:
        f_out.write(
            f"Overall:  f1(macro):{f} acc:{acc} \nSeen:  f1(macro):{f_seen} acc:{acc_in} p:{p_seen} r:{r_seen}\n"
            f"=====> Unseen(Experiment) <=====:  f1(marco):{f_unseen} acc:{acc_ood} p:{p_unseen} r:{r_unseen}\n\n"
            f"Classes:\n{classes}\n\n"
            f"AUROC(OOD):\n{auroc}\n\n"
            f"FPR-95:\n{fpr95}\n\n"
            f"Threshold:\n{threshold}\n\n"
            f"Confusion matrix:\n{confusion_matrix}")
    with open(os.path.join(output_dir, "results.json"), "a") as f_out:
        json.dump({
            "f1_overall": f,
            "acc_overall": acc,
            "f1_seen": f_seen,
            "acc_seen": acc_in,
            "p_seen": p_seen,
            "r_seen": r_seen,
            "f1_unseen": f_unseen,
            "acc_unseen": acc_ood,
            "p_unseen": p_unseen,
            "r_unseen": r_unseen,
            "auroc": auroc,
            "fpr95": fpr95,
            "classes": classes,
            "confusion_matrix": confusion_matrix.tolist(),
            "threshold": threshold
        }, fp=f_out, ensure_ascii=False, indent=4)


def tsne_file_name(args, model_name):
    if args.split:
        tsne_folder_name = os.path.join(hydra.utils.to_absolute_path(args.save_dir), f"{args.task_name}-{args.split_ratio}-seed{args.seed}", model_name, 'tsne')
    else:
        tsne_folder_name = os.path.join(hydra.utils.to_absolute_path(args.save_dir), f"{args.task_name}-no_split-seed{args.seed}", model_name, 'tsne')
    os.makedirs(tsne_folder_name, exist_ok=True)
    
    f_name = os.path.join(tsne_folder_name, 'ind_features.txt')
    l_f_name = os.path.join(tsne_folder_name, 'ind_labels.txt')
    ood_f_name = os.path.join(tsne_folder_name, 'ood_features.txt')
    ood_l_f_name = os.path.join(tsne_folder_name, 'ood_labels.txt')
    
    return f_name, l_f_name, ood_f_name, ood_l_f_name, tsne_folder_name
def train_tsne(tsne_file_names, tsne_mode='intent', visualize_num = 1000):
    f_name, l_f_name, ood_f_name, ood_l_f_name, tsne_folder_name = tsne_file_names
    # if os.path.exists(os.path.join(tsne_folder_name, f'tsne_{tsne_mode}.png')):
    #     print('Delete previous t-SNE image')
    #     os.remove(os.path.join(tsne_folder_name, f'tsne_{tsne_mode}.png'))

    # t-SNE 모델 생성 및 학습
    rep = np.loadtxt(f_name, delimiter=',')[:visualize_num]
    label = np.loadtxt(l_f_name, delimiter=',')[:visualize_num]
    ood_rep = np.loadtxt(ood_f_name, delimiter=',')[:visualize_num // 2]
    ood_label = np.loadtxt(ood_l_f_name, delimiter=',')[:visualize_num // 2]    
    
    label = label.astype(np.int64)
    ood_label = ood_label.astype(np.int64)

    print(f'Training T-SNE, {tsne_mode} mode.')
    tsne = TSNE(random_state=0, n_iter=500)
    rep_tsne = tsne.fit_transform(rep)
    ood_rep_tsne = tsne.fit_transform(ood_rep)

    # colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525', '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E', '#080808']
    colors = ['#228b22','#00008b','#b03060','#004500','#ffff00','#deb887','#00ff00','#00ffff','#ff00ff','#6495ed']
    
    x = label
    for i in range(len(rep)): 
        plt.scatter(rep_tsne[i, 0], rep_tsne[i, 1], marker='o', color=colors[label[i] % 10], s=0.5, linewidths=0.5)
        # if label[i] == -1 :
        #     plt.text(rep_tsne[i, 0], rep_tsne[i, 1], str(x[i]), # x, y , 그룹
        #             color=colors[label[i]], # 색상
        #             fontdict={'weight': 'bold', 'size':9}) # font
        # else:
        #     plt.text(rep_tsne[i, 0], rep_tsne[i, 1], str(x[i]), # x, y , 그룹
        #             color=colors[label[i]%10], # 색상
        #             fontdict={'weight': 'normal', 'size':2}) # font
    x = [-1]*len(ood_label)
    print(f'Number of rep: {rep.shape}')
    print(f'Number of label: {label.shape}')
    print(f'Number of ood_rep: {ood_rep.shape}')
    print(f'Number of ood_label: {ood_label.shape}')
    for i in range(len(ood_rep)): 
        # OOD data
        plt.scatter(ood_rep_tsne[i, 0], ood_rep_tsne[i, 1], marker='x', color='red', s=3, linewidths=0.5)
        # plt.text(ood_rep_tsne[i, 0], ood_rep_tsne[i, 1], str(x[i]), # x, y , 그룹
        #         color='#ff0000', # 색상
        #         fontdict={'weight': 'bold', 'size':3}) # font
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.xlim(rep_tsne[:, 0].min()-5, rep_tsne[:, 0].max()+5) # 최소, 최대
    plt.ylim(rep_tsne[:, 1].min()-5, rep_tsne[:, 1].max()+5) # 최소, 최대
    
    plt.savefig(os.path.join(tsne_folder_name, f'tsne_{tsne_mode}.png'), dpi=1000)
    plt.close('all')
    
    
def save_rep_labels(args, model, ind_loader, ood_loader, use_aug):
    model.eval()
    tsne_file_names = tsne_file_name(args)[:-1]
    tsne_file_exists = True
    for filename in tsne_file_names:
        tsne_file_exists *= os.path.exists(filename)
    if tsne_file_exists:
        print('Already prepared for tsne.')
        return
    f_name, l_f_name, ood_f_name, ood_l_f_name = tsne_file_names
    # for tokens, labels, data_indices in tqdm(ind_loader, desc ='Saving sentence representataion ...'):
    for batch in tqdm(ind_loader, desc ='Saving sentence representataion ...'):
        
        with torch.no_grad():
            if use_aug: 
                labels = batch[0]['labels']
                bat = {key: value.to(args.device) for key, value in batch[0].items()}
                reps = model(**bat)[1]  # global projection, representation for t-SNE
            else:
                labels = batch['labels']
                batch = {key: value.to(args.device) for key, value in batch.items()}
                reps = model(**batch)[1]  # global projection, representation for t-SNE

        reps = F.normalize(reps, dim=1)
        reps = reps.detach().cpu().numpy()
        labels = labels.detach().numpy()

        with open(f_name, 'a') as f:
            np.savetxt(f, reps, fmt='%f', delimiter=',', newline='\r')
        with open(l_f_name, 'a') as f2:
            np.savetxt(f2, labels, fmt='%d', delimiter=',', newline='\r')

    # for ood_tokens, ood_labels, ood_data_indices in tqdm(ood_loader, desc ='Saving sentence representataion ...'):
    for batch in tqdm(ood_loader, desc ='Saving sentence representataion ...'):
        ood_labels = batch['labels']
        batch = {key: value.to(args.device) for key, value in batch.items()}

        with torch.no_grad():
            ood_reps =  model(**batch)[1]
        ood_reps = F.normalize(ood_reps, dim=1)
        ood_reps = ood_reps.detach().cpu().numpy()
        ood_labels = ood_labels.detach().numpy()

        with open(ood_f_name, 'a') as f:
            np.savetxt(f, ood_reps, fmt='%f', delimiter=',', newline='\r')
        with open(ood_l_f_name, 'a') as f2:
            np.savetxt(f2, ood_labels, fmt='%d', delimiter=',', newline='\r')
            
            
def tsne(args, model, ind_loader, ood_loader, use_aug=False):
    tsne_file_names = model.save_rep_labels(args, ind_loader, ood_loader)
    # save_rep_labels(args, model, ind_loader, ood_loader, use_aug)
    # train_tsne(args, tsne_mode='domain')
    train_tsne(tsne_file_names, tsne_mode='intent')
    
    
def analyze_result(args, model_name, ood_method='maha'):
    if args.split:
        ind_tsv_path = os.path.join(hydra.utils.to_absolute_path(args.save_dir), args.augment_method, f"{args.task_name}-{args.split_ratio}-seed{args.seed}", model_name, 'ind', 'error_cases.tsv')
    else:
        ind_tsv_path = os.path.join(hydra.utils.to_absolute_path(args.save_dir), args.augment_method, f"{args.task_name}-no_split-seed{args.seed}", model_name, 'ind', 'error_cases.tsv')
    if args.split:
        ood_dir = os.path.join(hydra.utils.to_absolute_path(args.save_dir), args.augment_method, f"{args.task_name}-{args.split_ratio}-seed{args.seed}", model_name, f'Wrong_OOD({ood_method})')
    else:
        ood_dir = os.path.join(hydra.utils.to_absolute_path(args.save_dir), args.augment_method, f"{args.task_name}-no_split-seed{args.seed}", model_name, f'Wrong_OOD({ood_method})')
    ood_tsv_path = os.path.join(ood_dir, 'error_cases.tsv')
    ind_rows = []
    ind_dict = {}
    with open(ind_tsv_path, 'r') as f:
        ind_reader = csv.reader(f, delimiter='\t')
        for row in ind_reader:
            ind_rows.append(row)
            
    label_loc = ind_rows[0].index('True_label')
    for i_row in ind_rows[1:]:
        ind_label = i_row[label_loc]
        if ind_label not in ind_dict.keys():
            ind_dict[ind_label] = []
        ind_dict[ind_label].append(i_row)
            
            
    ood_rows = []
    ood_dict = {}
    with open(ood_tsv_path, 'r') as f:
        ood_reader = csv.reader(f, delimiter='\t')
        for row in ood_reader:
            ood_rows.append(row)
            
            
    label_loc = ood_rows[0].index('True_label')
    for o_row in ood_rows[1:]:
        ood_label = o_row[label_loc]
        if ood_label not in ood_dict.keys():
            ood_dict[ood_label] = [o_row]
        else:
            ood_dict[ood_label].append(o_row)
        
    scores_summary = {}
    label_counter = {}
    pred_results = {}
    for k, oods in ood_dict.items():    # k:true label
        curr_scores = []
        scores_summary[k] = [0, 0]
        label_counter[k] = {}
        pred_results[k] = {}
        for idx, text, true_label, pred_label, ood_score in oods:
            curr_scores.append(float(ood_score))
            if pred_label not in label_counter[k].keys():
                label_counter[k][pred_label] = 1
            else:
                label_counter[k][pred_label] += 1
            if pred_label not in pred_results[k].keys():
                pred_results[k][pred_label] = [text]
            else:
                pred_results[k][pred_label].append(text)
        scores_summary[k][0] = sum(curr_scores) / len(curr_scores)
        scores_summary[k][1] = pstdev(curr_scores)
    # pdb.set_trace()
    scores_summary_sorted = sorted(scores_summary.items(), key=lambda x: x[1][0])
    
    with open(os.path.join(ood_dir, 'avg_scores.txt'), 'w') as f:
        f.write(f'label\taverage\tstdev\n')
        for label, (avg, stdev) in scores_summary_sorted:
            f.write(f'{label}\t{avg}\t{stdev}\n')
            
    with open(os.path.join(ood_dir, 'ood_summary.txt'), 'w') as f:
        f.write(f'True label -> Pred label\n')
        for true_label, v in label_counter.items():
            label_count_sorted = sorted(v.items(), key=lambda x: x[1], reverse=True)
            f.write(f'----------------------------{true_label}----------------------------\n')
            for pred_label, cnt in label_count_sorted:
                f.write(f'{true_label} -> {pred_label} ({cnt} occurences)\n')
                for text in pred_results[true_label][pred_label]:
                    f.write(text + '\n')
    
