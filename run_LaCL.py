import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer
from transformers.optimization import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import set_seed, collate_fn, collate_fn2, PairDataset, LACL_load_aug, write_log2, cal_num_labels
from model import *
from torch.utils.data import DataLoader
from evaluation import *
import hydra
from omegaconf import DictConfig
import warnings
from loss import *
import os
from torch.optim import AdamW
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
    

    
    for epoch in range(int(args.num_train_epochs)):    
        # dataset load & log
        benchmarks = ()
        train_aug1_dataset, train_aug2_dataset, test_ind_dataset, test_ood_dataset, train_raw_dataset = LACL_load_aug(args, tokenizer)
        benchmarks = (('ood_'+ args.task_name, test_ood_dataset),) + benchmarks
        comb_dset = PairDataset(train_aug1_dataset, train_aug2_dataset)
        comb_train_dataloader = DataLoader(comb_dset, shuffle=True, drop_last=True, collate_fn=collate_fn2, batch_size=args.train_batch, num_workers=args.num_workers)
        train_dataloader = DataLoader(train_raw_dataset, batch_size=100, collate_fn=collate_fn, shuffle=True, drop_last=True)

        # scheduler
        total_steps = int(len(comb_train_dataloader) * args.num_train_epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max = total_steps, eta_min = args.learning_rate * args.eta_min)
        model.zero_grad()
        
        # initialize
        train_bar = tqdm(comb_train_dataloader)
        total_cl_loss = total_reg_loss = 0


        # train code
        for idx, (batch1,batch2) in enumerate(train_bar):
            
            # batch initialize
            model.train()
            batch1 = {key: value.to(DEVICE) for key, value in batch1.items()}
            batch2 = {key: value.to(DEVICE) for key, value in batch2.items()}
            
            # Feed batch into model
            b1_dict = model(**batch1)
            b2_dict = model(**batch2)
            

            total_loss = 0

            # Contrastive Learning loss ()
            label = batch1['labels'].cuda()
            cont = torch.cat([b1_dict['global_projection'],b2_dict['global_projection']], dim=0)
            sim_mat = get_sim_mat(cont)
            loss_cl_gp = Supervised_NT_xent(sim_mat, labels=label, temperature=args.temperature)
            total_cl_loss += loss_cl_gp.item()
            total_loss += loss_cl_gp
            
            # Correlation Regularization loss
            tok_embeddings = [b1_dict['lw_mean_embedding'], b2_dict['lw_mean_embedding']]
            loss_reg = reg_loss(tok_embeddings,args)
            loss_cr = loss_reg
            total_reg_loss += loss_reg.item()
            total_loss += loss_cr * args.reg_loss_weight

            # Train_bar description
            train_bar.set_description(f'Train epoch {epoch}, GCL_CL Loss: {(total_cl_loss)/(idx+1):.2f},Reg Loss {(total_reg_loss)/(idx+1):.2f}:')

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()


        # Test accuracy evaluation
        if epoch > 0 and epoch % args.log_interval == 0:

            model.prepare_ood(train_dataloader)
            for tag, ood_features in benchmarks:
                base_results = evaluate_ood(args, model, test_ind_dataset, ood_features, tag)
                print(base_results)

            write_log2(args, args.model_name, base_results)
                


@hydra.main(config_path=".", config_name="model_config.yaml")
def main(args: DictConfig) -> None:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.device_count())
    args.n_gpu = torch.cuda.device_count()
    DEVICE = device
    dataset = args.task_name
    set_seed(args)

    num_labels = cal_num_labels(args)

    if args.model_name_or_path.startswith('bert'):
        config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        if args.no_dropout:
            config.attention_probs_dropout_prob = 0.0
            config.hidden_dropout_prob = 0.0
        config.output_attentions = True
        config.output_hidden_states = True

        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = LACL.from_pretrained(args.model_name_or_path, config=config,args=args)            
        model.to(0)

    print(args)
    train(args, model, tokenizer)


if __name__ == "__main__":
    main()

