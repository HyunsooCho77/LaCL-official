import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from transformers import BertPreTrainedModel, BertModel
from sklearn.covariance import EmpiricalCovariance,LedoitWolf
from loss import *



def mean_pooling(output, attention_mask):
    # output : B X seq_len X D
    # attention_mask : B x seq_len
    input_mask_expanded = attention_mask[:,0:].unsqueeze(-1).expand(output.size()).float()
    return torch.sum(output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class LACL(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.lacl_config = args
        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        # --------------------------------------------------------------- #
        self.num_layers = 12
        self.prj_dim = int (args.projection_dim/len(self.lacl_config.gp_layers))
        # self.prj_dim = args.projection_dim
        
        if self.lacl_config.gp_pooling == 'concat':
            self.global_projector = nn.Sequential(nn.Linear(768,args.encoder_dim),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Linear(args.encoder_dim, self.prj_dim))
        elif self.lacl_config.gp_pooling == 'mean_pool':
            self.global_projector = nn.Sequential(nn.Linear(768,args.encoder_dim),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Linear(args.encoder_dim, 768))
        
        self.cls_projector = nn.Sequential(nn.Linear(768,args.encoder_dim),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Linear(args.encoder_dim, 128))
        # --------------------------------------------------------------- #
        self.init_weights()
        self.model_name = 'LACL_fixed'
        

    def forward(self, input_ids=None, attention_mask=None, indices=None,labels=None):
        
        return_dict = {}        
        # feed input
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        out_all_hidden = outputs[2]
        return_dict['all_hidden'] = out_all_hidden
        out_last_cls = out_all_hidden[-1][:, 0, :]
        
        
        # last_cls
        return_dict['last_cls'] = out_last_cls

        # Global_projection & Reg_loss input
        lw_gp = []
        lw_mean_tok_embedding= []
        for i in range(1,1 + self.num_layers):    

            if i in self.lacl_config.gp_layers:

                if self.lacl_config.gp_location =='cls':
                    all_tokens = out_all_hidden[i][:,0,:]
                    lw_mean_tok_embedding.append(all_tokens)
                    lw_gp.append(self.global_projector(out_all_hidden[i][:,0,:])) 

                elif self.lacl_config.gp_location =='token':
                    pooled = mean_pooling(out_all_hidden[i][:,0:,:], attention_mask)
                    lw_mean_tok_embedding.append(pooled)
                    lw_gp.append(self.global_projector(pooled)) 

        
        if self.lacl_config.gp_pooling == 'concat':
            global_projection = torch.cat(lw_gp, dim=1)
        elif self.lacl_config.gp_pooling == 'mean_pool':
            global_projection = sum(lw_gp)/len(lw_gp)

        return_dict['lw_mean_embedding'] = lw_gp
        return_dict['global_projection'] = global_projection


        return return_dict

    def compute_ood(self,input_ids=None,attention_mask=None,labels=None,indices=None,ind=False):
        # outputs = self.bert()
        
        b_dict = self.forward(input_ids, attention_mask=attention_mask)
        pooled = b_dict['global_projection']

        maha_score = []
        for c in self.all_classes:
            centered_pooled = pooled - self.class_mean[c].unsqueeze(0)
            ms = torch.diag(centered_pooled @ self.class_var @ centered_pooled.t())
            maha_score.append(ms)
        maha_score = torch.stack(maha_score, dim=-1)
        maha_score, pred = maha_score.min(-1)
        maha_score = -maha_score
        
        if ind == True:
            correct = (labels == pred).float().sum()
        else:
            correct = 0

        norm_pooled = F.normalize(pooled, dim=-1)

        #! HS
        cosine_score = norm_pooled @ self.norm_bank.t()
        # cosine_score = F.cosine_similarity(pooled,self.norm_bank)
        # mean_cosine_score = norm_pooled @ F.normalize(self.class_mean, dim=-1).t()
        # cosine_score += mean_cosine_score.max(-1)[0]
        
        cosine_score, cosine_idx = cosine_score.topk(k=self.lacl_config.cosine_top_k,dim=-1)
        harmonic_weight = np.reciprocal([float(i) for i in range(1,1+self.lacl_config.cosine_top_k)])
        cosine_score = (cosine_score * torch.from_numpy(harmonic_weight).cuda()).sum(dim=-1)
        

        cosine_correct = sum(self.label_bank[[cosine_idx.squeeze()]] ==labels)

        ood_keys = {
            'maha': maha_score.tolist(),
            'cosine': cosine_score.tolist(),
            'maha_acc': correct,
            'cosine_correct': cosine_correct
             
        }
        return ood_keys

    def prepare_ood(self, dataloader=None):
        self.bank = None
        self.label_bank = None
        for batch in dataloader:
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            labels = batch['labels']
            
            b_dict = self.forward(**batch)
            pooled = b_dict['global_projection']

            if self.bank is None:
                self.bank = pooled.clone().detach()
                self.label_bank = labels.clone().detach()
            else:
                bank = pooled.clone().detach()
                label_bank = labels.clone().detach()
                self.bank = torch.cat([bank, self.bank], dim=0)
                self.label_bank = torch.cat([label_bank, self.label_bank], dim=0)


        self.norm_bank = F.normalize(self.bank, dim=-1)
        N, d = self.bank.size()
        self.all_classes = list(set(self.label_bank.tolist()))
        self.class_mean = torch.zeros(max(self.all_classes) + 1, d).cuda()

        for c in self.all_classes:
            self.class_mean[c] = (self.bank[self.label_bank == c].mean(0))
        centered_bank = (self.bank - self.class_mean[self.label_bank]).detach().cpu().numpy()
        
        # precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(np.float32)
        precision = LedoitWolf().fit(centered_bank).precision_.astype(np.float32)
        self.class_var = torch.from_numpy(precision).float().cuda()
