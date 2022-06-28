import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from utils import *
from loss import nt_xent_sup
import pdb
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel, RobertaModel, BertModel
from sklearn.covariance import EmpiricalCovariance,LedoitWolf

from sklearn.neighbors import LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import EmpiricalCovariance
from loss import *



def mean_pooling(output, attention_mask):
    # output : B X seq_len X D
    # attention_mask : B x seq_len
    input_mask_expanded = attention_mask[:,0:].unsqueeze(-1).expand(output.size()).float()
    return torch.sum(output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class LACL_fixed(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.lacl_config = args
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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


    def reset_proxy(self, dataloader):

        self.bank = None
        self.label_bank = None

        for batch in dataloader:
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            labels = batch['labels']
            b_dict = self.forward(**batch)

            if self.lacl_config.final_rep == 'last_cls':
                pooled = b_dict['last_cls']
            elif self.lacl_config.final_rep == 'global_projection':
                pooled = b_dict['global_projection']
            elif self.lacl_config.final_rep == 'last_cls+global_projection':
                pooled = torch.cat([b_dict['global_projection'],b_dict['last_cls']], dim=1)

            if self.bank is None:
                self.bank = pooled.clone().detach()
                self.label_bank = labels.clone().detach()
            else:
                bank = pooled.clone().detach()
                label_bank = labels.clone().detach()
                self.bank = torch.cat([bank, self.bank], dim=0)
                self.label_bank = torch.cat([label_bank, self.label_bank], dim=0)

        # self.norm_bank = F.normalize(self.bank, dim=-1)
        N, d = self.bank.size()
        self.all_classes = list(set(self.label_bank.tolist()))
        # initialize
        self.class_mean = torch.zeros(max(self.all_classes) + 1, d).cuda()
        # calculate centroid
        for c in self.all_classes:
            self.class_mean[c] = F.normalize(self.bank[self.label_bank == c].mean(0),dim=0)

        
        # print min, max, avg cosine similarity
        total, idx, max_sim, min_sim =0,0,0,1
        for i in range(len(self.class_mean)):
            for j in range(len(self.class_mean)):
                if i != j:
                    idx +=1
                    # import pdb
                    # pdb.set_trace()
                    sim = (self.class_mean[i] @ self.class_mean[j].t()).clamp(min=1e-7)
                    total +=sim
                    if sim >= max_sim :
                        max_sim = sim
                    if sim <= min_sim :
                        min_sim = sim
        print(f'average, maximum, minimum similarity between centroids: {total*100/idx:.3f}, {max_sim*100:.3f}, {min_sim*100:.3f}')

        # self.proxy = nn.Embedding.from_pretrained(self.class_mean)
        # self.proxy = nn.Parameter(F.normalize(self.class_mean,dim=-1))
        self.proxy = nn.Parameter(self.class_mean)
        

    def forward(self, input_ids=None, attention_mask=None, indices=None,labels=None):
        
        return_dict = {}        
        # feed input
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        out_all_hidden = outputs[2]
        return_dict['all_hidden'] = out_all_hidden
        out_last_cls = out_all_hidden[-1][:, 0, :]
        
        
        # last_cls
        return_dict['last_cls'] = out_last_cls

        # Contrastive learning input (1)
        cls_projection = self.cls_projector(out_last_cls) if self.lacl_config.cl2_projector == True else out_last_cls
        return_dict['cls_projection'] = cls_projection
            

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
                    lw_gp.append(self.global_projector(pooled)) # without [cls] token

        
        if self.lacl_config.gp_pooling == 'concat':
            global_projection = torch.cat(lw_gp, dim=1)
        elif self.lacl_config.gp_pooling == 'mean_pool':
            global_projection = sum(lw_gp)/len(lw_gp)

        # return_dict['lw_mean_embedding'] = lw_mean_tok_embedding
        return_dict['lw_mean_embedding'] = lw_gp
        return_dict['global_projection'] = global_projection

        # Cross entropy input
        #! HS
        pooler = outputs[1]
        logits = self.classifier(self.dropout(pooler))
        #! HS
        # logits = self.classifier(self.dropout(out_last_cls))
        return_dict['logits'] = logits
        

        # proxy-anchor loss
        try:
            global_projection = F.normalize(global_projection,dim = -1)
            proxy_sim = global_projection.mm(self.proxy.t())
            return_dict['proxy_sim'] = proxy_sim
        except:
            pass



        return return_dict

    def compute_ood(self,final_rep,input_ids=None,attention_mask=None,labels=None,indices=None,ind=False):
        # outputs = self.bert()
        
        b_dict = self.forward(input_ids, attention_mask=attention_mask)
        if final_rep == 'last_cls':
            pooled = b_dict['last_cls']
        elif final_rep == 'global_projection':
            pooled = b_dict['global_projection']
            dim = pooled.size()[1] //2
            pooled = pooled[:,dim:]

        elif final_rep == 'global_projection_half':
            pooled = b_dict['global_projection']

        elif final_rep == 'cls_projection':
            pooled = b_dict['cls_projection']
        elif final_rep == 'last_cls+global_projection':
            pooled = torch.cat([b_dict['global_projection'],b_dict['last_cls']], dim=1)
        elif final_rep == 1 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[1][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[1][:,0:,:], attention_mask)
        elif final_rep == 2 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[2][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[2][:,0:,:], attention_mask)
        elif final_rep == 3 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[3][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[3][:,0:,:], attention_mask)
        elif final_rep == 4 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[4][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[4][:,0:,:], attention_mask)
        elif final_rep == 5 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[5][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[5][:,0:,:], attention_mask)
        elif final_rep == 6 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[6][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[6][:,0:,:], attention_mask)
        elif final_rep == 7 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[7][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[7][:,0:,:], attention_mask)
        elif final_rep == 8 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[8][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[8][:,0:,:], attention_mask)
        elif final_rep == 9 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[9][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[9][:,0:,:], attention_mask)
        elif final_rep == 10 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[10][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[10][:,0:,:], attention_mask)
        elif final_rep == 11 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[11][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[11][:,0:,:], attention_mask)
        elif final_rep == 12 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[12][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[12][:,0:,:], attention_mask)
        
        
        logits = b_dict['logits']

        ood_keys = None
        softmax_score = F.softmax(logits, dim=-1).max(-1)[0]

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

        #! raw
        cosine_score = norm_pooled @ self.norm_bank.t()
        # cosine_score = pooled @ self.norm_bank.t()
        #! HS
        # cosine_score = F.cosine_similarity(pooled,self.norm_bank)
        
        # cosine_score = cosine_score.max(-1)[0]
        # cosine_score = cosine_score.topk(k=self.lacl_config.cosine_top_k,dim=-1)[0].sum(dim=-1)
        cosine_score, cosine_idx = cosine_score.topk(k=self.lacl_config.cosine_top_k,dim=-1)
        harmonic_weight = np.reciprocal([float(i) for i in range(1,1+self.lacl_config.cosine_top_k)])
        cosine_score = (cosine_score * torch.from_numpy(harmonic_weight).cuda()).sum(dim=-1)
        

        cosine_correct = sum(self.label_bank[[cosine_idx.squeeze()]] ==labels)

        
        #! HS
        # mean_cosine_score = norm_pooled @ F.normalize(self.class_mean, dim=-1).t()
        # cosine_score += mean_cosine_score.max(-1)[0]
        #! HS
        

        energy_score = torch.logsumexp(logits, dim=-1)

        ood_keys = {
            'softmax': softmax_score.tolist(),
            'maha': maha_score.tolist(),
            'cosine': cosine_score.tolist(),
            'energy': energy_score.tolist(),
            'maha_acc': correct,
            'cosine_correct': cosine_correct
             
        }
        return ood_keys

    def prepare_ood(self, final_rep, dataloader=None):
        self.bank = None
        self.label_bank = None
        for batch in dataloader:
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            labels = batch['labels']
            
            b_dict = self.forward(**batch)
            if final_rep == 'last_cls':
                pooled = b_dict['last_cls']
            elif final_rep == 'cls_projection':
                pooled = b_dict['cls_projection']
            elif final_rep == 'global_projection':
                pooled = b_dict['global_projection']
                dim = pooled.size()[1]//2
                pooled = pooled[:, dim:]

            elif final_rep == 'global_projection_half':
                pooled = b_dict['global_projection']
                dim = pooled.size()[1]
                pooled = pooled[:][dim:]
                
            elif final_rep == 'last_cls+global_projection':
                pooled = torch.cat([b_dict['global_projection'],b_dict['last_cls']], dim=1)
            elif final_rep == 1 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[1][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[1][:,0:,:], batch['attention_mask'])
            elif final_rep == 2 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[2][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[2][:,0:,:], batch['attention_mask'])
            elif final_rep == 3 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[3][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[3][:,0:,:], batch['attention_mask'])
            elif final_rep == 4 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[4][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[4][:,0:,:], batch['attention_mask'])
            elif final_rep == 5 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[5][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[5][:,0:,:], batch['attention_mask'])
            elif final_rep == 6 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[6][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[6][:,0:,:], batch['attention_mask'])
            elif final_rep == 7 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[7][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[7][:,0:,:], batch['attention_mask'])
            elif final_rep == 8 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[8][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[8][:,0:,:], batch['attention_mask'])
            elif final_rep == 9 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[9][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[9][:,0:,:], batch['attention_mask'])
            elif final_rep == 10 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[10][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[10][:,0:,:], batch['attention_mask'])
            elif final_rep == 11 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[11][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[11][:,0:,:], batch['attention_mask'])
            elif final_rep == 12 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[12][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[12][:,0:,:], batch['attention_mask'])
                

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

        # import pdb
        # pdb.set_trace()
        for c in self.all_classes:
            self.class_mean[c] = (self.bank[self.label_bank == c].mean(0))
        centered_bank = (self.bank - self.class_mean[self.label_bank]).detach().cpu().numpy()
        
        # precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(np.float32)
        precision = LedoitWolf().fit(centered_bank).precision_.astype(np.float32)
        self.class_var = torch.from_numpy(precision).float().cuda()

        import pdb
        pdb.set_trace()








class Baselines(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.lacl_config = args
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
        self.model_name = 'Baselines'
        

    def forward(self, input_ids=None, attention_mask=None, indices=None,labels=None):
        return_dict = {}        
        # feed input
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        out_all_hidden = outputs[2]
        return_dict['all_hidden'] = out_all_hidden
        out_last_cls = out_all_hidden[-1][:, 0, :]

        out_last_pool = mean_pooling(out_all_hidden[-1][:,0:,:], attention_mask)
        
        # last_cls
        return_dict['last_cls'] = out_last_cls

        # Contrastive learning input (1)
        if self.lacl_config.ce_cl2_projection == 'token':

            cl2_projection = self.cls_projector(out_last_pool) if self.lacl_config.cl2_projector == True else out_last_pool
            return_dict['cls_projection'] = cl2_projection

            logits = self.classifier(self.dropout(out_last_pool))
            return_dict['logits'] = logits

        elif self.lacl_config.ce_cl2_projection == 'cls':
            cl2_projection = self.cls_projector(out_last_cls) if self.lacl_config.cl2_projector == True else out_last_cls
            return_dict['cls_projection'] = cl2_projection

            logits = self.classifier(self.dropout(out_last_cls))
            return_dict['logits'] = logits

        return return_dict

    def compute_ood(self,final_rep,input_ids=None,attention_mask=None,labels=None,indices=None,ind=False):
        # outputs = self.bert()
        
        b_dict = self.forward(input_ids, attention_mask=attention_mask)
        if final_rep == 'last_cls':
            pooled = b_dict['last_cls']
        elif final_rep == 'global_projection':
            pooled = b_dict['global_projection']
            dim = pooled.size()[1] //2
            pooled = pooled[:,dim:]

        elif final_rep == 'global_projection_half':
            pooled = b_dict['global_projection']

        elif final_rep == 'cls_projection':
            pooled = b_dict['cls_projection']
        elif final_rep == 'last_cls+global_projection':
            pooled = torch.cat([b_dict['global_projection'],b_dict['last_cls']], dim=1)
        elif final_rep == 1 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[1][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[1][:,0:,:], attention_mask)
        elif final_rep == 2 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[2][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[2][:,0:,:], attention_mask)
        elif final_rep == 3 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[3][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[3][:,0:,:], attention_mask)
        elif final_rep == 4 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[4][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[4][:,0:,:], attention_mask)
        elif final_rep == 5 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[5][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[5][:,0:,:], attention_mask)
        elif final_rep == 6 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[6][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[6][:,0:,:], attention_mask)
        elif final_rep == 7 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[7][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[7][:,0:,:], attention_mask)
        elif final_rep == 8 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[8][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[8][:,0:,:], attention_mask)
        elif final_rep == 9 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[9][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[9][:,0:,:], attention_mask)
        elif final_rep == 10 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[10][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[10][:,0:,:], attention_mask)
        elif final_rep == 11 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[11][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[11][:,0:,:], attention_mask)
        elif final_rep == 12 :
            out_all_hidden = b_dict['all_hidden']
            if self.lacl_config.gp_location =='cls':
                pooled =  out_all_hidden[12][:,0,:]
            elif self.lacl_config.gp_location == 'token':
                pooled = mean_pooling(out_all_hidden[12][:,0:,:], attention_mask)
        
        
        logits = b_dict['logits']

        ood_keys = None
        softmax_score = F.softmax(logits, dim=-1).max(-1)[0]

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

        #! raw
        cosine_score = norm_pooled @ self.norm_bank.t()
        # cosine_score = pooled @ self.norm_bank.t()
        #! HS
        # cosine_score = F.cosine_similarity(pooled,self.norm_bank)
        
        # cosine_score = cosine_score.max(-1)[0]
        # cosine_score = cosine_score.topk(k=self.lacl_config.cosine_top_k,dim=-1)[0].sum(dim=-1)
        cosine_score, cosine_idx = cosine_score.topk(k=self.lacl_config.cosine_top_k,dim=-1)
        harmonic_weight = np.reciprocal([float(i) for i in range(1,1+self.lacl_config.cosine_top_k)])
        cosine_score = (cosine_score * torch.from_numpy(harmonic_weight).cuda()).sum(dim=-1)
        

        cosine_correct = sum(self.label_bank[[cosine_idx.squeeze()]] ==labels)

        
        #! HS
        # mean_cosine_score = norm_pooled @ F.normalize(self.class_mean, dim=-1).t()
        # cosine_score += mean_cosine_score.max(-1)[0]
        #! HS
        

        energy_score = torch.logsumexp(logits, dim=-1)

        ood_keys = {
            'softmax': softmax_score.tolist(),
            'maha': maha_score.tolist(),
            'cosine': cosine_score.tolist(),
            'energy': energy_score.tolist(),
            'maha_acc': correct,
            'cosine_correct': cosine_correct
             
        }
        return ood_keys

    def prepare_ood(self, final_rep, dataloader=None):
        self.bank = None
        self.label_bank = None
        for batch in dataloader:
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            labels = batch['labels']
            
            b_dict = self.forward(**batch)
            if final_rep == 'last_cls':
                pooled = b_dict['last_cls']
            elif final_rep == 'cls_projection':
                pooled = b_dict['cls_projection']
            elif final_rep == 'global_projection':
                pooled = b_dict['global_projection']
                dim = pooled.size()[1]//2
                pooled = pooled[:, dim:]

            elif final_rep == 'global_projection_half':
                pooled = b_dict['global_projection']
                dim = pooled.size()[1]
                pooled = pooled[:][dim:]
                
            elif final_rep == 'last_cls+global_projection':
                pooled = torch.cat([b_dict['global_projection'],b_dict['last_cls']], dim=1)
            elif final_rep == 1 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[1][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[1][:,0:,:], batch['attention_mask'])
            elif final_rep == 2 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[2][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[2][:,0:,:], batch['attention_mask'])
            elif final_rep == 3 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[3][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[3][:,0:,:], batch['attention_mask'])
            elif final_rep == 4 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[4][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[4][:,0:,:], batch['attention_mask'])
            elif final_rep == 5 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[5][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[5][:,0:,:], batch['attention_mask'])
            elif final_rep == 6 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[6][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[6][:,0:,:], batch['attention_mask'])
            elif final_rep == 7 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[7][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[7][:,0:,:], batch['attention_mask'])
            elif final_rep == 8 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[8][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[8][:,0:,:], batch['attention_mask'])
            elif final_rep == 9 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[9][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[9][:,0:,:], batch['attention_mask'])
            elif final_rep == 10 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[10][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[10][:,0:,:], batch['attention_mask'])
            elif final_rep == 11 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[11][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[11][:,0:,:], batch['attention_mask'])
            elif final_rep == 12 :
                out_all_hidden = b_dict['all_hidden']
                if self.lacl_config.gp_location =='cls':
                    pooled =  out_all_hidden[12][:,0,:]
                elif self.lacl_config.gp_location == 'token':
                    pooled = mean_pooling(out_all_hidden[12][:,0:,:], batch['attention_mask'])
                

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

        # import pdb
        # pdb.set_trace()
        for c in self.all_classes:
            self.class_mean[c] = (self.bank[self.label_bank == c].mean(0))
        centered_bank = (self.bank - self.class_mean[self.label_bank]).detach().cpu().numpy()
        
        # precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(np.float32)
        precision = LedoitWolf().fit(centered_bank).precision_.astype(np.float32)
        self.class_var = torch.from_numpy(precision).float().cuda()








class LACL_SA(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.lacl_config = args
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # --------------------------------------------------------------- #
        self.num_layers = 12
        self.prj_dim = int (args.projection_dim/len(self.lacl_config.gp_layers))
        # self.prj_dim = args.projection_dim
        self.global_projector = nn.Sequential(nn.Linear(768,args.encoder_dim),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Linear(args.encoder_dim, self.prj_dim))

        # Self-attention projector
        self.gp_query = nn.Linear(768,768)
        self.gp_key = nn.Linear(768,768)
        self.gp_value = nn.Linear(768,768)
        self.gp_SA = nn.MultiheadAttention(embed_dim = 768, num_heads = 12)
        
        self.cls_projector = nn.Sequential(nn.Linear(768,args.encoder_dim),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Linear(args.encoder_dim, 128))
        # --------------------------------------------------------------- #
        self.init_weights()
        self.model_name = 'LACL_SA'


    def reset_proxy(self, dataloader):

        self.bank = None
        self.label_bank = None

        for batch in dataloader:
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            labels = batch['labels']
            b_dict = self.forward(**batch)

            if self.lacl_config.final_rep == 'last_cls':
                pooled = b_dict['last_cls']
            elif self.lacl_config.final_rep == 'global_projection':
                pooled = b_dict['global_projection']
            elif self.lacl_config.final_rep == 'last_cls+global_projection':
                pooled = torch.cat([b_dict['global_projection'],b_dict['last_cls']], dim=1)

            if self.bank is None:
                self.bank = pooled.clone().detach()
                self.label_bank = labels.clone().detach()
            else:
                bank = pooled.clone().detach()
                label_bank = labels.clone().detach()
                self.bank = torch.cat([bank, self.bank], dim=0)
                self.label_bank = torch.cat([label_bank, self.label_bank], dim=0)

        # self.norm_bank = F.normalize(self.bank, dim=-1)
        N, d = self.bank.size()
        self.all_classes = list(set(self.label_bank.tolist()))
        # initialize
        self.class_mean = torch.zeros(max(self.all_classes) + 1, d).cuda()
        # calculate centroid
        for c in self.all_classes:
            self.class_mean[c] = F.normalize(self.bank[self.label_bank == c].mean(0),dim=0)

        
        # print min, max, avg cosine similarity
        total, idx, max_sim, min_sim =0,0,0,1
        for i in range(len(self.class_mean)):
            for j in range(len(self.class_mean)):
                if i != j:
                    idx +=1
                    # import pdb
                    # pdb.set_trace()
                    sim = (self.class_mean[i] @ self.class_mean[j].t()).clamp(min=1e-7)
                    total +=sim
                    if sim >= max_sim :
                        max_sim = sim
                    if sim <= min_sim :
                        min_sim = sim
        print(f'average, maximum, minimum similarity between centroids: {total*100/idx:.3f}, {max_sim*100:.3f}, {min_sim*100:.3f}')

        # self.proxy = nn.Embedding.from_pretrained(self.class_mean)
        # self.proxy = nn.Parameter(F.normalize(self.class_mean,dim=-1))
        self.proxy = nn.Parameter(self.class_mean)
        

    def forward(self, input_ids=None, attention_mask=None, indices=None,labels=None):
        
        return_dict = {}        
        # feed input
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        out_all_hidden = outputs[2]
        return_dict['all_hidden'] = out_all_hidden
        out_last_cls = out_all_hidden[-1][:, 0, :]
        return_dict['last_cls'] = out_last_cls

        # Contrastive learning input (1)
        cls_projection = self.cls_projector(out_last_cls) if self.lacl_config.cl2_projector == True else out_last_cls
        return_dict['cls_projection'] = cls_projection

        # Global_projection & Reg_loss input
        lw_gp = []
        lw_mean_tok_embedding= []


        # SA projector
        key, value = [], [] 
        for i in range(1,1 + self.num_layers):    

            if i in self.lacl_config.gp_layers:

                if self.lacl_config.gp_location =='cls':
                    all_tokens = out_all_hidden[i][:,0,:]
                    lw_mean_tok_embedding.append(all_tokens)
                    
                    key.append(self.gp_key(all_tokens))
                    value.append(self.gp_value(all_tokens))

                    # key.append(all_tokens)
                    # value.append(all_tokens)
                    
                    if i ==12:
                        query = self.gp_query(pooled)
                        query.unsqueeze_(dim=0)


                elif self.lacl_config.gp_location =='token':
                    pooled = mean_pooling(out_all_hidden[i][:,0:,:], attention_mask)
                    lw_mean_tok_embedding.append(pooled)

                    key.append(self.gp_key(pooled))
                    value.append(self.gp_value(pooled))
                    
                    if i ==12:
                        query = self.gp_query(pooled)
                        query.unsqueeze_(dim=0)

        key = torch.stack(key,1)
        value = torch.stack(value,1)
        
        global_projection = self.gp_SA(query, key, value)[0].squeeze()


        # import pdb
        # pdb.set_trace()

        # global_projection = torch.mean(global_projection[0],dim=1)


        # global_projection = torch.cat(lw_gp, dim=1)
        return_dict['lw_mean_embedding'] = lw_mean_tok_embedding
        return_dict['global_projection'] = global_projection

        # Cross entropy input
        logits = self.classifier(self.dropout(out_last_cls))
        return_dict['logits'] = logits
        

        # proxy-anchor loss
        try:
            global_projection = F.normalize(global_projection,dim = -1)
            proxy_sim = global_projection.mm(self.proxy.t())
            return_dict['proxy_sim'] = proxy_sim
        except:
            pass



        return return_dict

    def compute_ood(self,final_rep,input_ids=None,attention_mask=None,labels=None,indices=None,ind=False):
        # outputs = self.bert()
        
        b_dict = self.forward(input_ids, attention_mask=attention_mask)
        if final_rep == 'last_cls':
            pooled = b_dict['last_cls']
        elif final_rep == 'global_projection':
            pooled = b_dict['global_projection']
        elif final_rep == 'last_cls+global_projection':
            pooled = torch.cat([b_dict['global_projection'],b_dict['last_cls']], dim=1)
        
        
        logits = b_dict['logits']

        ood_keys = None
        softmax_score = F.softmax(logits, dim=-1).max(-1)[0]

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

        #! raw
        # cosine_score = norm_pooled @ self.norm_bank.t()
        cosine_score = pooled @ self.norm_bank.t()
        #! HS
        # cosine_score = F.cosine_similarity(pooled,self.norm_bank)
        
        cosine_score = cosine_score.max(-1)[0]
        # cosine_score = cosine_score.topk(k=1,dim=-1)[0].sum(dim=-1)
        #! raw
        
        #! HS
        # mean_cosine_score = norm_pooled @ F.normalize(self.class_mean, dim=-1).t()
        # cosine_score += mean_cosine_score.max(-1)[0]
        #! HS
        

        energy_score = torch.logsumexp(logits, dim=-1)

        ood_keys = {
            'softmax': softmax_score.tolist(),
            'maha': maha_score.tolist(),
            'cosine': cosine_score.tolist(),
            'energy': energy_score.tolist(),
            'maha_acc': correct, 
        }
        return ood_keys

    def prepare_ood(self, final_rep, dataloader=None):
        self.bank = None
        self.label_bank = None
        for batch in dataloader:
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            labels = batch['labels']
            
            b_dict = self.forward(**batch)
            if final_rep == 'last_cls':
                pooled = b_dict['last_cls']
            elif final_rep == 'global_projection':
                pooled = b_dict['global_projection']
            elif final_rep == 'last_cls+global_projection':
                pooled = torch.cat([b_dict['global_projection'],b_dict['last_cls']], dim=1)

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
