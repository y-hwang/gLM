# Copyright (c) President and Fellows of Harvard College 2023.

# This source code is licensed under the Academic and Non-Commercial Research Use Software License 
# found in the LICENSE file in the root directory of this source tree.

# usage: python operon_logreg.py -d ../data/ecoli_operon_data/batched_ecoli/ -m ../model/glm.bin --annot ../data/ecoli_operon_data/operon.annot
import torch
import sys
from torch import nn
from gLM import *
from glm_utils import *
from transformers import RobertaConfig
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import random
import os
import numpy as np
import argparse
import pathlib
import matplotlib.pyplot as plt
import datetime
import logging
import scipy.stats
import scipy.special
import pickle as pk
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import classification_report

random.seed(1)
torch.cuda.empty_cache()

def read_annot_file(annot_path):
    f = open(annot_path)
    annot_dict = {}
    for line in f:
        l = line.strip().split("\t")
        annot_dict[int(l[0])] = (l[1],l[2],l[3],l[4])

    return annot_dict

def get_operons(ids, annot_dict):
    operons = []
    for i in ids:
        if i != 0:
            operons.append(annot_dict[i][2])
    return operons

def get_annotations(ids, annot_dict):
    annots = []
    for i in ids:
        if i != 0:
            annots.append(annot_dict[i][0])
    return annots

def get_descriptions(ids, annot_dict):
    descs = []
    for i in ids:
        if i != 0:
            descs.append(annot_dict[i][1])
    return descs
def same_operon(i, j, annot_dict):
    
    if i == j:
        return False
    if annot_dict[i][2] == "None":
        return False
    elif annot_dict[j][2] == "None":
        return False
    else:
        return annot_dict[i][2] == annot_dict[j][2]

def draw_operon(ind,ids,annot_dict,all_attentions,predictor,max_cor_mat_inds,ax):
    prot_ids = ids[ind]
    annots = get_annotations(prot_ids, annot_dict)
    max_head =  all_attentions[ind,:len(annots),:len(annots),NHEADS*max_cor_mat_inds[0]+max_cor_mat_inds[1]].squeeze()
    contacts = all_attentions[ind]
    contacts  = contacts.reshape(900,190)
    output_mat = np.zeros((3,len(annots)-1))
    
    for ind_a, a in enumerate(prot_ids):
        for ind_b, b in enumerate(prot_ids):
            if a > 0 and b > 0 and ind_b > ind_a and ind_b - ind_a < 2 and ind_b != ind_a :
                pred = predictor.predict(np.expand_dims(contacts[30*ind_a+ind_b],axis =0))
                output_mat[1][ind_a] = max_head[ind_a][ind_b]
                if same_operon(a,b,annot_dict):
                    output_mat[2][ind_a] = 1 
                if pred == 1 and same_operon(a,b,annot_dict):    #true positive
                    output_mat[0][ind_a] = 1                     
                elif pred == 1 and not same_operon(a,b,annot_dict): #false positive
                    output_mat[0][ind_a] = -1   
    ylabels = ["LogReg", "L"+str(max_cor_mat_inds[0]+1)+" -H"+str(max_cor_mat_inds[1]+1)+" raw", "Operons"]
    ax.pcolor(output_mat,cmap='RdBu',vmax = 1.5, vmin=-1.5)
    ax.set_xticks(np.arange(output_mat.shape[1]+1), minor=False)
    ax.set_yticks(np.arange(output_mat.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(annots,rotation=90, minor=False)
    ax.set_yticklabels(ylabels, minor=False)
    ax.set_aspect('equal')
    return None

def visualize_operons(all_contacts,ids,annot_dict,logging):
    X = []
    y = []
    contact_mats = []
    # populate operon matrices
    for i, multi_head_contacts in enumerate(all_contacts):
        prot_ids = ids[i]
        annots = get_annotations(prot_ids, annot_dict)
        prot_ids = ids[i]
        contact_mat = np.zeros((len(annots),len(annots)))
        for ind_a, a in enumerate(prot_ids):
            for ind_b, b in enumerate(prot_ids):
                if ind_a == ind_b:
                    continue
                if a > 0 and b >0 and a > b:
                    X.append(multi_head_contacts[ind_a,ind_b,:])
                    if same_operon(a,b, annot_dict):
                        contact_mat[ind_a][ind_b] = 1
                        contact_mat[ind_b][ind_a] = 1
                        y.append(1)
                    else:
                        y.append(0)
        contact_mats.append(contact_mat)
                   
    X = np.array(X)
    n,_= X.shape
    X_new = X.reshape(n,NLAYERS,NHEADS)
    _,l,a = X_new.shape
    cor_mat = np.zeros((NLAYERS,NHEADS))
    pval_mat = np.zeros((NLAYERS, NHEADS), dtype=np.float64)
    fig, axs = plt.subplots(NLAYERS,NHEADS,figsize=(100, 190))
    for i in range(l):
        for j in range(a):
            corr = scipy.stats.pearsonr(X_new[:,i,j], y)
            cor_mat[i,j] = corr[0]
            pval_mat[i, j] = corr[1]/190
            head =  all_contacts[0,:len(annots),:len(annots),NHEADS*i+j].squeeze()
            axs[i][j].pcolor(head,cmap='Blues')
            axs[i][j].axis('off')
    logging.info("max_corr: "+str(np.max(cor_mat)))
    filename = FIGURES_DIR+"heads.png"
    fig.savefig(filename)
    plt.close(fig) # all heads, supplemental figure 
    plt.rcParams.update({'font.size': 8})
    fig = plt.figure(figsize=(12,12))
    ax1 = plt.subplot2grid((3,3),(0,0), rowspan = 2)
    ax2 = plt.subplot2grid((3,3),(0,1), rowspan =1, colspan=2)
    ax3 = plt.subplot2grid((3,3),(1,1), rowspan =1, colspan=2)
    ax4 = plt.subplot2grid((3,3),(2,1), rowspan =1, colspan=2)
    ax5 = plt.subplot2grid((3,3),(2,0), rowspan =1, colspan=1)
    heatmap1=ax1.pcolor(cor_mat,cmap='Blues') #correlation heatmap
    max_cor_mat_inds = np.unravel_index(np.argmax(cor_mat), np.array(cor_mat).shape)

    ax1.set_xticks(np.arange(cor_mat.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(cor_mat.shape[0]) + 0.5, minor=False)
    ax1.set_xticklabels(np.arange(1,cor_mat.shape[1]+1),rotation=90, minor=False)
    ax1.set_yticklabels(np.arange(1,cor_mat.shape[0]+1), minor=False)
    ax1.set_xlabel("Heads")
    ax1.set_ylabel("Layers")
    ax1.set_title("Correlation coefficient")
    plt.colorbar(heatmap1,location="bottom",ax=ax1,pad=0.1, fraction=0.05)
   
    # perform logistic regression
    X = []
    y = []
    for i, multi_head_contacts in enumerate(all_contacts):
        prot_ids = ids[i]
        annots = get_annotations(prot_ids, annot_dict)
        prot_ids = ids[i]
        
        for ind_a, a in enumerate(prot_ids):
            for ind_b, b in enumerate(prot_ids):
                if a > 0 and b > 0 and ind_b > ind_a and ind_b - ind_a < 2 and ind_b != ind_a :
                    X.append(multi_head_contacts[ind_a,ind_b,:].ravel())
                    if same_operon(a,b,annot_dict):                            
                        y.append(1)
                    else:
                        y.append(0)

    k_fold = KFold(n_splits=5, shuffle=True, random_state=16)
    predictor = LogisticRegression(random_state=16, max_iter=1000)
    y_real = []
    y_proba = []
    X = np.array(X)
    y = np.array(y)
    for i, (train_index, test_index) in enumerate(k_fold.split(X)):
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        predictor.fit(Xtrain, ytrain)
        pred_proba = predictor.predict_proba(Xtest)
        precision, recall, _ = precision_recall_curve(ytest, pred_proba[:,1])
        ax5.step(recall, precision)
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    ypred = predictor.predict(Xtest)
    print(classification_report(ytest, ypred))
    lab = 'mean average precision=%.2f' % (auc(recall, precision))
    ax5.step(recall, precision, label=lab, lw=2, color='black')
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.legend(loc='lower left', fontsize='small')
    
    # pick random samples for visualization
    samples = random.sample(range(0,144),3)   
    ind1 = samples[0]
    ind2 =samples[1]
    ind3 =samples[2]
    
    draw_operon(ind1,ids,annot_dict,all_contacts,predictor,max_cor_mat_inds,ax2)
    draw_operon(ind2,ids,annot_dict,all_contacts,predictor,max_cor_mat_inds,ax3)
    draw_operon(ind3,ids,annot_dict,all_contacts,predictor,max_cor_mat_inds,ax4)
    
    filename = FIGURES_DIR+"figure4.png"
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def examine_model(logging, data_dir, model, device, annot_dict):
    f_list = os.listdir(data_dir)
    test_pkls=[]
    for pkl_f in f_list:
        test_pkls.append(str(os.path.join(data_dir,pkl_f)))

    """examines the model"""
    torch.cuda.empty_cache()
    logging.info("testing model...")
    scaler = None
    input_embs = []
    hidden_embs = []
    all_contacts = []
    all_prot_ids = []
    predicted_embeds_masked = []
    all_probs = []
    if HALF:
        logging.info("testing using a mixed precision model")
        scaler = torch.cuda.amp.GradScaler()
    for pkl_f in test_pkls:
        pickle_file =  open(pkl_f, 'rb')
        dataset = pk.load(pickle_file)
        id_to_label = {}
        for seq in dataset:
            labels =seq['label_embeds']
            prot_ids = seq['prot_ids']
            for ind, i in enumerate(prot_ids):
                if i !=0:
                    id_to_label[i] = labels[ind]
       
        if B_SIZE < len(dataset):
            loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=False, drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=False)
        loop = tqdm(loader, leave=True)
        for batch in loop:
            # pull all tensor batches required for testing
            inputs_embeds= batch['embeds'].type(torch.FloatTensor)
            attention_mask = batch['attention_mask'].type(torch.FloatTensor)
            mask = torch.zeros(attention_mask.shape) #nothing is masked
            masked_tokens = (mask==1) & (attention_mask != 0)
            masked_tokens = torch.unsqueeze(masked_tokens, -1)
            masked_tokens = masked_tokens.to(device)
            inputs_embeds = inputs_embeds.to(device)
            inputs_embeds  = torch.where(masked_tokens, -1.0, inputs_embeds)
            attention_mask = attention_mask.to(device)
            labels = batch['label_embeds'].type(torch.FloatTensor)
            labels = labels.to(device)
            input_embs.append(inputs_embeds.cpu().detach().numpy())
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # process
                    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels = labels, masked_tokens = masked_tokens, output_attentions = False)
                    # extract loss
                    last_hidden_states = outputs.last_hidden_state
                    hidden_embs.append(last_hidden_states.cpu().detach().numpy())
                    prot_ids = batch['prot_ids']
                    all_contacts.append(outputs.contacts.cpu().detach().numpy())
                    all_prot_ids.append(prot_ids)
                    logits_all_preds = outputs.logits_all_preds
                    all_preds = logits_all_preds[masked_tokens.squeeze(-1)]
                    predicted_probs = outputs.probs
                    raw_probs = predicted_probs.view(-1,4)
                    softmax = nn.Softmax(dim=1)
                    probs = softmax(raw_probs)
                    predicted_embeds_masked.append(all_preds.cpu().detach().numpy())
                    all_probs.append(probs.cpu().detach().numpy())
    input_embs = np.concatenate(input_embs, axis = 0)
    hidden_embs = np.concatenate(hidden_embs, axis=0)
    all_prot_ids = np.concatenate(all_prot_ids, axis = 0 )
    all_contacts = np.concatenate(all_contacts, axis= 0)
    visualize_operons(all_contacts,all_prot_ids,annot_dict,logging)
    return None

parser = argparse.ArgumentParser(description = "outputs glm embeddings")
parser.add_argument('-d','--data_dir', type=pathlib.Path, help='batched data directory')
parser.add_argument('-id', '--id_path',  help='path to prot_index_dict.pkl file', default = None)
parser.add_argument('-m','--model_path', help="path to pretrained model, glm.bin")
parser.add_argument('-b','--batch_size', type=int, help='batch_size', default = 100)
parser.add_argument('-o', '--output_path', type=str, help='inference output directory', default = None)
parser.add_argument('--attention',action='store_true', help='output attention matrices ', default = False)
parser.add_argument('--hidden_size', type=int, help='hidden size', default = 1280)
parser.add_argument('-n', '--ngpus', type=int, help='number of GPUs to use',  default = 1)
parser.add_argument('--annot_path',help='path to operon annotation file', default = None)
parser.add_argument('-a', '--all_results',action='store_true', help='output all results including plm_embs/glm_embs/prot_ids/outputs/output_probabilitess', default = False)

# load all arguments 
args = parser.parse_args()
if args.data_dir is None :
    parser.error('--data_dir must be specified')
if args.model_path is None :
    parser.error('--model must be specified')
if args.annot_path is None :
    parser.error('--annot_path must be specified')
# define all parameters
data_dir = args.data_dir
ngpus = args.ngpus
model_path = args.model_path
num_pred = 4
max_seq_length = 30 

output_path = args.output_path
pos_emb = "relative_key_query"
pred_probs = True
id_path = args.id_path
annot_path = args.annot_path
# define all global variables 
HIDDEN_SIZE = args.hidden_size
B_SIZE = args.batch_size # batch size
HALF = True
EMB_DIM = 1281
NUM_PC_LABEL = 100
ATTENTION = args.attention
ALL_RESULTS = args.all_results
NHEADS = 10
NLAYERS = 19
e = datetime.datetime.now()
if output_path == None:
    output_path = "operon"+ e.strftime("-%d-%m-%Y-%H:%M:%S")
if not os.path.exists(output_path):
    os.mkdir(output_path)

logfile_path = output_path+"/info.log"

if not os.path.exists(output_path+"/figures"):
    os.mkdir(output_path+"/figures")
FIGURES_DIR = output_path+"/figures/"


# log is stored in both logfile and streamed to stdout
# begin logging 
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=[
        logging.FileHandler(logfile_path),
        logging.StreamHandler()])
logging.info("output folder: " +output_path)
logging.info("log file is located here: " +logfile_path)
string_of_command = f"{' '.join(sys.argv)}"
logging.info("command: " + string_of_command)

# populate config 
config = RobertaConfig(
    max_position_embedding = max_seq_length,
    hidden_size = HIDDEN_SIZE,
    num_attention_heads = NHEADS,
    type_vocab_size = 1,
    tie_word_embeddings = False,
    num_hidden_layers = NLAYERS,
    num_pc = NUM_PC_LABEL, 
    num_pred = num_pred,
    predict_probs = pred_probs,
    emb_dim = EMB_DIM,
    output_attentions=True,
    output_hidden_states=True,
    position_embedding_type = pos_emb,
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model =  gLM(config)
model.load_state_dict(torch.load(model_path, map_location=device),strict=False)
model.eval()
if ngpus>1:
    model = torch.nn.DataParallel(model)
model.to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

logging.info("batch_size: "+str(B_SIZE))

if id_path != None:
    id_dict = pk.load(open(id_path, "rb"))
else:
    id_dict = None
annot_dict= read_annot_file(annot_path)
with torch.no_grad():
    examine_model(logging, data_dir, model, device, annot_dict)
