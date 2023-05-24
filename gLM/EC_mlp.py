# Copyright (c) President and Fellows of Harvard College 2023.

# This source code is licensed under the Academic and Non-Commercial Research Use Software License 
# found in the LICENSE file in the root directory of this source tree.

# usage: python EC_mlp.py <save/split/train> <data_path> <plm/plmglm> 
# step 1: python EC_mlp.py save inference/results/ plm or python EC_mlp.py save inference/results/ plmglm
# step 2: python EC_mlp.py split EC_dat_mlp.plm.pkl plm or python EC_mlp.py split EC_dat_mlp.plmglm.pkl plmglm 
# step 3: python EC_mlp.py train train_split.plm.pkl plm or python EC_mlp.py split train_split.plmglm.pkl plmglm 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pickle as pk
from multiprocessing import Pool
from sklearn.preprocessing import label_binarize
from tqdm import tqdm  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,precision_recall_curve,auc,average_precision_score

torch.manual_seed(12345)
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        self.layer_1 = nn.Linear(num_feature, 2048)
        self.layer_2 = nn.Linear(2048, 1024)
        self.layer_3 = nn.Linear(1024, 514)
        self.layer_out = nn.Linear(514, num_class) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(2048)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.batchnorm3 = nn.BatchNorm1d(514)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        return x

class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def read_EC_file(EC_path):
    prot_to_ec = defaultdict()
    EC_f = open(EC_path, "r")
    ec_ids = []
    prots = set()
    ecs = set()
    for line in tqdm(EC_f.readlines(), len(EC_f.readlines())):
        l = line.strip().split("\t")
        ec = l[0]
        level4 = ec.split(".")[3]
        id = int(l[1][4:])
        if id not in prots and level4 != "-": 
            prots.add(id)
            ec_ids.append(ec)
    ec_counter = Counter(ec_ids)
    ecs = [ec for ec, c in ec_counter.items() if c > 50] # remove ECs with less than 50 datapoints
    id_file = open("ec_index.txt", "w")
    for ec in ecs:
        id_file.write(ec+"\n")
    id_file.close()
    print("ec_index file saved")
    ec_set = set(ecs)
    EC_f.close()
    EC_f = open(EC_path, "r")
    for line in tqdm(EC_f.readlines(), len(EC_f.readlines())):
        l = line.strip().split("\t")
        ec = l[0]
        level4 = ec.split(".")[3]
        id = int(l[1][4:])
        if id not in prot_to_ec.keys() and level4 != "-" and ec in ec_set:
            prot_to_ec[id] = ecs.index(ec)
    prot_to_ec_file = open("prot_to_ec.pkl", "wb")
    pk.dump(prot_to_ec, prot_to_ec_file)
    prot_to_ec_file.close()
    return prot_to_ec

def extract_ec_data_from_results_pkls(result_pkl_path,added_pids, type):
    result_pkl_f = open(result_pkl_path, "rb")
    results = pk.load(result_pkl_f)
    prot_ids = results['all_prot_ids']
    filter_inds = []
    ecs = []
    for i, pid in enumerate(prot_ids):
        if pid in PROT_TO_EC.keys() and pid not in added_pids:
            ec = PROT_TO_EC[pid]
            ecs.append(ec)
            filter_inds.append(i)
            added_pids.add(pid)

    input_embs = results['input_embs']
    input_embs_new=np.take(input_embs,filter_inds,axis=0)

    hidden_embs = results['hidden_embs']
    hidden_embs_new=np.take(hidden_embs,filter_inds,axis=0)

    if type == "plmglm":
        label_hidden_concat = np.concatenate((input_embs_new[:,:-1], hidden_embs_new), axis=1)
        return label_hidden_concat, ecs, added_pids
    elif type == "plm":  #remove orientation feature
        return input_embs_new[:,:-1], ecs, added_pids
        
def extract_ec_data_from_results_pkls_sequential(results_pkls, type):
    embs =[]
    ecs =[]
    added_pids = set()
    for result_pkl in tqdm(results_pkls, total=len(results_pkls)):
        h,e,pid = extract_ec_data_from_results_pkls(result_pkl, added_pids, type)
        embs.extend(h)
        ecs.extend(e)
        added_pids = pid
    embs = np.array(embs)
    ecs = np.array(ecs)
    return  embs, ecs

def save_data(results_dir, type):
    f_list = os.listdir(results_dir)
    results_pkls = []
    
    for pkl_f in f_list:
        if pkl_f.startswith('train'):
            results_pkls.append(str(os.path.join(results_dir,pkl_f)))
    first_100 = results_pkls[:10]
    embs, ecs= extract_ec_data_from_results_pkls_sequential(first_100, type)
    ec_dat_f = open("EC_dat_mlp."+type+".pkl", "wb")
    pk.dump([embs,ecs], ec_dat_f)
    ec_dat_f.close()
    return None

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc

def split_data(data_path, type):
    dat = pk.load(open(data_path,"rb"))
    X = dat[0]
    y = dat[1]
    ec_counter = Counter(y)
    y_filter = [ec for ec, c in ec_counter.items() if c < 3]
    y_new = np.delete(y,np.where(np.isin(y,y_filter))[0],axis =0)
    X_new = np.delete(X,np.where(np.isin(y,y_filter))[0],axis =0)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X_new, y_new, test_size=0.2, stratify=y_new, random_state=69)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)
    train_dat_f = open("train_split."+type+".pkl","wb")
    pk.dump([X_train, y_train, X_val, y_val,X_test, y_test],train_dat_f )
    train_dat_f.close()    

def train(data_path, type):
    X_train, y_train, X_val, y_val,X_test, y_test = pk.load(open(data_path,"rb"))
    if type == "plmglm":
        nfeatures = 2560
    else:
        nfeatures =1280
    print("successfully loaded data")

    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    print("made datasets")
    target_list = []
    for _, t in train_dataset:
        target_list.append(t)
    target_list = torch.tensor(target_list)
    class_count = [0]*(max(target_list).item()+1)
    y_counter = Counter(y_train)
    for key,val in y_counter.items():
        class_count[key] = val
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )
    print("finished weighted sampler")

    EPOCHS = 88
    BATCH_SIZE = 5000
    LEARNING_RATE = 0.001
    NUM_FEATURES = nfeatures
    NUM_CLASSES = len(class_count)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            sampler=weighted_sampler
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }
    print("Begin training.")
    for e in range(1, EPOCHS+1):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        loop = tqdm(train_loader, leave=True)
        for X_train_batch, y_train_batch in loop:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            loop.set_description(f'Epoch {e}')
            loop.set_postfix(train_loss=train_loss.item(),accuracy=train_acc.item())
            
        # VALIDATION    
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.eval()
            loop = tqdm(val_loader, leave=True)
            for X_val_batch, y_val_batch in loop:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
                loop.set_description(f'Epoch {e}')
                loop.set_postfix(train_loss=val_loss.item(),accuracy=val_acc.item())
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))          
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

    # Create dataframes
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    
    # Plot the dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
    sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
    plt.savefig("acc_curve."+str(type+".png")) # train and validation accuracy curves
    plt.close(fig)
    y_pred_list = []
    y_pred_proba = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in tqdm(test_loader):
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
            y_pred_proba.append(y_test_pred.cpu().numpy())
    y_pred_list= np.concatenate(y_pred_list,axis =0)
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_pred_proba = np.concatenate(y_pred_proba,axis = 0)
    report= classification_report(y_test, y_pred_list, output_dict=True)
    dataframe = pd.DataFrame(report).transpose()
    dataframe.to_csv("report."+type+".tsv",sep="\t") # save per class classification report 

    precision = dict()
    recall = dict()
    average_precision = dict()
    y_bin = label_binarize(y_test, classes=list(range(max(y_train)+1)))
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_bin.ravel(), np.array(y_pred_proba).ravel()
    )
    average_precision["micro"] = average_precision_score(y_bin, y_pred_proba, average="micro")
    dat = (precision, recall,average_precision)
    pk.dump(dat,open("PR_dat."+type+".pkl", "wb"))
    lab = type+' mAP =%.2f' % (average_precision["micro"])
    plt.plot(precision["micro"], recall["micro"], marker='.', label=lab)
    plt.xlabel('Recall("micro")')
    plt.ylabel('Precisio("micro")')
    plt.legend(loc='lower left', fontsize='small')
    plt.savefig("PR_curve."+type+".png") # save PR curve figure
    save_path = "EC_MLP."+str(type)+".pt"
    torch.save(model.state_dict(), save_path) # save model 

data_path = sys.argv[2]
type = sys.argv[3]
if sys.argv[1] == "save":
    EC_path = "MGnify_EC.m8"
    PROT_TO_EC = read_EC_file(EC_path) 
    save_data(data_path, type)
elif sys.argv[1] == "split":
    split_data(data_path, type)
elif sys.argv[1] == "train":
    train(data_path, type)
else:
    print("You must specify operation type: save or split or train")

if type != "plm" and type != "plmglm":
    print("You must specify data type: plm or plmglm")


    