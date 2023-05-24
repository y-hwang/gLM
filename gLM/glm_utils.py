# Copyright (c) President and Fellows of Harvard College 2023.

# This source code is licensed under the Academic and Non-Commercial Research Use Software License 
# found in the LICENSE file in the root directory of this source tree.

from collections import defaultdict
import torch
from tqdm import tqdm
from collections import defaultdict
import os
import h5py
import numpy as np
from multiprocessing import Pool

def read_cluster_dict(cluster_file):
    prot_to_cluster = defaultdict()
    # total_l = 2445772679 # total number of proteins 
    # total_l = 623796864 # total number of protein clusters
    total_l = 954752 
    with open(cluster_file, 'r' ) as f:
        for line in tqdm(f,total= total_l):
            l = line.strip().split()
            prot = int(l[0][4:])
            cluster = int(l[1][4:])
            prot_to_cluster[prot] = cluster
    return prot_to_cluster
def read_embedding_file(hfile):
    print("start reading")
    hf = h5py.File(hfile)
    emb_dict = defaultdict(list)
    for k,v in hf.items():
            key = k.strip().split()[0]
            value = v[:]
            emb_dict[int(key[4:])] = np.array(value)
        

    return emb_dict

def read_embeddings_pool(emb_dir, num_cpus):
    emb_dict = defaultdict(list)
    f_list = os.listdir(emb_dir)
    hfs = []
    results = []
    for emb_f in f_list: 
        hfs.append(str(os.path.join(emb_dir,emb_f)))
    
    with Pool(num_cpus) as p:
        results= p.map(read_embedding_file, hfs)
    for result in results:
        emb_dict.update(result)
    return emb_dict
def read_embeddings(emb_dir):
    emb_dict = defaultdict(list)
    f_list = os.listdir(emb_dir)
    for emb_f in tqdm(f_list, total=len(f_list)):
        hf = h5py.File(str(os.path.join(emb_dir,emb_f)))
        for k,v in hf.items():
            key = k.strip().split()[0]
            value = v[:]
            emb_dict[int(key[4:])] = np.array(value)
    return emb_dict
def read_embeddings_base(emb_dir):
    emb_dict = defaultdict(list)
    f_list = os.listdir(emb_dir)
    for emb_f in tqdm(f_list, total=len(f_list)):
        hf = h5py.File(str(os.path.join(emb_dir,emb_f)))
        for k,v in hf.items():
            key = k.strip().split()[0]
            value = v[:]
            emb_dict[key] = np.array(value)
    return emb_dict
def contigs_to_prots_dict(contigs_to_prots_f, max_seq_length, hidden_size):
    prot_to_contig_dict={}
    contigs_prots = []
    for line in open(contigs_to_prots_f, 'r'):
        l =  line.strip().split('\t')
        contig = l[0]
        prots = l[1].split(";")
        num_prots = len(prots)
        num_segs = num_prots // max_seq_length + 1
        for i in range(num_segs):
            contig_id = contig+"_"+str(i)
            if i == num_segs-1: # last segment in contig
                prots_in_seg = prots[i*max_seq_length:]
                
            else: 
                prots_in_seg = prots[i*max_seq_length:(i+1)*max_seq_length]
            if len(prots_in_seg) > max_seq_length/2:

                # sequence length less than half of the max sequence length, so don't bother batching
                contigs_prots.append((contig_id,prots_in_seg))
                for prot in prots_in_seg:
                    prot_to_contig_dict[int(prot[4:])] = contig_id
    return contigs_prots, prot_to_contig_dict


def contigs_to_prots(contigs_to_prots_f, max_seq_length):
    contigs_prots = []
    for line in open(contigs_to_prots_f, 'r'):
        l =  line.strip().split('\t')
        
        contig = l[0]
        prots = l[1].split(";")
        num_prots = len(prots)
        num_segs = num_prots // max_seq_length + 1
        for i in range(num_segs):
            contig_id = contig+"_"+str(i)
            if i == num_segs-1: # last segment in contig
                prots_in_seg = prots[i*max_seq_length:]
            else: 
                prots_in_seg = prots[i*max_seq_length:(i+1)*max_seq_length]
            if len(prots_in_seg) > max_seq_length/2:
                # sequence length less than half of the max sequence length, so don't bother batching
                contigs_prots.append((contig_id,prots_in_seg))
    return contigs_prots

def batch_contigs_base(contigs_prots, emb_dict, prot_to_cluster, max_seq_length):
    batch = []
    for contig, prots in contigs_prots:
        b = {}
        num_prots = len(prots)
        embeds = np.zeros((max_seq_length, 1280))
        prot_ids = [""]*max_seq_length
        attention_mask = np.concatenate([np.ones(num_prots), np.zeros(max_seq_length-num_prots)])
        for i in range(num_prots):
            prot_index = i 
            key = prots[prot_index] 
            numeric_key = key
            if numeric_key not in emb_dict.keys():
                if numeric_key not in prot_to_cluster.keys():
                    print(key)
                    embeds[i] = MISSING_TOK
                else:
                    new_numeric_key = prot_to_cluster[numeric_key]
                    embeds[i] = emb_dict[new_numeric_key]
                    
                
            else:
                embeds[i] = emb_dict[numeric_key]
            prot_ids[i] = numeric_key
        b['prot_ids'] = prot_ids
        b['embeds'] = embeds
        b['attention_mask'] = attention_mask
        batch.append(b)
    
    
    batch = np.array(batch)
    return batch

def batch_contigs_ori(contigs_prots, emb_dict, prot_to_cluster, max_seq_length, hidden_size):
    batch = []
    all_embeds = []
    MISSING_TOK = np.ones((hidden_size-20,))
    for contig, prots in contigs_prots:
        num_prots = len(prots) 
        for i in range(num_prots):
            prot_index = i 
            key = prots[prot_index]
            
            numeric_key = int(key[5:])
            # this is where gene orientation is encoded 
            ori = key[0]
            if numeric_key not in emb_dict.keys():
                if numeric_key not in prot_to_cluster.keys():
                    print(key)
                    if ori == "-":
                        embeds = np.append(MISSING_TOK,[0]*20)
                        
                    else:
                        embeds = np.append(MISSING_TOK,[1]*20)
                else:
                    new_numeric_key = prot_to_cluster[numeric_key]
                    if ori == "-":
                        embeds= np.append(emb_dict[new_numeric_key], [0]*20)
                    else:
                        embeds= np.append(emb_dict[new_numeric_key], [1]*20)
            else:
                if ori == "-":
                    embeds= np.append(emb_dict[numeric_key], [0]*20)
                else:
                    embeds= np.append(emb_dict[numeric_key], [1]*20)
            
            all_embeds.append(embeds)
    
    all_embeds_mean = np.mean(all_embeds, axis =0)
    all_embeds_std = np.std(all_embeds, axis =0)
    all_embeds_norm = (all_embeds-all_embeds_mean)/all_embeds_std
    embed_ind =0
    for contig, prots in contigs_prots:
        b = {}
        num_prots = len(prots)
        embeds = np.zeros((max_seq_length, hidden_size))
        prot_ids = np.zeros(max_seq_length, dtype=int)
        attention_mask = np.concatenate([np.ones(num_prots), np.zeros(max_seq_length-num_prots)])
        for i in range(num_prots):
            prot_index = i 
            embeds[i] = all_embeds_norm[embed_ind]
            embed_ind +=1
            prot_ids[i] = int(numeric_key)
        b['prot_ids'] = prot_ids
        b['embeds'] = embeds
        b['attention_mask'] = attention_mask
        batch.append(b)

    batch = np.array(batch)
    return batch

def batch_contigs_tmp(contigs_prots, emb_dict, prot_to_cluster, max_seq_length, hidden_size):
    batch = []
    all_embeds = []
    MISSING_TOK = np.ones((hidden_size,))
    for contig, prots in contigs_prots:
        num_prots = len(prots) 
        for i in range(num_prots):
            prot_index = i 
            key = prots[prot_index]
            
            numeric_key = int(key[5:])
            # this is where gene orientation is encoded 
            ori = key[0]
            if numeric_key not in emb_dict.keys():
                if numeric_key not in prot_to_cluster.keys():
                    print(key)
                    if ori == "-":
                        embeds = MISSING_TOK
                        embeds[hidden_size-1] = 0
                        
                    else:
                        embeds = MISSING_TOK
                        embeds[hidden_size-1] = 1
                else:
                    new_numeric_key = prot_to_cluster[numeric_key]
                    if ori == "-":
                        embeds=emb_dict[new_numeric_key]
                        embeds[hidden_size-1] = 0
                    else:
                        embeds=emb_dict[new_numeric_key]
                        embeds[hidden_size-1] = 1
            else:
                if ori == "-":
                    embeds=emb_dict[numeric_key]
                    embeds[hidden_size-1] = 0
                else:
                    embeds=emb_dict[numeric_key]
                    embeds[hidden_size-1] = 1
            
            all_embeds.append(embeds)
    
    all_embeds_mean = np.mean(all_embeds, axis =0)
    all_embeds_std = np.std(all_embeds, axis =0)
    all_embeds_norm = (all_embeds-all_embeds_mean)/all_embeds_std
    embed_ind =0
    for contig, prots in contigs_prots:
        b = {}
        num_prots = len(prots)
        embeds = np.zeros((max_seq_length, hidden_size))
        prot_ids = np.zeros(max_seq_length, dtype=int)
        attention_mask = np.concatenate([np.ones(num_prots), np.zeros(max_seq_length-num_prots)])
        for i in range(num_prots):
            prot_index = i 
            embeds[i] = all_embeds_norm[embed_ind]
            embed_ind +=1
            prot_ids[i] = int(numeric_key)
        b['prot_ids'] = prot_ids
        b['embeds'] = embeds
        b['attention_mask'] = attention_mask
        batch.append(b)

    batch = np.array(batch)
    return batch

def batch_contigs(contigs_prots, emb_dict, prot_to_cluster, max_seq_length, hidden_size):
    batch = []
    all_embeds = []
    MISSING_TOK = np.ones((hidden_size,))
    for contig, prots in contigs_prots:
        num_prots = len(prots) 
        embeds = np.zeros((max_seq_length, hidden_size))
        for i in range(num_prots):
            prot_index = i 
            key = prots[prot_index]
            
            numeric_key = int(key[4:])
            if numeric_key not in emb_dict.keys():
                if numeric_key not in prot_to_cluster.keys():
                    print(key)
                    embeds[i] = MISSING_TOK
                else:
                    new_numeric_key = prot_to_cluster[numeric_key]
                    embeds[i] = emb_dict[new_numeric_key]  
            else:
                embeds[i] = emb_dict[numeric_key]
            all_embeds.append(embeds[i])
    
    all_embeds_mean = np.mean(all_embeds, axis =0)
    all_embeds_std = np.std(all_embeds, axis =0)
    all_embeds_norm = (all_embeds-all_embeds_mean)/all_embeds_std
    embed_ind =0
    for contig, prots in contigs_prots:
        b = {}
        num_prots = len(prots)
        embeds = np.zeros((max_seq_length, hidden_size))
        prot_ids = np.zeros(max_seq_length, dtype=int)
        attention_mask = np.concatenate([np.ones(num_prots), np.zeros(max_seq_length-num_prots)])
        for i in range(num_prots):
            prot_index = i 
            embeds[i] = all_embeds_norm[embed_ind]
            embed_ind +=1
            prot_ids[i] = int(numeric_key)
        b['prot_ids'] = prot_ids
        b['embeds'] = embeds
        b['attention_mask'] = attention_mask
        batch.append(b)

    batch = np.array(batch)
    return batch

def get_pfam_dict(pfam_path):
    pfam_f = open(pfam_path,'r')
    lines = pfam_f.readlines()
    pfam_dict = defaultdict()
    for l in lines:
        id = int(l.strip().split()[0][4:])
        pfam_id = l.strip().split()[1]
        pfam_dict[id] = pfam_id
    
    return pfam_dict

def get_annots(ids,pfam_dict):
    annots = []
    for i in ids:
        if i in pfam_dict.keys():
            annots.append(pfam_dict[i])

        else:
            annots.append("unlabelled")
    return annots

def get_annot_dict( annotation_path):
    annot_f = open(annotation_path, 'r')
    lines = annot_f.readlines()
    annot_dict = defaultdict()
    for l in lines: 
        elems = l.strip().split('\t')
        id = elems[0]
        if id.startswith("MGYP"):
            id = int(id[4:])
        annot = elems[8]

        annot = " ".join(annot.split(" ")[1:])
        annot = annot.split("n=")[0]
        annot_dict[id] = annot
    return annot_dict
def batch_to_encodings(batch):
    embeds = [x['embeds'] for x in batch]
    prot_ids = [x['prot_ids'] for x in batch]

    prot_ids = torch.from_numpy(np.stack(prot_ids))
    #import pdb; pdb.set_trace()
    labels = torch.from_numpy(np.stack(embeds, axis=0)).float()
    # [batch_size, max_seq_len, 1280]
    input_embeds = labels.detach().clone()
    # [batch_size, max_seq_len]
    attentions = [x['attention_mask'] for x in batch]
    attention_mask = torch.from_numpy(np.stack(attentions, axis=0)).float()

    
    encodings = {'input_embeds': input_embeds, 'attention_mask': attention_mask, 'labels': labels, 'prot_ids':prot_ids}
    return encodings
def create_masked_dataset_base_test(batch,max_seq_len):
    #creates a dataset where every protein is masked once
    embeds = [x['embeds'] for x in batch]
    prot_ids = [x['prot_ids'] for x in batch]
    labels = torch.from_numpy(np.stack(embeds, axis=0)).float()
    # [batch_size, max_seq_len, 1280]
    input_embeds = labels.detach().clone()
    # [batch_size, max_seq_len]
    attentions = [x['attention_mask'] for x in batch]
    attention_mask = torch.from_numpy(np.stack(attentions, axis=0)).float()
    masked_tokens_test = []
    attention_mask_test = attention_mask.repeat((max_seq_len-10,1))
    labels_test = labels.repeat((max_seq_len-10,1,1))
    prot_ids_test = prot_ids*(max_seq_len-10)
    for i in range(5,max_seq_len-5):
        # [batch_size, max_seq_len]
        rand = torch.zeros(attention_mask.shape)
        rand[:,i] = 1
        # 1 if p<mask_prob and not padding.
        # [batch, max_seq_len]
        masked_tokens = (rand == 1) & (attention_mask != 0)
        # [batch, max_seq_len, 1]
        masked_tokens = torch.unsqueeze(masked_tokens, -1)
        masked_tokens_test.append(masked_tokens)
        # Zero out masked tokens.
        input_embeds  = torch.where(masked_tokens, 0.0, input_embeds)
        if i == 5:
            input_embeds_test = input_embeds
        else:
            input_embeds_test = torch.cat((input_embeds_test,input_embeds),dim=0)
    masked_tokens_test_cat = torch.cat(masked_tokens_test,dim=0)    
    encodings = {'input_embeds': input_embeds_test, 'attention_mask': attention_mask_test, 'labels': labels_test, 'masked_tokens': masked_tokens_test_cat, 'prot_ids':prot_ids_test}
    
    return encodings
def create_masked_dataset_base(batch, mask_prob):
    embeds = [x['embeds'] for x in batch]
    prot_ids = [x['prot_ids'] for x in batch]
    labels = torch.from_numpy(np.stack(embeds, axis=0)).float()
    # [batch_size, max_seq_len, 1280]
    input_embeds = labels.detach().clone()
    # [batch_size, max_seq_len]
    attentions = [x['attention_mask'] for x in batch]
    attention_mask = torch.from_numpy(np.stack(attentions, axis=0)).float()
    # [batch_size, max_seq_len]
    rand = torch.rand(attention_mask.shape)
    # 1 if p<mask_prob and not padding.
    # [batch, max_seq_len]
    masked_tokens = (rand < mask_prob) & (attention_mask != 0)
    # [batch, max_seq_len, 1]
    masked_tokens = torch.unsqueeze(masked_tokens, -1)
    # Zero out masked tokens.
    input_embeds  = torch.where(masked_tokens, 0.0, input_embeds)
    
    encodings = {'input_embeds': input_embeds, 'attention_mask': attention_mask, 'labels': labels, 'masked_tokens': masked_tokens, 'prot_ids':prot_ids}
    return encodings
def create_masked_dataset_test(batch, max_seq_len):
    embeds = [x['embeds'] for x in batch]
    prot_ids = [x['prot_ids'] for x in batch]

    prot_ids = torch.from_numpy(np.stack(prot_ids))
    labels = torch.from_numpy(np.stack(embeds, axis=0)).float()
    # [batch_size, max_seq_len, 1280]
    input_embeds = labels.detach().clone()
    # [batch_size, max_seq_len]
    attentions = [x['attention_mask'] for x in batch]
    attention_mask = torch.from_numpy(np.stack(attentions, axis=0)).float()
    masked_tokens_test = []
    attention_mask_test = attention_mask.repeat((max_seq_len-2,1))
    labels_test = labels.repeat((max_seq_len-2,1,1))
    prot_ids_test = prot_ids.repeat((max_seq_len-2,1))
    for i in range(1,max_seq_len-1):
        # [batch_size, max_seq_len]
        rand = torch.zeros(attention_mask.shape)
        rand[:,i] = 1
        # 1 if p<mask_prob and not padding.
        # [batch, max_seq_len]
        masked_tokens = (rand == 1) & (attention_mask != 0)
        # [batch, max_seq_len, 1]
        masked_tokens = torch.unsqueeze(masked_tokens, -1)
        masked_tokens_test.append(masked_tokens)
        # Zero out masked tokens.
        input_embeds  = torch.where(masked_tokens, 0.0, input_embeds)
        if i == 1:
            input_embeds_test = input_embeds
        else:
            input_embeds_test = torch.cat((input_embeds_test,input_embeds),dim=0)

    masked_tokens_test_cat = torch.cat(masked_tokens_test,dim=0)

    encodings = {'input_embeds': input_embeds_test, 'attention_mask': attention_mask_test, 'labels': labels_test, 'masked_tokens': masked_tokens_test_cat, 'prot_ids':prot_ids_test}
    return encodings
def create_masked_dataset(batch, mask_prob):
    embeds = [x['embeds'] for x in batch]
    prot_ids = [x['prot_ids'] for x in batch]

    prot_ids = torch.from_numpy(np.stack(prot_ids))
    #import pdb; pdb.set_trace()
    labels = torch.from_numpy(np.stack(embeds, axis=0)).float()
    # [batch_size, max_seq_len, 1280]
    input_embeds = labels.detach().clone()
    # [batch_size, max_seq_len]
    attentions = [x['attention_mask'] for x in batch]
    attention_mask = torch.from_numpy(np.stack(attentions, axis=0)).float()
    # [batch_size, max_seq_len]
    rand = torch.rand(attention_mask.shape)
    # 1 if p<mask_prob and not padding.
    # [batch, max_seq_len]
    masked_tokens = (rand < mask_prob) & (attention_mask != 0)
    # [batch, max_seq_len, 1]
    masked_tokens = torch.unsqueeze(masked_tokens, -1)
    # Zero out masked tokens.
    input_embeds  = torch.where(masked_tokens, -1.0, input_embeds)
    
    encodings = {'input_embeds': input_embeds, 'attention_mask': attention_mask, 'labels': labels, 'masked_tokens': masked_tokens, 'prot_ids':prot_ids}
    return encodings
