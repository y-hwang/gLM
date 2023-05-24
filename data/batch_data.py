# Copyright (c) President and Fellows of Harvard College 2023.

# This source code is licensed under the Academic and Non-Commercial Research Use Software License 
# found in the LICENSE file in the root directory of this source tree.

# usage: python batch_data.py <output.esm.embs.pkl> <contig_to_prots.tsv> <output_dir> 
import os
import sys
import numpy as np
import pickle as pk

emb_f = sys.argv[1]
contig_to_prots_f = sys.argv[2]
output_dir = sys.argv[3]
norm_factors_f = "norm.pkl"
pca_pkl_f = "pca.pkl"

contigs_to_prots_file = open(contig_to_prots_f, "r")
emb_file = open(emb_f, 'rb')
esm_embs = pk.load(emb_file)
embs = []
all_prot_ids = []
for key,val in esm_embs:
    all_prot_ids.append(key.split(" ")[0])
    embs.append(val)

emb_file.close()

embs = np.array(embs, dtype = np.float16)
norm_factors_file = open(norm_factors_f, "rb")
norm_factors = pk.load(norm_factors_file)
all_embeds_mean = norm_factors['mean']
all_embeds_std = norm_factors['std']
norm_factors_file.close()
normalized_embs = (embs-all_embeds_mean)/all_embeds_std
pickle_file_label =open(pca_pkl_f, 'rb')
PCA_LABEL = pk.load(pickle_file_label)
pickle_file_label.close()
all_labels = PCA_LABEL.transform(normalized_embs)
MAX_SEQ_LENGTH = 30 
EMB_DIM = 1281
LABEL_DIM = 100 
counter = 0
batch = []
index = 0
prot_ids = []
prot_to_id = {}
i = 1
count = 0
for line in contigs_to_prots_file:
    b={}
    embeds = np.zeros((MAX_SEQ_LENGTH, EMB_DIM), dtype=np.float16)
    label_embeds=np.zeros((MAX_SEQ_LENGTH, LABEL_DIM), dtype=np.float16)
    prot_ids =  np.zeros(MAX_SEQ_LENGTH, dtype =int)
    attention_mask = np.zeros(MAX_SEQ_LENGTH, dtype =int)
    elems = line.strip().split("\t")
    prots_in_contig = elems[1].split(";")
    for ind, prot_id in enumerate(prots_in_contig):
        ori = prot_id[0]
        pid = prot_id[1:]
        prot_index = all_prot_ids.index(pid)
        emb = normalized_embs[prot_index]
        label = all_labels[prot_index]
        if ori == "+":
            emb_o = np.append(emb,0.5)
            label_o = np.append(label,0.5)
        else:
            emb_o = np.append(emb,-0.5)
            label_o = np.append(label,-0.5)
        embeds[ind] = emb_o 
        label_embeds[ind] = label_o 
        prot_to_id[i] = pid
        
        prot_ids[ind] = i
        i+=1
        attention_mask[ind] = 1 
        count +=1
    b['prot_ids'] = prot_ids
    b['embeds'] = embeds
    b['label_embeds'] = label_embeds
    b['attention_mask'] = attention_mask
    batch.append(b)
batch = np.array(batch)
print(str(count)+" prots processed")
f = open(output_dir+"/batch.pkl", "wb")
pk.dump(batch,f)
f.close()

# save prot index to prot id dictionary as a pk file
f = open(output_dir+"/prot_index_dict.pkl", "wb")
pk.dump(prot_to_id, f)
f.close()

    
    