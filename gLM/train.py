# Copyright (c) President and Fellows of Harvard College 2023.

# This source code is licensed under the Academic and Non-Commercial Research Use Software License 
# found in the LICENSE file in the root directory of this source tree.
# usage: python train.py -d <batched_data_dir> -o <output_dir> 

import torch
from torch import nn
from gLM import *
from transformers import RobertaConfig
from transformers import AdamW
from tqdm import tqdm
import os
import sys
import numpy as np
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from glm_utils import *
import argparse
import pathlib
import datetime
import logging
from transformers.optimization import get_constant_schedule_with_warmup
import pickle as pk
import glob

def eval_model(logging, eval_pkls, model, step, writer, num_pred, device):
    """evaluates the model"""
    torch.cuda.empty_cache()
    logging.info("evaluating model")
    scaler = None
    if HALF:
        logging.info("evaluating using a mixed precision model")
        scaler = torch.cuda.amp.GradScaler()
    for pkl_f in eval_pkls:
        pickle_file =  open(pkl_f, 'rb')
        dataset = pk.load(pickle_file)
        if B_SIZE < len(dataset):
            loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=True, drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=True)
        loop = tqdm(loader, leave=True)
        for batch in loop:
            # pull all tensor batches required for eval
            # mask randomly according to mask_prob 
            inputs_embeds= batch['embeds'].type(torch.FloatTensor)
            attention_mask = batch['attention_mask'].type(torch.FloatTensor)
            rand = torch.rand(attention_mask.shape)
            masked_tokens = (rand < mask_prob) & (attention_mask != 0)
            masked_tokens = torch.unsqueeze(masked_tokens, -1)
            masked_tokens = masked_tokens.to(device)
            inputs_embeds = inputs_embeds.to(device)
            inputs_embeds  = torch.where(masked_tokens, -1.0, inputs_embeds)
            attention_mask = attention_mask.to(device)
            labels = batch['label_embeds'].type(torch.FloatTensor)
            if not NO_CLIP:
                labels = torch.clip(labels,-CLIP_VAL,CLIP_VAL )
            labels = labels.to(device)
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # process
                    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels = labels, masked_tokens = masked_tokens, output_attentions = False)
                    # extract loss
                    loss = outputs.loss.mean() #this is total loss 
                    masked_lm_loss = loss #this is lm_loss (mseloss)
                    cos_dist = outputs.cos_dist.mean()
                    # extract all relevant base metrics 
                    prediction_var = outputs.prediction_var.mean() #this is the variance of predictions
                    label_mean = outputs.label_mean.mean() # this is the mean of the labels (of masked tokens)
                    accuracy = outputs.accuracy.mean() # number of correct token predictions / total masked tokens, where correct prediction is defined as prediction being closest to label in given sequence 
            else:
                # process
                outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels = labels, masked_tokens = masked_tokens, output_attentions = False)
                # extract loss
                loss = outputs.loss.mean() #this is total loss 
                masked_lm_loss = loss #this is lm_loss (mseloss)
                cos_dist = outputs.cos_dist.mean()
                # extract all relevant base metrics 
                prediction_var = outputs.prediction_var.mean() #this is the variance of predictions
                label_mean = outputs.label_mean.mean() # this is the mean of the labels (of masked tokens)
                accuracy = outputs.accuracy.mean() # number of correct token predictions / total masked tokens, where correct prediction is defined as prediction being closest to label in given sequence
                # calculate loss for every parameter that needs grad update
            loop.set_description(f'step {step}')
            writer.add_scalar("Sequence_var",  prediction_var, step)
            writer.add_scalar("Sequence_mean",label_mean, step)
            writer.add_scalar("Total_Loss", loss, step)
            writer.add_scalar("accuracy",  accuracy, step)
            writer.add_scalar("cosine_distance",  cos_dist, step)
            if num_pred > 1:
                # this is for when there are multiple predictions. 
                masked_lm_loss = outputs.loss.mean() # this is lm_loss (mseloss)
                probs_loss = outputs.probs_loss.mean() # this is the loss corresponding to probability associated with each prediction
                avg_var = outputs.avg_var.mean() # variance across num_preds
                most_likely_lm_loss = outputs.most_likely_lm_loss.mean()
                writer.add_scalar('Most_likely_LM_Loss', most_likely_lm_loss, step)
                writer.add_scalar("Probs_Loss",  probs_loss, step)
                writer.add_scalar("avg_variance",avg_var, step)
                loop.set_postfix(total_loss_eval=loss.item(), masked_lm_loss_eval = masked_lm_loss.item(), probs_loss_eval = probs_loss.item(), avg_var_eval = avg_var.item(),  most_likely_lm_loss_eval = most_likely_lm_loss.item())
            else:
                loop.set_postfix(total_loss_eval=loss.item(), accuracy_eval= accuracy.item())
        pickle_file.close()
    return None


def train_model(logging,writerfile, data_dir, model, epochs, model_path, num_pred, checkpoint_path, from_checkpoint, ngpus, mask_prob):
    """model trainer"""
    logging.info("begin training model")
    writerfile_train = writerfile+"_train"
    writer = SummaryWriter(writerfile_train)
    writerfile_eval = writerfile+"_eval"
    eval_writer = SummaryWriter(writerfile_eval)
    f_list = os.listdir(data_dir)
    train_pkls=[]
    eval_pkls = []
    for pkl_f in f_list:
        if pkl_f.startswith('train'):
            train_pkls.append(str(os.path.join(data_dir,pkl_f)))
        elif pkl_f.startswith('eval'):
            eval_pkls.append(str(os.path.join(data_dir,pkl_f)))
    logging.info("total number of batched sequences in training dataset: "+str(len(train_pkls)*10000))
    logging.info("total number of batched sequences in eval dataset: "+str(len(eval_pkls)*10000))
    optim = AdamW(model.parameters(), lr=LR)
    begin_epoch = 0
    scheduler = None
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if WARMUP is not None:
        scheduler = get_constant_schedule_with_warmup(optim, num_warmup_steps = WARMUP)
    if from_checkpoint is not None:
        # load model from checkpoint
        checkpoint = torch.load(from_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        begin_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logging.info("loaded from checkpoint:"+str(from_checkpoint))
        logging.info("loaded from checkpoint, continuing from epoch "+str(begin_epoch))
    else:
        model.to(device)
    scaler = None
    if HALF:
        logging.info("this is a mixed precision model")
        scaler = torch.cuda.amp.GradScaler()
    #setting gradient checkpointing to true model.encoder.gradient_checkpointing = GRAD
    
    step = begin_epoch * ((SEQS_PER_BATCH*len(train_pkls))//B_SIZE)
    for epoch in range(begin_epoch, epochs):
        with accelerator.accumulate(model):
            for pkl_f in train_pkls:
                pickle_file =  open(pkl_f, 'rb')
                dataset = pk.load(pickle_file)
                if B_SIZE < len(dataset):
                    loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=True, drop_last=True)
                else:
                    loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=True)

                # setup loop with TQDM and dataloader
                loop = tqdm(loader, leave=True)

                for batch in loop:
                    # initialize calculated gradients (from prev step)
                    optim.zero_grad()
                    # pull all tensor batches required for training
                    # mask randomly according to mask_prob 
                    inputs_embeds= batch['embeds'].type(torch.FloatTensor)
                    attention_mask = batch['attention_mask'].type(torch.FloatTensor)
                    rand = torch.rand(attention_mask.shape)
                    masked_tokens = (rand < mask_prob) & (attention_mask != 0)
                    masked_tokens = torch.unsqueeze(masked_tokens, -1)
                    masked_tokens= masked_tokens.to(device)
                    inputs_embeds = inputs_embeds.to(device)
                    inputs_embeds  = torch.where(masked_tokens, -1.0, inputs_embeds)
                    attention_mask = attention_mask.to(device)
                    labels = batch['label_embeds'].type(torch.FloatTensor)
                    labels = labels.to(device)
                    if not NO_CLIP:
                        labels = torch.clip(labels,-CLIP_VAL,CLIP_VAL )
                    labels_hist = labels.view(-1)
                    embeds_hist = inputs_embeds.view(-1)
                    if scaler is not None:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        # process
                            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels = labels, masked_tokens = masked_tokens, output_attentions = False)
                            loss = outputs.loss.mean() #this is total loss 
                            cos_dist = outputs.cos_dist.mean()
                            masked_lm_loss = loss #this is lm_loss (mseloss)
                            # extract all relevant base metrics 
                            prediction_var = outputs.prediction_var.mean() #this is the variance of predictions
                            prediction_mean = outputs.prediction_mean.mean() # this is the mean of the predictions 
                            accuracy = outputs.accuracy.mean() # number of correct token predictions / total masked tokens, where correct prediction is defined as prediction being closest to label in given sequence 
                            accelerator.backward(scaler.scale(loss))
                            scaler.step(optim)
                            scaler.update()
                    else:
                        # process
                        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels = labels, masked_tokens = masked_tokens, output_attentions = False)
                        # extract loss
                        loss = outputs.loss.mean() #this is total loss 
                        cos_dist = outputs.cos_dist.mean()
                        masked_lm_loss = loss #this is lm_loss (mseloss)
                        # extract all relevant base metrics 
                        prediction_var = outputs.prediction_var.mean() #this is the variance of predictions
                        prediction_mean = outputs.prediction_mean.mean() # this is the mean of the predictions 
                        accuracy = outputs.accuracy.mean() # number of correct token predictions / total masked tokens, where correct prediction is defined as prediction being closest to label in given sequence
                        # calculate loss for every parameter that needs grad update
                        accelerator.backward(loss)
                        #loss.backward()
                        # update parameters
                        optim.step()
                    nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                    if scheduler is not None:
                        scheduler.step()
                        last_lr = scheduler.get_last_lr()
                        last_lr = np.mean(last_lr)
                        writer.add_scalar("learning rate", last_lr, step)
                    # print relevant info to progress bar
                    loop.set_description(f'Epoch {epoch}')
                    writer.add_scalar("Sequence_var", prediction_var, step)
                    writer.add_scalar("Sequence_mean", prediction_mean, step)
                    writer.add_scalar("Total_Loss", loss, step)
                    writer.add_scalar("accuracy", accuracy, step)
                    writer.add_scalar("cosine_distance", cos_dist, step)
                    step +=1

                    if epoch == 0 :
                        writer.add_histogram("label_hist", labels_hist, step)
                        writer.add_histogram("embeds_hist", embeds_hist, step)
                    if num_pred > 1:
                        # this is for when there are multiple predictions. 
                        masked_lm_loss = outputs.loss.mean() # this is lm_loss (mseloss)
                        probs_loss = outputs.probs_loss.mean() # this is the loss corresponding to probability associated with each prediction 
                        avg_var = outputs.avg_var.mean() # variance across num_preds
                        most_likely_lm_loss = outputs.most_likely_lm_loss.mean()
                        writer.add_scalar("Most_likely_LM_Loss", most_likely_lm_loss, step)
                        writer.add_scalar("Probs_Loss", probs_loss, step)
                        writer.add_scalar("avg_variance", avg_var, step)
                        loop.set_postfix(total_loss_train=loss.item(), masked_lm_loss_train=masked_lm_loss.item(), probs_loss_train=probs_loss.item(), avg_var_train=avg_var.item(),  most_likely_lm_loss_train = most_likely_lm_loss.item())
                    else:
                        loop.set_postfix(total_loss_train=loss.item(), prediction_var=prediction_var.item(), accuracy= accuracy.item())
                pickle_file.close()
            if epoch % CP_FREQ == 0 and epoch != 0:
                # checkpoint_saving at CP_FREQ 
                logging.info("checkpointing at epoch "+str(epoch))
                checkpoint_path_epoch = checkpoint_path +"_"+ str(epoch) + ".cp"
                if scheduler is not None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict' :model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss' : loss,
                    }, checkpoint_path_epoch)
                else: 
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict' :model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'loss' : loss,
                    }, checkpoint_path_epoch)
                model_epoch_path = model_path+"_"+str(epoch)+".pt"
                if ngpus > 1:
                    model.module.save_pretrained(model_epoch_path)
                else:
                    model.save_pretrained(model_epoch_path)
                logging.info("checkpoint stored at "+checkpoint_path_epoch)
                logging.info("pretrained stored at "+model_epoch_path)
            if epoch % EVAL_FREQ ==0 and not NO_EVAL:
                # run eval and save pretrained model at EVAL_FREQ
                eval_model(logging, eval_pkls, model, step, eval_writer, num_pred, device)

    writer.flush()
    writer.close()
    
parser = argparse.ArgumentParser(description = "Trains the gLM model")
parser.add_argument('-d','--data_dir', type=pathlib.Path, help='data directory')
parser.add_argument('--pos_type', type=str, help="position embedding type", default = "relative_key_query")
parser.add_argument('-annot', '--annot_path',  help='path to pfam ids', default = None)
parser.add_argument('-n', '--ngpus', type=int, help='number of GPUs to use',  default = 1)
parser.add_argument('--epochs', type=int, help='number epochs', default = 1000)
parser.add_argument('-b','--batch_size', type=int, help='batch_size', default = 1000)
parser.add_argument('-m','--max_seq_len', type=int, help='Max sequence length', default = 30)
parser.add_argument('-a','--num_attention_heads', type=int, help='number of attention heads', default = 10)
parser.add_argument('-l','--num_hidden_layers', type=int, help='number of hidden layers', default =19)
parser.add_argument('-hidden','--output_hidden_layers', action='store_true', help='False if no hidden layers should be output', default =True)
parser.add_argument('-npred','--num_pred', type = int, help='number of predictions', default =4)
parser.add_argument('--eval_freq', type=int, help='frequency of evalualtion (X epochs)', default =10)
parser.add_argument('-cp', '--from_checkpoint',  help='path to checkpoint if you want to re-run from checkpoint', default =None)
parser.add_argument('-cp_freq', '--checkpoint_freq', type = int, help='checkpoint frequency (X epochs)', default =10)
parser.add_argument('-o', '--output_path', type=str, help='train output directory', default = None)
parser.add_argument('-lr', '--lr', type=float, help='learning rate', default = 1e-4)
parser.add_argument('--hidden_size', type=int, help='hidden size', default = 1280)
parser.add_argument('--num_pc_label', type=int, help='number of components for PCA', default = 100)
parser.add_argument('--warmup', type=int, help='number of warm up steps', default = 5000)
parser.add_argument('--full', action='store_true', help='full precision', default = False)
parser.add_argument('--admin', action='store_true', help='admin initialization', default = False)
parser.add_argument('--grad_check', action='store_true', help='gradient_checkpointing', default = False)
parser.add_argument('--no_clip', action='store_true', help = 'no clipping of input embedding', default = False)
parser.add_argument('--emb_dim', type=int, help = 'emb_dim', default = 1281 )
parser.add_argument('--clip_val', type=int, help = 'clipping value (default -10,10)', default = 10)
parser.add_argument('--no_eval', action='store_true', help = 'no evaluation', default = False)
parser.add_argument('--seqs_per_batch', type=int, help = 'sequences per pkl file', default = 10000)

# load all arguments, issue erros
args = parser.parse_args()
if args.data_dir is None :
    parser.error('--data_dir must be specified')


# define all parameters
data_dir = args.data_dir
ngpus = args.ngpus
epochs = args.epochs
num_pred = args.num_pred
max_seq_length = args.max_seq_len 
num_attention_heads = args.num_attention_heads
num_hidden_layers= args.num_hidden_layers
output_hidden_layers = args.output_hidden_layers
from_checkpoint = args.from_checkpoint
output_path = args.output_path
pos_type = args.pos_type

# define all global variables 
CP_FREQ = args.checkpoint_freq
B_SIZE = args.batch_size #48, batch size
EVAL_FREQ = args.eval_freq
HIDDEN_SIZE = args.hidden_size
LR = args.lr #learning rate 
WARMUP = args.warmup
HALF = not args.full
ADMIN = args.admin
GRAD = args.grad_check
NO_CLIP = args.no_clip
EMB_DIM = args.emb_dim
NUM_PC_LABEL = args.num_pc_label
NO_EVAL = args.no_eval
CLIP_VAL = args.clip_val
SEQS_PER_BATCH = args.seqs_per_batch
# begin logging 
e = datetime.datetime.now()
torch.cuda.empty_cache()
accelerator = Accelerator(gradient_accumulation_steps=10)  
# make output folder if not specified
if output_path == None:
    output_path = "training_results"+ e.strftime("-%d-%m-%Y-%H:%M:%S")
if not os.path.exists(output_path):
    os.mkdir(output_path)
else:
    output_path = output_path
logfile_path = output_path+"/info"+ e.strftime("-%d-%m-%Y-%H:%M:%S")+".log"
if not os.path.exists(output_path+"/runs"):
    os.mkdir(output_path+"/runs")

writerfile = output_path + "/runs/"+ str(output_path)+"_"+str(os.path.basename(data_dir))+"_l_"+str(num_hidden_layers)+"_a_"+str(num_attention_heads)+"_b_"+str(B_SIZE)+"_lr_"+str(LR)+"_h_"+str(HIDDEN_SIZE)+"_lpc_"+str(NUM_PC_LABEL)+"_num_pred_"+str(num_pred)

# log is stored in both logfile and streamed to stdout
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=[
        logging.FileHandler(logfile_path),
        logging.StreamHandler()])
logging.info("output folder: " +output_path)
logging.info("log file is located here: " +logfile_path)
string_of_command = f"{' '.join(sys.argv)}"
logging.info("command: " + string_of_command)

# make checkpoint and pretrained model directories in output folder
if not os.path.exists(output_path+"/checkpoints"):
    os.mkdir(output_path+"/checkpoints")

checkpoint_path = output_path+"/checkpoints/" + str(os.path.basename(data_dir)) 
checkpoint_files = glob.glob(output_path+"/checkpoints/*")
if len(checkpoint_files) > 0:
    from_checkpoint =  max(checkpoint_files , key = os.path.getmtime)
logging.info("find checkpoints here: "+output_path+"/checkpoints/")
if not os.path.exists(output_path+"/pretrained_models"):
    os.mkdir(output_path+"/pretrained_models")
model_path = output_path+"/pretrained_models/"+str(os.path.basename(data_dir))
logging.info("find pretrained_models here: "+output_path+"/pretrained_models/")

#populate config 
if num_pred > 1:
    pred_probs = True
else:
    pred_probs = False
config = RobertaConfig(
    max_position_embedding = max_seq_length,
    hidden_size = HIDDEN_SIZE,
    num_attention_heads = num_attention_heads,
    type_vocab_size = 1,
    tie_word_embeddings = False,
    num_hidden_layers = num_hidden_layers,
    num_pc = NUM_PC_LABEL, 
    num_pred = num_pred,
    predict_probs = pred_probs,
    emb_dim = EMB_DIM,
    position_embedding_type = pos_type,
)

mask_prob = 0.15
model = gLM(config)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info("number of hidden layers: "+str(num_hidden_layers))
logging.info("number of attention heads: "+str(num_attention_heads))
logging.info("embedding dimension/hidden size: "+str(HIDDEN_SIZE))
logging.info("number of predictions: "+str(num_pred))
logging.info("learning rate: "+str(LR))
logging.info("batch_size: "+str(B_SIZE))
logging.info("number of gpus: "+str(ngpus))
logging.info("total number of trainable parameters: "+str(total_params))
logging.info("pos embedding type: "+str(pos_type))
logging.info("number of warm up steps "+str(WARMUP))

# parallelize model if there are multiple gpus provided
if ngpus>1:
    model = torch.nn.DataParallel(model)

train_model(logging,writerfile,data_dir,model,epochs,model_path,num_pred= num_pred, checkpoint_path=checkpoint_path, from_checkpoint=from_checkpoint, ngpus=ngpus, mask_prob=mask_prob)