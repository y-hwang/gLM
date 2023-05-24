# Copyright (c) President and Fellows of Harvard College 2023.

# This source code is licensed under the Academic and Non-Commercial Research Use Software License 
# found in the LICENSE file in the root directory of this source tree.

from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import *
import torch
from torch import nn
from torch.nn import MSELoss
from typing import Optional, Tuple, Union
import torch.utils.checkpoint
from transformers.utils import ModelOutput


class MultiPredMaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    masked_lm_loss: Optional[torch.FloatTensor] = None
    probs_loss: Optional[torch.FloatTensor] = None
    most_likely_lm_loss: Optional[torch.FloatTensor] = None
    avg_var: Optional[torch.FloatTensor] = None
    logits_all_preds: torch.FloatTensor = None
    logits_closest: torch.FloatTensor = None
    logits_most_likely: torch.FloatTensor = None
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None


class gLMMultiHead(nn.Module):
    """Roberta Head for masked language modeling with num_pred predictions."""

    def __init__(self, config):
        super().__init__()
        self.num_pred = config.num_pred
        self.num_pc = config.num_pc
        self.dense = nn.Linear(config.hidden_size, config.num_pc*self.num_pred)
        self.hidden_size = config.hidden_size
        self.predict_probs = config.predict_probs
        if config.predict_probs:
            self.dense_prob = nn.Linear(config.hidden_size, self.num_pred)


    def forward(self, features, **kwargs):
        #[batch, max_seq_len, hidden_size].
        x = self.dense(features)
        x_shape = list(x.shape)
        # [batch, max_seq_len, num_pred, hidden_size].
        x = x.view(*x_shape[:-1],self.num_pred, self.num_pc)
        if self.predict_probs:
            # [batch, max_seq_len, num_pred].
            probs = self.dense_prob(features)
        else:
            probs = None
        return x, probs

class gLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()        
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.num_pc)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-06)
    def forward(self, features, **kwargs):
        x = self.dense(features)
        return x

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_embeds'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_embeds, and attention_mask for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}

def symmetrize(x):
    """Make layer symmetric in final two dimensions, used for contact prediction."""
    return x + x.transpose(-1, -2)


class ContactPredictionHead(nn.Module):
    """modified fair esm's contact prediction head"""
    """Performs symmetrization and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        bias=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.regression = nn.Linear(in_features, 1, bias)
        self.activation = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tokens, attentions):
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: B x C x T x T
        attentions = attentions.to(
            self.regression.weight.device
        )  # attentions always float32, may need to convert to float16
        attentions = symmetrize(attentions)
        attentions = attentions.permute(0, 2, 3, 1)
        return attentions
class gLM_base(RobertaModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.num_pred = config.num_pred
        self.predict_probs = config.predict_probs
        # define head here depending on the number of predictions requested 
        if  self.num_pred == 1:
            self.lm_head = gLMHead(config)
        else:
            self.lm_head = gLMMultiHead(config)

        self.contact_head = ContactPredictionHead(config.num_hidden_layers*config.num_attention_heads)
        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder
 
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        masked_tokens: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor],  MultiPredMaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        sequence_output = outputs[0]
        if labels is not None:
            loss_fct = MSELoss()
            labels_masked = labels * masked_tokens

            if self.num_pred > 1: 
                predicted_embeds, predicted_probs = self.lm_head(sequence_output)
                batch_size, max_seq_len, n_pred, num_pc = predicted_embeds.shape
                # labels: [batch, max_seq_len, hidden_size]
                # predicted_embeds: [batch, max_seq_len, num_pred, hidden_size]
                # masked_tokens: [batch,max_seq_len, 1]
                # predicted_probs: [batch, max_seq_len, num_pred]
                num_masked_tokens = torch.sum(masked_tokens.view(-1))
                predicted_embeds_masked = predicted_embeds * masked_tokens.unsqueeze(-1)
                predicted_probs_masked = predicted_probs * masked_tokens
                raw_probs = predicted_probs_masked.view(-1,4)
                softmax = nn.Softmax(dim=1)
                probs = softmax(raw_probs)
                # [batch, max_seq_len, num_pred, hidden_size]
                diff = predicted_embeds_masked - torch.unsqueeze(labels_masked, 2)
                # [batch, max_seq_len, num_pred]
                dist = torch.linalg.norm(diff, dim=-1)
                # [batch,mag_seq_len]
                avg_var = torch.sum(torch.sum(torch.var(predicted_embeds_masked,dim=-2),dim=-1)/(num_masked_tokens*num_pc))
                # [batch, max_seq_len]
                argmin_dist = torch.argmin(dist, dim=-1)
                # [batch * max_seq_len]
                argmin_dist_flat = argmin_dist.view(-1)
                # [batch * max_seq_len, num_pred, hidden_size]
                predicted_embeds_flat = predicted_embeds_masked.view(-1, n_pred, num_pc)
                closest_embeds_flat = predicted_embeds_flat[torch.arange(batch_size * max_seq_len),argmin_dist_flat]
                predicted_probs_flat = probs.view(-1,n_pred,1)
                closest_probs_flat = predicted_probs_flat[torch.arange(batch_size * max_seq_len),argmin_dist_flat]
                closest_probs = closest_probs_flat.view(batch_size,max_seq_len,1)
                # [batch_size, max_seq_len, hidden_size]
                closest_predicted_embeds = closest_embeds_flat.view(batch_size,max_seq_len,num_pc)
                masked_lm_loss = loss_fct(closest_predicted_embeds, labels_masked)

                # cosine similarity
                cos_dist = cos(closest_predicted_embeds[masked_tokens.squeeze(-1)], labels[masked_tokens.squeeze(-1)])
                #avg_norm_diff = torch.sum(torch.absolute(torch.linalg.norm(closest_embeds_flat,dim=1)-torch.linalg.norm(labels_masked.view(batch_size*max_seq_len,hidden_size), dim=1)))/num_masked_tokens
                argmax_prob = torch.argmax(predicted_probs, dim = -1)
                argmax_prob_flat = argmax_prob.view(-1)
                most_likely_embeds_flat = predicted_embeds_flat[torch.arange(batch_size * max_seq_len),argmax_prob_flat]
                most_likely_embeds = most_likely_embeds_flat.view(batch_size,max_seq_len,num_pc)
                most_likely_lm_loss = loss_fct(most_likely_embeds, labels_masked)

                # Compute probs loss.
                probs_loss_fct = nn.CrossEntropyLoss()
                # [batch, seq_len]
                probs_label = argmin_dist
                probs_label_masked = probs_label * masked_tokens.squeeze(-1)
                probs_loss = probs_loss_fct(predicted_probs_masked.view(-1,n_pred), probs_label_masked.view(-1))
                prob_loss_wt = 0.0001 
                total_loss = masked_lm_loss + prob_loss_wt * probs_loss

                # calculate metrics
                prediction_var = torch.var(closest_predicted_embeds[masked_tokens.squeeze(-1)])
                label_var = torch.var(labels[masked_tokens.squeeze(-1)])
                prediction_mean = torch.mean(closest_predicted_embeds[masked_tokens.squeeze(-1)])
                label_mean = torch.mean(labels[masked_tokens.squeeze(-1)])
                
                # calculate accuracy per batch
                (masked_indices_batch, masked_indices_seq) = torch.where(masked_tokens.squeeze(-1))
                correct_prediction = 0
                masked_total = torch.sum(masked_tokens)

                for i in range(masked_total):
                    batch_index = masked_indices_batch[i]
                    seq_index = masked_indices_seq[i]
                    predicted_emb = closest_predicted_embeds[batch_index][seq_index]
                    seq_to_prediction = labels[batch_index] -  predicted_emb
                    distance = torch.linalg.norm(seq_to_prediction, dim=-1)
                    argmin_ind = torch.argmin(distance)
                    if argmin_ind == seq_index:
                        correct_prediction+=1
                accuracy = correct_prediction/masked_total
            
            else:

                # single prediction mode 
                predicted_embeds = self.lm_head(sequence_output)
                batch_size, max_seq_len, num_pc = predicted_embeds.shape
                closest_predicted_embeds = predicted_embeds
                
                # predicted_embeds = [batch, max_seq_len, hidden_size]
                predicted_embeds_masked = predicted_embeds * masked_tokens
                
                # calculate MSE loss on masked tokens only
                masked_lm_loss = loss_fct(predicted_embeds[masked_tokens.squeeze(-1)], labels[masked_tokens.squeeze(-1)])
                total_loss = masked_lm_loss
                # cosine similarity
                cos_dist = cos(predicted_embeds[masked_tokens.squeeze(-1)], labels[masked_tokens.squeeze(-1)])

                # set parameters that are not needed for single prediction as None 
                probs_loss = None
                most_likely_lm_loss =None
                predicted_probs = None
                avg_var = None
                most_likely_embeds = None

                # calculate metrics
                prediction_var = torch.var(predicted_embeds[masked_tokens.squeeze(-1)])
                label_var = torch.var(labels[masked_tokens.squeeze(-1)])
                prediction_mean = torch.mean(predicted_embeds[masked_tokens.squeeze(-1)])
                label_mean = torch.mean(labels[masked_tokens.squeeze(-1)])

                # calculate accuracy per batch
                (masked_indices_batch, masked_indices_seq) = torch.where(masked_tokens.squeeze(-1))
                correct_prediction = 0
                masked_total = torch.sum(masked_tokens)

                for i in range(masked_total):
                    batch_index = masked_indices_batch[i]
                    seq_index = masked_indices_seq[i]
                    predicted_emb = predicted_embeds[batch_index][seq_index]
                    seq_to_prediction = labels[batch_index] -  predicted_emb
                    distance = torch.linalg.norm(seq_to_prediction, dim=-1)
                    argmin_ind = torch.argmin(distance)
                    if argmin_ind == seq_index:
                        correct_prediction+=1
                accuracy = correct_prediction/masked_total
                
        if not return_dict:
            output = (predicted_embeds,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        contacts = None
        
        if output_attentions:
            attn_weights = []
            for attn_per_layer in outputs['attentions']:
                attn_weights.append(attn_per_layer)
            attentions = torch.stack(attn_weights, 1)
            contacts = self.contact_head(inputs_embeds, attentions)
        all_hidden_states = []
        if output_hidden_states: 
            for states_per_layer in outputs['hidden_states']:
                all_hidden_states.append(states_per_layer)
            all_hidden_states = torch.stack(all_hidden_states, 1)
        
        return MultiPredMaskedLMOutput(
            loss=total_loss,
            cos_dist=cos_dist,
            logits_all_preds=predicted_embeds,
            prediction_mean = prediction_mean,
            prediction_var = prediction_var,
            label_var = label_var,
            label_mean=label_mean,
            accuracy = accuracy,
            masked_lm_loss = masked_lm_loss,
            probs_loss = probs_loss,
            most_likely_lm_loss = most_likely_lm_loss,
            avg_var=avg_var,
            probs = predicted_probs,
            logits_closest = closest_predicted_embeds,
            last_hidden_state = sequence_output,
            logits_most_likely = most_likely_embeds,
            contacts = contacts,
            all_hidden_states = all_hidden_states,
            closest_probs = closest_probs,
        )
class gLM(RobertaModel):
    
    def __init__(self, config):
        super().__init__(config) 
        self.roberta = gLM_base(config)
        # linear resizing 
        self.dense = nn.Linear(config.emb_dim, config.hidden_size)
        self.output_attentions = config.output_attentions
        # The LM head weights require special treatment only when they are tied with the word embeddings
        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        masked_tokens: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor],  MultiPredMaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = self.dense(inputs_embeds)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels = labels,
            masked_tokens = masked_tokens,
        )
        
        return outputs
    
