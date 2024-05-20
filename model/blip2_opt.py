"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from ogb.utils import smiles2graph
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data
import numpy as np
from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from model.blip2 import Blip2Base
from model.help_funcs import get_not_allowed_tokens_ids
from transformers import AutoTokenizer
from transformers import OPTForCausalLM, OPTConfig
# from opendelta import LoraModel
# from opendelta.delta_models.lora import LoraConfig
# from opendelta.delta_configs

opt_model_list = [
    "facebook/galactica-125m",
    "facebook/galactica-1.3b",
    "facebook/galactica-6.7b",
    "facebook/galactica-30b",
]

def mask_by_len(input, lens, fill_value=0):
    '''
    input: shape = [N, D]
    lens: shape = [N]
    '''
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input


def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

import re
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")


def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].

    Parameters
    ----------
    n : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"

def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization

    Parameters
    ----------
    text : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)

def smiles_handler(text, mol_ph):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)

    text = CUSTOM_SEQ_RE.sub(r'\1\3\4%s' % (mol_ph), text)
    text = escape_custom_split_sequence(text)
    return text, smiles_list


class Blip2OPT(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def __init__(
        self,
        bert_name,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        tune_qformer=False,
        num_query_token=32,
        cross_attention_freq=2,
        llm_tune='freeze',
        peft_dir='',
        opt_model="facebook/galactica-1.3b",
        prompt="",
        args=None,
    ):
        super().__init__()
        self.args = args

        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn
        self.tune_qformer = tune_qformer
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        else:
            logging.info("tune graph encoder")

        self.num_query_token = num_query_token
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
        if not tune_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze qformer encoder")
        else:
            logging.info("tune qformer encoder")
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        opt_config_params = {k[len("optconfig_"):]: v for k, v in vars(args).items() if k.startswith("optconfig_")}
        config = OPTConfig.from_pretrained(opt_model, **opt_config_params)
        ## initialize opt model
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False, padding_side='right')
        self.opt_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.opt_tokenizer.add_tokens('<mol>') # molecule placeholder
        self.mol_token = '<mol>'
        self.opt_tokenizer.mol_token_id = self.opt_tokenizer("<mol>", add_special_tokens=False).input_ids[0]

        self.collater = Collater([], [])

        if opt_model == 'facebook/galactica-125m':
            self.opt_model = OPTForCausalLM.from_pretrained(opt_model, config=config)
        else:
            if torch.cuda.is_bf16_supported():
                self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16, config=config)
            else:
                self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16, config=config)
        self.opt_model.resize_token_embeddings(len(self.opt_tokenizer)) ## this will cause bug when full fine-tuning the opt model

        self.llm_tune = llm_tune
        if llm_tune == 'lora':
            if peft_dir:
                self.opt_model = PeftModel.from_pretrained(self.opt_model, peft_dir, is_trainable=True)
            else:
                if self.args.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.args.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
                self.peft_config = peft_config
                self.opt_model = get_peft_model(self.opt_model, peft_config)
                self.opt_model.print_trainable_parameters()
        elif llm_tune == 'freeze':
            for name, param in self.opt_model.named_parameters():
                param.requires_grad = False
        elif llm_tune == 'full':
            pass
        else:
            raise NotImplementedError()

        ## fixme: this is different from the original BLIP2
        if args.mode=='pretrain_eval':
            self.eos_token_id = self.opt_tokenizer(
                "[START_SMILES]\n", add_special_tokens=False
            ).input_ids
        else:
            self.eos_token_id = self.opt_tokenizer(
                "\n", add_special_tokens=False
            ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        ## fixme: no prompt yet
        self.prompt = prompt
        self.rxn_batch_size = args.rxn_batch_size
        self.generate_restrict_tokens = args.generate_restrict_tokens
        self.train_restrict_tokens = args.train_restrict_tokens
        if self.generate_restrict_tokens or self.train_restrict_tokens:
            self.bad_words_ids = get_not_allowed_tokens_ids(opt_model)
        # prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)
    
    def opt_forward_v2(
        self, 
        inputs_embeds,
        attention_mask,
        labels,
        bad_word_ids=None,
    ):
        output = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )
        logits = output.logits
        labels = labels.to(logits.device)
        # Shift so that tokens < n predict n

        if bad_word_ids:
            bad_word_ids = torch.tensor(bad_word_ids, device=logits.device, dtype=torch.long)
            bad_word_ids = bad_word_ids.squeeze()
            logits[:, :, bad_word_ids] = -100

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, self.opt_model.config.vocab_size)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels.view(-1))
        return loss

    def forward_action(self, batch, use_gragh=True):
        # batch unpack
        rxn_ids, graphs, text_tokens = batch
        if use_gragh:
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            if not self.tune_gnn:
                graph_embeds = graph_embeds.detach()

            # graph embedding calculation
            graph_embeds = self.ln_graph(graph_embeds, graph_masks)
            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
                return_dict=True,
            )
            mol_tokens = self.opt_proj(query_output.last_hidden_state) # graph_num x num_query_token x D
        else:
            del graphs

        pad_mask = text_tokens.input_ids == self.opt_tokenizer.pad_token_id
        targets = text_tokens.input_ids.masked_fill(pad_mask, -100)
        targets = targets.masked_fill(text_tokens.is_mol_token, -100)
        targets = targets.masked_fill(text_tokens.token_type_ids == 0, -100)

        inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
        if use_gragh:
            inputs_embeds[text_tokens.is_mol_token] = mol_tokens.flatten(0, 1) # graph_num x emb_dim

        if self.train_restrict_tokens:
            loss = self.opt_forward_v2(
                inputs_embeds=inputs_embeds,
                attention_mask=text_tokens.attention_mask,
                labels=targets,
                bad_word_ids=self.bad_words_ids,
            )
        else:
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss
        return {"loss": loss}

    def forward_abstract(self, batch, use_gragh=True):
        # batch unpack
        graphs, text_tokens = batch
        if use_gragh:
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            if not self.tune_gnn:
                graph_embeds = graph_embeds.detach()

            # graph embedding calculation
            graph_embeds = self.ln_graph(graph_embeds, graph_masks)
            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
                return_dict=True,
            )
            mol_tokens = self.opt_proj(query_output.last_hidden_state) # graph_num x num_query_token x D
        else:
            del graphs

        pad_mask = text_tokens.input_ids == self.opt_tokenizer.pad_token_id
        targets = text_tokens.input_ids.masked_fill(pad_mask, -100)
        targets = targets.masked_fill(text_tokens.is_mol_token, -100)

        inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
        if use_gragh:
            inputs_embeds[text_tokens.is_mol_token] = mol_tokens.flatten(0, 1) # graph_num x emb_dim

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        use_graph=True,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        graphs = samples['graphs']
        prompt_tokens = samples['prompt_tokens']
        # prompt_lens = samples['prompt_lens']
        # with self.maybe_autocast():
        if use_graph:
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            graph_embeds = self.ln_graph(graph_embeds)

            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_masks,
                return_dict=True,
            )
            mol_tokens = self.opt_proj(query_output.last_hidden_state)

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        if use_graph:
            prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1).to(dtype=prompt_embeds.dtype)
        extra_params = {}
        if self.generate_restrict_tokens:
            extra_params['bad_words_ids'] = self.bad_words_ids

        outputs = self.opt_model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_tokens.attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            # pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            # use_cache=False,
            **extra_params
        )
        output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        output_text = [text.strip() for text in output_text]
        return output_text
