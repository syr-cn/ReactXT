import os
from typing import Any, Dict
import torch
from model.blip2_opt import Blip2OPT
from model.blip2_llama import Blip2Llama
from model.blip2_t5 import Blip2T5
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
from model.opt_flash_attention import replace_opt_attn_with_flash_attn, replace_opt_attn_with_original_attn
import torch.distributed as dist
from peft import LoraConfig, TaskType
from model.help_funcs import caption_evaluate, AttrDict
from transformers import Adafactor
from torch_ema import ExponentialMovingAverage

def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}

    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)


# def load_ignore_mismatch(model, state_dict):
#     keys = set(model.state_dict().keys())
#     extra_keys = set()
#     for key in state_dict:
#         if key not in keys:
#             extra_keys.add(key)
#     missing_keys = set()
#     for key in keys:
#         if key not in state_dict:
#             missing_keys.add(key)
#     ## try to print keys that are not included
#     model.load_state_dict(state_dict, strict=False)


def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict
# peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
class Blip2Model(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.llm_tune != 'full':
            to_be_removed = []
            for key in checkpoint['state_dict']:
                if key.startswith('blip2opt.opt_model') or key.startswith('blip2opt.llm_model'):
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)
        if isinstance(self.args.save_every_n_epochs, int) and self.args.save_every_n_epochs > 0:
            if self.llm_tune == 'lora' and (self.current_epoch + 1) % self.args.save_every_n_epochs == 0:
                if self.local_rank == 0: # manually fix a bug in peft module
                    if self.args.peft_config:
                        peft_config = LoraConfig(**LoraConfig.from_json_file(self.args.peft_config))
                    else:
                        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=self.args.lora_r, lora_alpha=self.args.lora_alpha, lora_dropout=self.args.lora_dropout)
                    if hasattr(self.blip2opt, 'opt_model'):
                        self.blip2opt.opt_model.peft_config['default'] = peft_config
                        self.blip2opt.opt_model.save_pretrained(os.path.join(self.logger.save_dir, f'lora_epoch_{self.current_epoch}'))
                    elif hasattr(self.blip2opt, 'llm_model'):
                        self.blip2opt.llm_model.peft_config['default'] = peft_config
                        self.blip2opt.llm_model.save_pretrained(os.path.join(self.logger.save_dir, f'lora_epoch_{self.current_epoch}'))
        return super().on_save_checkpoint(checkpoint)

    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if not hasattr(args, 'do_sample'):
            args.do_sample = False
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_inference_len = args.max_inference_len
        self.min_inference_len = args.min_inference_len
        self.num_generate_captions = args.num_generate_captions
        self.reaction_weight = args.reaction_weight
        self.llm_tune = args.llm_tune
        self.enable_flash = args.enable_flash
        if args.opt_model.find('galactica') >= 0:
            self.blip2opt = Blip2OPT(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, not args.not_tune_qformer, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt, args)
        elif args.opt_model.find('llama') >= 0 or args.opt_model.find('vicuna') >= 0:
            self.blip2opt = Blip2Llama(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt, args)
        elif args.opt_model.find('t5') >= 0:
            self.blip2opt = Blip2T5(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt, args)
        else:
            raise NotImplementedError()
        self.tokenizer = self.blip2opt.init_tokenizer()
        self.mode = args.mode
        self.downstream_task = args.downstream_task
        self.save_hyperparameters(args)
        self.save_ema_checkpoint = args.save_ema_checkpoint
        if self.save_ema_checkpoint:
            self.ema = ExponentialMovingAverage(self.parameters(), 0.99)
        self.save_on_steps = args.save_on_steps

    def load_from_stage1_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict']
        graph_encoder_dict = get_module_state_dict(state_dict, 'blip2qformer.graph_encoder')
        qformer_dict = get_module_state_dict(state_dict, 'blip2qformer.Qformer')
        ln_graph_dict = get_module_state_dict(state_dict, 'blip2qformer.ln_graph')
        qs_weight = get_module_state_dict(state_dict, 'blip2qformer.query_tokens')
        load_ignore_unexpected(self.blip2opt.Qformer, qformer_dict)
        self.blip2opt.graph_encoder.load_state_dict(graph_encoder_dict)
        self.blip2opt.ln_graph.load_state_dict(ln_graph_dict)
        self.blip2opt.query_tokens.data.copy_(qs_weight)
        return self

    # def load_from_stage1_checkpoint(self, path):
    #     ckpt = torch.load(path, map_location='cpu')
    #     state_dict = ckpt['state_dict']
    #     state_dict = {k[13:]: v for k,v in state_dict.items()}
    #     load_ignore_mismatch(self.blip2opt, state_dict)
    #     return self

    def configure_optimizers(self):
        if self.args.optimizer == 'adafactor':
            print('Using adafactor optimizer')
            optimizer = Adafactor(
                self.parameters(),
                lr=1e-3,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
            self.scheduler = None
        else:
            self.trainer.fit_loop.setup_data()
            # self.trainer.reset_train_dataloader()
            warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
            optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
            if self.args.scheduler == 'linear_warmup_cosine_lr':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
            elif self.args.scheduler == 'linear_warmup_step_lr':
                self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
            elif self.args.scheduler == 'None':
                self.scheduler = None
            else:
                raise NotImplementedError()
        return optimizer

    def test_epoch_end(self, outputs):
        print('test epoch end')
        list_ids, list_predictions, list_targets = zip(*outputs)
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]

        all_ids = [None for _ in range(self.trainer.world_size)]
        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]

        dist.all_gather_object(all_ids, list_ids)
        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_targets, targets)
        print(len(all_ids), len(all_predictions), len(all_targets))
        if self.global_rank == 0:
            print(f'saveing predictions to {self.logger.log_dir}')

            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            self.save_predictions(all_ids, all_predictions, all_targets)
            ## fixme: I am not sure if the max length is the same as previous experiments
            bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                caption_evaluate(all_predictions, all_targets, self.tokenizer, self.max_inference_len * 2)
            self.log("bleu2", bleu2, sync_dist=False)
            self.log("bleu4", bleu4, sync_dist=False)
            self.log("rouge_1", rouge_1, sync_dist=False)
            self.log("rouge_2", rouge_2, sync_dist=False)
            self.log("rouge_l", rouge_l, sync_dist=False)
            self.log("meteor_score", meteor_score, sync_dist=False)

    def save_predictions(self, rxn_ids, predictions, targets):
        assert False
        assert len(rxn_ids) == len(targets)
        assert len(predictions) == len(targets)
        with open(os.path.join(self.logger.log_dir, 'predictions.txt'), 'w', encoding='utf8') as f:
            for i, p, t in zip(rxn_ids, predictions, targets):
                line = {'index': i, 'prediction': p, 'target': t}
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        assert False

    def gather_dict_results(self, dict_list):
        list_of_dict_list = [None for _ in range(self.trainer.world_size)]
        dist.all_gather_object(list_of_dict_list, dict_list)
        dict_list = [i for ii in list_of_dict_list for i in ii] ## dict list, each dict has values that are lists of predictions, etc.
        keys = dict_list[0].keys()
        gathered_dict = {} # each value is a list of predictions, etc.
        for key in keys:
            gathered_dict[key] = [i for d in dict_list for i in d[key]]
        if self.num_generate_captions>1:
            M = self.num_generate_captions
            N = len(gathered_dict['index'])
            assert len(gathered_dict['predictions'])==N*M
            gathered_dict['predictions'] = [
                gathered_dict['predictions'][i * M:(i + 1) * M]
                for i in range(N)
            ]
        dict_list = []
        for i in range(len(gathered_dict['predictions'])):
            d = {k:gathered_dict[k][i] for k in keys}
            dict_list.append(d)
        return dict_list

    def save_results(self, dict_list, log_prefix=""):
        if log_prefix:
            name = f'{log_prefix}_predictions.txt'
        else:
            name = 'predictions.txt'
        with open(os.path.join(self.logger.log_dir, name), 'w', encoding='utf8') as f:
            for i in range(len(dict_list)):
                f.write(json.dumps(dict_list[i], ensure_ascii=True) + '\n')

    def on_validation_epoch_start(self):
        if self.enable_flash:
            replace_opt_attn_with_original_attn()
        self.saved_dict_list = []

    def on_validation_epoch_end(self):
        if self.enable_flash:
            replace_opt_attn_with_flash_attn()
        if (self.current_epoch+1) % self.caption_eval_epoch != 0:
            return 
        result_list = self.gather_dict_results(self.saved_dict_list)
        ## empty cache
        self.saved_dict_list = []
        if self.global_rank == 0:
            self.save_results(result_list, 'epoch_{}'.format(self.current_epoch))
            if self.downstream_task == 'synthesis':
                return
            all_predictions = [i['predictions'] for i in result_list]
            all_targets = [i['targets'] for i in result_list]
            bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                caption_evaluate(all_predictions, all_targets, self.tokenizer, self.max_inference_len * 2)
            self.log("bleu2", bleu2, sync_dist=False)
            self.log("bleu4", bleu4, sync_dist=False)
            self.log("rouge_1", rouge_1, sync_dist=False)
            self.log("rouge_2", rouge_2, sync_dist=False)
            self.log("rouge_l", rouge_l, sync_dist=False)
            self.log("meteor_score", meteor_score, sync_dist=False)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=1):
        if dataloader_idx == 0:
            return
        elif dataloader_idx == 1:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return
            rxn_ids, graphs, prompt_tokens, texts, inputs = batch
            ###============== Captioning Results ===================###
            samples = {'graphs': graphs, 'prompt_tokens': prompt_tokens}
            if self.mode in {'ft', 'eval', 'pretrain_eval'}:
                predictions = self.blip2opt.generate(
                    samples,
                    do_sample=self.do_sample,
                    num_beams=self.num_beams,
                    max_length=self.max_inference_len,
                    min_length=self.min_inference_len,
                    num_captions=self.num_generate_captions,
                    use_graph=not self.args.disable_graphs
                )
            else:
                raise NotImplementedError()
            self.saved_dict_list.append({
                'index': rxn_ids,
                'input': inputs,
                'predictions': predictions,
                'targets': texts
            })
        else:
            raise NotImplementedError

    def on_train_start(self):
        if hasattr(self, 'ema'):
            self.ema.to(self.device)

    def on_before_zero_grad(self, *args, **kwargs):
        if self.save_ema_checkpoint:
            if self.trainer.global_step % 100 == 0:
                self.ema.update(self.parameters())
        if self.trainer.global_step in self.save_on_steps:
            checkpoint_path = os.path.join(f"all_checkpoints/{self.args.filename}/", f'step{self.trainer.global_step}.ckpt')
            self.trainer.save_checkpoint(checkpoint_path)

    def on_train_epoch_end(self):
        save_every_n_epochs = self.args.save_every_n_epochs if self.args.save_every_n_epochs > 0 else self.args.max_epochs
        if (self.current_epoch + 1) % save_every_n_epochs != 0:
            return
        if self.save_ema_checkpoint:
            with self.ema.average_parameters():
                checkpoint_path = os.path.join(f"all_checkpoints/{self.args.filename}/", f'ema_epoch{self.current_epoch}.ckpt')
                self.trainer.save_checkpoint(checkpoint_path)

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = batch[-1].input_ids.size(0)
        ###============== Overall Loss ===================###
        if self.mode == 'ft':
            loss = self.blip2opt.forward_action(batch, use_gragh=not self.args.disable_graphs)
        elif self.mode == 'pretrain':
            loss = self.blip2opt.forward_abstract(batch, use_gragh=not self.args.disable_graphs)
        else:
            raise NotImplementedError()
        self.log("molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True, prog_bar=True)
        return loss['loss']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--tune_gnn', action='store_true', default=False)
        parser.add_argument('--not_tune_qformer', action='store_true', default=False)
        parser.add_argument('--disable_graphs', action='store_true', default=False)
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=2048, help='')
        parser.add_argument('--bert_name', type=str, default='scibert')
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        # OPT
        parser.add_argument('--opt_model', type=str, default="facebook/galactica-1.3b")
        # parser.add_argument('--prompt', type=str, default='a molecule of ')
        parser.add_argument('--num_beams', type=int, default=5)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--max_inference_len', type=int, default=512)
        parser.add_argument('--min_inference_len', type=int, default=8)
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--peft_config', type=str, default=None)
        parser.add_argument('--peft_dir', type=str, default='')

        parser.add_argument('--save_every_n_epochs', type=int, default=0)
        ## quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)

        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)

        # optimization
        parser.add_argument('--reaction_weight', type=float, default=1.0)
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--optimizer', type=str, default='adamw', help='type of scheduler')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=10)
        parser.add_argument('--num_generate_captions', type=int, default=1)
        
        # OPT Config
        parser.add_argument('--optconfig_attention_dropout', type=float, default=0.0)
        parser.add_argument('--optconfig_dropout', type=float, default=0.0)

        # others
        parser.add_argument('--save_ema_checkpoint', action='store_true', default=False)
        parser.add_argument('--save_on_steps', default=[], nargs='+', type=int)
        return parent_parser
