import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from data_provider.pretrain_dm import PretrainDM
from data_provider.tune_dm import TuneDM
from model.opt_flash_attention import replace_opt_attn_with_flash_attn
from model.blip2_model import Blip2Model
from model.dist_funs import MyDeepSpeedStrategy

## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)

try:
    class MyDDPSpawnStrategy(strategies.DDPSpawnStrategy):
        def load_model_state_dict(self, checkpoint):
            assert self.lightning_module is not None
            self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)
except:
    pass

def main(args):
    pl.seed_everything(args.seed)
    # model
    if args.init_checkpoint:
        model = Blip2Model(args)
        ckpt = torch.load(args.init_checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded model from {args.init_checkpoint}")
    else:
        model = Blip2Model(args)

    print('total params:', sum(p.numel() for p in model.parameters()))

    if args.opt_model.find('galactica') >= 0 or args.opt_model.find('t5') >= 0:
        tokenizer = model.blip2opt.opt_tokenizer
    elif args.opt_model.find('llama') >= 0 or args.opt_model.find('vicuna') >= 0:
        tokenizer = model.blip2opt.llm_tokenizer
    else:
        raise NotImplementedError
    # data
    if args.mode in {'pretrain', 'pretrain_eval'}:
        dm = PretrainDM(
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            root=args.root,
            text_max_len=args.text_max_len,
            rxn_max_len=args.rxn_max_len,
            smi_max_len=args.smi_max_len,
            tokenizer=tokenizer,
            args=args
        )
    elif args.mode in {'ft', 'eval'}:
        dm = TuneDM(
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            root=args.root,
            text_max_len=args.text_max_len,
            rxn_max_len=args.rxn_max_len,
            smi_max_len=args.smi_max_len,
            tokenizer=tokenizer,
            downstream_task=args.downstream_task,
            args=args
        )

    callbacks = [TQDMProgressBar(refresh_rate=args.tqdm_interval)]
    ## fixme save only used parameters
    # callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", every_n_epochs=10, save_top_k=-1))
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/",
                                         filename='{epoch:02d}',
                                         every_n_epochs=args.save_every_n_epochs,
                                         save_last=True,
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))
    if len(args.devices.split(',')) > 1:
        if args.strategy_name == 'fsdp':
            strategy = strategies.DDPFullyShardedNativeStrategy()
        elif args.strategy_name == 'deepspeed':
            strategy = strategies.DeepSpeedStrategy(stage=3)
        elif args.strategy_name == 'mydeepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            strategy = MyDDPSpawnStrategy(find_unused_parameters=True)
    else:
        strategy = None
        args.devices = eval(args.devices)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    reload_freq = 1 if args.mode == 'pretrain' else 0
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
        reload_dataloaders_every_n_epochs=reload_freq
        #  limit_train_batches=100,
    )

    if args.mode in {'pretrain', 'ft'}:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
    elif args.mode in {'eval', 'pretrain_eval'}:
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.validate(model, datamodule=dm)
        # trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="main")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--mode', type=str, default='pretrain', choices=['pretrain', 'ft', 'eval', 'pretrain_eval'])
    parser.add_argument('--strategy_name', type=str, default='mydeepspeed')
    parser.add_argument('--iupac_prediction', action='store_true', default=False)
    parser.add_argument('--ckpt_path', type=str, default=None)
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Model.add_model_specific_args(parser)  # add model args
    parser = PretrainDM.add_model_specific_args(parser)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--downstream_task', type=str, default='action', choices=['action', 'synthesis', 'caption', 'chebi'])
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--enable_flash', action='store_true', default=False)
    parser.add_argument('--disable_graph_cache', action='store_true', default=False)
    parser.add_argument('--predict_rxn_condition', action='store_true', default=False)
    parser.add_argument('--generate_restrict_tokens', action='store_true', default=False)
    parser.add_argument('--train_restrict_tokens', action='store_true', default=False)
    parser.add_argument('--smiles_type', type=str, default='default', choices=['default', 'canonical', 'restricted', 'unrestricted', 'r_smiles'])
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--tqdm_interval', type=int, default=50)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    args = parser.parse_args()

    if args.enable_flash:
        replace_opt_attn_with_flash_attn()
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args

if __name__ == '__main__':
    main(get_args())

