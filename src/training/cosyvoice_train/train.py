# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A6 COPY of cosyvoice/bin/train.py (repo commit 074ca6d, read-only upstream). Deviations, applied
# identically to both arms (see reports/training_budget_comparison.md):
#   * sys.path: ~/CosyVoice + third_party/Matcha-TTS + this workdir (so yaml can name src.training.processor_ext);
#   * uses src/training/cosyvoice_train/executor.py (max_steps + token/VRAM logging);
#   * --gradient_checkpointing: HF gradient checkpointing on the Qwen2 backbone (non-reentrant);
#   * train_conf.max_steps / warmup_steps (linear warmup then constant lr);
#   * prints trainable parameter counts and writes run_info.json (revisions, hashes, args) to model_dir;
#   * torch.cuda.reset_peak_memory_stats after model load so peak VRAM in train_stats is training-only;
#   * --reset_steps (OFF by default, so nothing changes for E2/E3): start the optimizer-step and epoch
#     counters at 0 even though the init checkpoint carries `step`/`epoch` scalars (train_utils.save_model
#     stores them inside the .pt). Needed by the E4 curriculum (reports/decisions.md 2026-08-29), where
#     every stage starts from the previous stage's checkpoint and gets its OWN --max_steps budget and its
#     OWN 50-step warmup ("warmup 50 на стадию"); without it a stage initialised from a checkpoint at
#     step 2600 with --max_steps 500 would stop before taking a single step.
#   * --mix_pattern / --mix_source (OFF by default, so nothing changes for E2/E3/E4): E6 Mixed-SFT
#     (reports/decisions.md 2026-08-30). With --mix_pattern the TRAIN stream is
#     src/training/mixed_dataset.MixedDataset -- one batch per optimizer step from ONE source, sources
#     cycled in the pattern (e.g. S,S,S,L), each source a stock Dataset over its own data.list with its own
#     token cap from the yaml `mix_sources` block; the cv stream is built exactly as in the default path.
#     --train_data stays required by the stock parser; in mixed mode it is only recorded in run_info.json
#     (the train sources are the --mix_source lists), see scripts/train_mix.sh.
from __future__ import print_function
import argparse
import datetime
import hashlib
import json
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import subprocess
import sys
import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKDIR = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
COSYVOICE_ROOT = os.environ.get('COSYVOICE_ROOT', 'third_party/CosyVoice')
for _p in (COSYVOICE_ROOT, os.path.join(COSYVOICE_ROOT, 'third_party', 'Matcha-TTS'), _WORKDIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import deepspeed

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.losses import DPOLoss
from src.training.cosyvoice_train.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--ref_model', required=False, help='ref model used in dpo')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--onnx_path', required=False, help='onnx path, which is required for online feature extraction')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--dpo',
                        action='store_true',
                        default=False,
                        help='Use Direct Preference Optimization')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=60,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False,
                        help='enable HF gradient checkpointing on the Qwen2 backbone (A6 deviation)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='stop after this many optimizer steps (overrides train_conf.max_steps)')
    parser.add_argument('--reset_steps', action='store_true', default=False,
                        help='ignore the step/epoch scalars stored in --checkpoint and restart the counters '
                             'and the LR warmup at 0 (A6, E4 curriculum stages; off by default)')
    parser.add_argument('--mix_pattern', default=None,
                        help='A6 E6 Mixed-SFT: cyclic source pattern for the train stream, e.g. S,S,S,L '
                             '(one batch per optimizer step from one source; off by default = stock single-source train stream)')
    parser.add_argument('--mix_source', action='append', default=None, metavar='NAME=DATA_LIST',
                        help='A6 E6 Mixed-SFT: data.list of one mix source (repeatable, e.g. S=... L=...); '
                             'caps come from the yaml mix_sources block; only used with --mix_pattern')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


@record
def main():
    args = get_args()
    # A6: stock code exports onnx_path here, which turns on ONLINE speech-token extraction (cosyvoice.utils.onnx
    # reads the env var at import time, i.e. when the yaml is loaded below). This recipe trains on OFFLINE
    # tokens stored in the parquet rows, so the env var is deliberately NOT set (asserted below).
    os.environ.pop('onnx_path', None)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # gan train has some special initialization logic
    gan = True if args.model == 'hifigan' else False

    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if gan is True:
        override_dict.pop('hift')
    if args.qwen_pretrain_path is not None:
        override_dict['qwen_pretrain_path'] = args.qwen_pretrain_path
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    if gan is True:
        configs['train_conf'] = configs['train_conf_gan']
    configs['train_conf'].update(vars(args))
    if args.max_steps is None:
        args.max_steps = configs['train_conf'].get('max_steps')

    # A6: speech tokens come from the parquet rows; the online ONNX extractor must not be active
    from cosyvoice.utils.onnx import online_feature
    assert online_feature is False, 'unset the onnx_path env var: offline speech tokens are required (A6 recipe)'

    # Init env for ddp
    init_distributed(args)

    # Get dataset & dataloader
    mix_info = None
    if args.mix_pattern is None:
        train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
            init_dataset_and_dataloader(args, configs, gan, args.dpo)
    else:
        # A6 E6 Mixed-SFT: the train stream cycles the pattern over per-source pipelines (each with its own
        # token cap from the yaml mix_sources block); the cv stream is the stock one. Only this branch
        # imports the mixing module, so a run without --mix_pattern builds exactly what it always built.
        from src.training.mixed_dataset import init_mixed_dataset_and_dataloader
        train_dataset, cv_dataset, train_data_loader, cv_data_loader, mix_info = \
            init_mixed_dataset_and_dataloader(args, configs, gan, args.dpo)
        logging.info('mixed train stream: pattern %s, sources %s (--train_data %s is only recorded, not read)',
                     mix_info['pattern'], mix_info['sources'], args.train_data)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # load checkpoint
    if args.dpo is True:
        configs[args.model].forward = configs[args.model].forward_dpo
    model = configs[args.model]
    start_step, start_epoch = 0, -1
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            # A6: an SFT checkpoint carries `step`/`epoch` (cosyvoice.utils.train_utils.save_model stores
            # them next to the weights). Resuming those counters is right for a continued run and WRONG for
            # a curriculum stage, which is a new run with its own --max_steps and its own warmup: see
            # --reset_steps above. Default behaviour (E2/E3) is unchanged.
            if args.reset_steps:
                logging.info('--reset_steps: ignoring checkpoint step=%s epoch=%s; counters restart at 0',
                             state_dict.get('step'), state_dict.get('epoch'))
            else:
                if 'step' in state_dict:
                    start_step = state_dict['step']
                if 'epoch' in state_dict:
                    start_epoch = state_dict['epoch']
        else:
            logging.warning('checkpoint {} do not exsist!'.format(args.checkpoint))

    # A6: trainable-module report (PLAN §11.1: LLM only; flow/hift are not even instantiated: override_dict -> None)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_module = {}
    for n, p in model.named_parameters():
        by_module[n.split('.')[0]] = by_module.get(n.split('.')[0], 0) + p.numel()
    logging.info('model %s: %d params, %d trainable (%.1f%%), by top-level module %s', args.model, n_total, n_train,
                 100.0 * n_train / max(n_total, 1), by_module)
    assert set(k for k in ('flow', 'hift', 'hifigan') if configs.get(k) is not None) == set(), 'only the llm must be built'
    if args.gradient_checkpointing:
        assert hasattr(model, 'llm') and hasattr(model.llm, 'model'), 'gradient checkpointing expects CosyVoice3LM.llm.model (HF)'
        model.llm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.llm.model.config.use_cache = False
        logging.info('gradient checkpointing enabled on %s', type(model.llm.model).__name__)

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan)
    warmup = int(configs['train_conf'].get('warmup_steps', 0) or 0)
    if warmup > 0 and configs['train_conf']['scheduler'] == 'constantlr':
        # A6: linear warmup then constant lr (stock ConstantLR has no warmup, stock WarmupLR decays as step^-0.5)
        from torch.optim.lr_scheduler import LambdaLR

        class WarmupConstantLR(LambdaLR):
            def __init__(self, opt, warmup_steps):
                super().__init__(opt, lambda st: min(1.0, (st + 1) / warmup_steps))

            def set_step(self, step):
                self.last_epoch = step
        scheduler = WarmupConstantLR(optimizer, warmup)
        logging.info('scheduler: linear warmup %d steps then constant lr %s', warmup, configs['train_conf']['optim_conf']['lr'])
    scheduler.set_step(start_step)
    if scheduler_d is not None:
        scheduler_d.set_step(start_step)

    # Save init checkpoints
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch
    save_model(model, 'init', info_dict)

    # DPO related
    if args.dpo is True:
        ref_model = deepcopy(configs[args.model])
        state_dict = torch.load(args.ref_model, map_location='cpu')
        ref_model.load_state_dict(state_dict, strict=False)
        dpo_loss = DPOLoss(beta=0.01, label_smoothing=0.0, ipo=False)
        # NOTE maybe it is not needed to wrap ref_model as ddp because its parameter is not updated
        ref_model = wrap_cuda_model(args, ref_model)
    else:
        ref_model, dpo_loss = None, None

    # Get executor
    executor = Executor(gan=gan, ref_model=ref_model, dpo_loss=dpo_loss)
    executor.step = start_step
    executor.max_steps = args.max_steps
    torch.cuda.reset_peak_memory_stats()
    if int(os.environ.get('RANK', 0)) == 0:
        def _sha(path):
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 24), b''):
                    h.update(chunk)
            return h.hexdigest()
        try:
            repo_rev = subprocess.check_output(['git', '-C', COSYVOICE_ROOT, 'rev-parse', 'HEAD'], text=True).strip()
        except Exception as e:  # noqa
            repo_rev = 'unknown: {}'.format(e)
        run_info = {'args': vars(args), 'cosyvoice_repo_rev': repo_rev, 'init_checkpoint': args.checkpoint,
                    'init_checkpoint_sha256': _sha(args.checkpoint) if args.checkpoint and os.path.exists(args.checkpoint) else None,
                    'config_sha256': _sha(args.config), 'train_data_sha256': _sha(args.train_data), 'cv_data_sha256': _sha(args.cv_data),
                    'n_params': n_total, 'n_trainable': n_train, 'params_by_module': by_module,
                    'torch': torch.__version__, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'gpu_name': torch.cuda.get_device_name(0), 'start_time': datetime.datetime.utcnow().isoformat() + 'Z',
                    'train_conf': {k: v for k, v in configs['train_conf'].items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}}
        if mix_info is not None:
            # A6 E6: the sources that were actually read (pattern, caps, lists + their sha256)
            run_info['mix'] = {'pattern': mix_info['pattern'], 'pattern_counts': mix_info['pattern_counts'],
                               'sources': {n: dict(r, data_list_sha256=_sha(r['data_list'])) for n, r in mix_info['sources'].items()}}
        with open(os.path.join(args.model_dir, 'run_info.json'), 'w') as f:
            json.dump(run_info, f, indent=2, default=str)

    # Init scaler, used for pytorch amp mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    print('start step {} start epoch {}'.format(start_step, start_epoch))

    # Start training loop
    for epoch in range(start_epoch + 1, info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        if gan is True:
            executor.train_one_epoc_gan(model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                                        writer, info_dict, scaler, group_join)
        else:
            executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join, ref_model=ref_model)
        dist.destroy_process_group(group_join)
        if executor.stop:
            logging.info('training finished: max_steps reached (step %d)', executor.step)
            break


if __name__ == '__main__':
    main()
