# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
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

# A6 COPY of cosyvoice/utils/executor.py (repo commit 074ca6d, read-only upstream). Deviations, applied
# identically to both arms:
#   * max_steps: stop after N optimizer steps (stock code is epoch-based only), then a final CV + save;
#   * per-step logging of tokens/step, padded positions, batch size, step time, peak VRAM, RSS to
#     <model_dir>/train_stats.jsonl and tensorboard (PLAN §11.3 "length/token distributions logged");
#   * an OOM on a batch is NOT swallowed: it is logged with the batch composition and re-raised;
#   * `source` / `source_epoch` per step in train_stats.jsonl (E6 Mixed-SFT, reports/decisions.md 2026-08-30):
#     the mix source the step's batch came from ('S'/'L') and that source's epoch; None for a non-mixed run
#     (no batch carries the key). Every previous field keeps its name and meaning.
import json
import logging
import resource
import time
from contextlib import nullcontext
import os

import torch
import torch.distributed as dist

from cosyvoice.utils.train_utils import update_parameter_and_lr, log_per_step, log_per_save, batch_forward, batch_backward, save_model, cosyvoice_join


class Executor:

    def __init__(self, gan: bool = False, ref_model: torch.nn.Module = None, dpo_loss: torch.nn.Module = None):
        self.gan = gan
        self.ref_model = ref_model
        self.dpo_loss = dpo_loss
        self.step = 0
        self.epoch = 0
        self.rank = int(os.environ.get('RANK', 0))
        self.device = torch.device('cuda:{}'.format(self.rank))
        self.max_steps = None          # set by train.py from train_conf
        self.stop = False              # set when max_steps reached
        self._stats_f = None
        self._t_last = None
        self.acc_tokens = 0            # tokens accumulated inside the current optimizer step
        self.acc_text = 0
        self.acc_positions = 0
        self.acc_samples = 0
        self.total_tokens = 0          # cumulative target speech tokens seen by the optimizer
        self.acc_sources = []          # mix source of every micro-batch of the current optimizer step (E6)
        self.source_steps = {}         # optimizer steps per mix source so far (E6; empty for a non-mixed run)

    def _log_step_stats(self, writer, info_dict, batch_dict):
        """Called after every micro-batch; writes one record per OPTIMIZER step (accum boundary)."""
        self.acc_tokens += int(batch_dict.get('n_speech_tokens', 0))
        self.acc_text += int(batch_dict.get('n_text_tokens', 0))
        self.acc_positions += int(batch_dict.get('padded_positions', 0))
        self.acc_samples += int(batch_dict.get('batch_size', len(batch_dict['utts'])))
        self.acc_sources.append((batch_dict.get('source'), batch_dict.get('source_epoch')))
        if (info_dict['batch_idx'] + 1) % info_dict['accum_grad'] != 0:
            return
        # E6: one source per optimizer step (accum_grad 1 is frozen); with accum_grad > 1 the micro-batches
        # could differ, then all names are joined with '+' rather than one being silently reported.
        names = [s for s, _ in self.acc_sources if s is not None]
        source = None if not names else (names[0] if len(set(names)) == 1 else '+'.join(names))
        epochs = [e for _, e in self.acc_sources if e is not None]
        source_epoch = None if not epochs else (epochs[0] if len(set(epochs)) == 1 else max(epochs))
        if source is not None:
            self.source_steps[source] = self.source_steps.get(source, 0) + 1
        now = time.time()
        step_time = (now - self._t_last) if self._t_last is not None else None
        self._t_last = now
        self.total_tokens += self.acc_tokens
        rec = {'step': self.step + 1, 'epoch': self.epoch, 'speech_tokens': self.acc_tokens, 'text_tokens': self.acc_text,
               'padded_positions': self.acc_positions, 'samples': self.acc_samples, 'step_time_sec': step_time,
               'loss': float(info_dict['loss_dict']['loss']) * info_dict['accum_grad'], 'acc': float(info_dict['loss_dict'].get('acc', 0.0)),
               'lr': info_dict.get('lr'), 'grad_norm': float(info_dict.get('grad_norm', 0.0)),
               'peak_vram_alloc_gb': torch.cuda.max_memory_allocated() / 1e9, 'peak_vram_reserved_gb': torch.cuda.max_memory_reserved() / 1e9,
               'rss_gb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 'total_speech_tokens': self.total_tokens,
               'source': source, 'source_epoch': source_epoch}
        if self.rank == 0:
            if self._stats_f is None:
                self._stats_f = open(os.path.join(info_dict['model_dir'], 'train_stats.jsonl'), 'a')
            self._stats_f.write(json.dumps(rec) + '\n')
            self._stats_f.flush()
            if writer is not None:
                for k in ('speech_tokens', 'padded_positions', 'samples', 'peak_vram_alloc_gb', 'total_speech_tokens'):
                    writer.add_scalar('TRAIN/{}'.format(k), rec[k], self.step + 1)
                if step_time is not None:
                    writer.add_scalar('TRAIN/step_time_sec', step_time, self.step + 1)
                for k, v in self.source_steps.items():   # E6: cumulative optimizer steps per mix source
                    writer.add_scalar('TRAIN/steps_source_{}'.format(k), v, self.step + 1)
        self.acc_tokens = self.acc_text = self.acc_positions = self.acc_samples = 0
        self.acc_sources = []

    def train_one_epoc(self, model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join, ref_model=None):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        if self.ref_model is not None:
            self.ref_model.eval()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                try:
                    with context():
                        info_dict = batch_forward(model, batch_dict, scaler, info_dict, ref_model=self.ref_model, dpo_loss=self.dpo_loss)
                        info_dict = batch_backward(model, scaler, info_dict)
                except torch.cuda.OutOfMemoryError:
                    logging.error('OOM at step %d batch_idx %d: batch_size %s speech_tokens %s padded_positions %s utts %s',
                                  self.step, batch_idx, batch_dict.get('batch_size'), batch_dict.get('n_speech_tokens'),
                                  batch_dict.get('padded_positions'), batch_dict['utts'])
                    raise

                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                log_per_step(writer, info_dict)
                self._log_step_stats(writer, info_dict, batch_dict)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                   (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                    self._t_last = time.time()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
                    if self.max_steps is not None and self.step >= self.max_steps:
                        logging.info('max_steps %d reached at epoch %d batch_idx %d', self.max_steps, self.epoch, batch_idx)
                        self.stop = True
                        break
        dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=(not self.stop))

    def train_one_epoc_gan(self, model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                           writer, info_dict, scaler, group_join):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    batch_dict['turn'] = 'discriminator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer_d, scheduler_d, scaler, info_dict)
                optimizer.zero_grad()
                log_per_step(writer, info_dict)
                with context():
                    batch_dict['turn'] = 'generator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                optimizer_d.zero_grad()
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                   (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        ''' Cross validation on
        '''
        logging.info('Epoch {} Step {} on_batch_end {} CV rank {}'.format(self.epoch, self.step + 1, on_batch_end, self.rank))
        model.eval()
        total_num_utts, total_loss_dict = 0, {}  # avoid division by 0
        for batch_idx, batch_dict in enumerate(cv_data_loader):
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            if self.gan is True:
                batch_dict['turn'] = 'generator'
            info_dict = batch_forward(model, batch_dict, None, info_dict)

            for k, v in info_dict['loss_dict'].items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = []
                total_loss_dict[k].append(v.mean().item() * num_utts)
            log_per_step(None, info_dict)
        for k, v in total_loss_dict.items():
            total_loss_dict[k] = sum(v) / total_num_utts
        info_dict['loss_dict'] = total_loss_dict
        log_per_save(writer, info_dict)
        model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end else 'epoch_{}_step_{}'.format(self.epoch, self.step + 1)
        save_model(model, model_name, info_dict)
