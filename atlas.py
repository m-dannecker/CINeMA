import os
import time
import matplotlib.pyplot as plt
import numpy as np
import copy as cp
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# import tsne
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import wandb as wd
from data_loading import dataset
from models.cina import CINA
from torch.cuda.amp import autocast, GradScaler
from utils import Criterion, compare2ref, save_atlas, save_img, create_coordinate_grid, scale_affine


class Atlas:
    def __init__(self, args):
        self.args = args
        self._init_attributes()
        bbox, cond_bounds = self._init_models(splits=['train'])
        self._load_datasets(bbox, cond_bounds)
        print("Train subjects: ")
        for sub in self.datasets['train']:
            sub_id = sub['sub_id']
            ses_id = sub['ses_id']
            print(f"{sub_id}_{ses_id}")
        print("--------")
        print("Validation subjects: ")
        for sub in self.datasets['val']:
            sub_id = sub['sub_id']
            ses_id = sub['ses_id']
            print(f"{sub_id}_{ses_id}")

        self._init_latents(splits=['train'])
        # get meomory size of self.datasets['train'] and self.datasets['val']
        self._init_conditions(splits=['train'])
        self._init_trafos(splits=['train'])
        self._init_optims(splits=['train'])
        self.criterion = torch.compile(Criterion(args).to(args['device']), disable=(not args['compile']))

    # run cina on test subjects to perform downstream tasks
    def inference(self):
        # steps of INFERENCE
        # 1. load model (load bbox, condition_bounds) of checkpoint
        # 2. treat test dataset as validation dataset --> special config_data yaml needs to specifiy test_data as val_data
        # 3. load datasets, use 0 subjects for training, unseen subs for testing
        # 4. run downstream task on test subjects
        #    - reconstruct subject (calling validate method)
        #    - query segmentation
        #    - compute PSNR, SSIM, DICE
        #    - predict scan age
        #    - predict birth age?
        #    - predict sex
        # 5. save results under test category
        print("Starting Inference... ")
        # metrics = []
        # for ip_deg in np.linspace(0.0, 1.0, 7):
        #     metrics = [self.interpolate_subjects(self.datasets['train'][0], self.datasets['train'][1], ip_deg, split='train')]
        #     self.log_validation(self.args, metrics, ip_deg, split='train')
        metrics_val = {}
        mtrcs = self.validate(epoch_train=0, split='val')
        metrics_val[0] = mtrcs
        return metrics_val

    def run(self):
        print("Starting training...")
        metrics_val = {}
        scaler = GradScaler() if self.args['amp'] else None
        if self.args['load_model']: self.validate(epoch_train=0)
        for epoch in range(self.args['epochs']):
            stime = time.time()
            self.models['train'].train()
            loss_histories = self.train(self.dataloaders['train'], scaler)
            self.log_loss_history(loss_histories, epoch, stime, split='train')

            if (epoch + 1) % self.args['val_every'] == 0:
                metrics = [self.reconstruct_subject(self.datasets['train'][0], epoch, split='train')]
                self.log_validation(self.args, metrics, epoch, split='train')
                mtrcs = self.validate(epoch_train=epoch)
                metrics_val[epoch] = mtrcs
        return metrics_val

    def validate(self, epoch_train, split='val'):
        # self.generate_conditional_data()
        if self.args['generate_cond_atlas']: self.generate_conditional_atlas()
        self._seed()
        self._init_trafos(splits=[split])
        self._init_latents(splits=[split])
        self._init_conditions(splits=[split])
        self._init_models(splits=[split])
        scan_age_acc_mean, sex_acc_mean, ba_mae_mean = 0, 0, 0
        if not self.args['val_init_by_mean']:
            scaler = GradScaler() if self.args['amp'] else None
            self._init_optims(splits=[split])
            best_loss_holdout = {'sr': 9e9, 'seg': 9e9}
            for epoch_val in range(self.args['val_epochs']):
                if split == 'val':
                    # run train without optimization on the hold-out set, save loss and check if early-stopping applies
                    loss_holdout = self.train(self.dataloaders[split], scaler, split=split, holdout=True)
                    if np.mean(loss_holdout['sr']) < best_loss_holdout['sr']:
                        best_loss_holdout['sr'] = np.mean(loss_holdout['sr'])
                        print(f"New best SR loss: {np.mean(loss_holdout['sr'])} at epoch: {epoch_val}.")
                    if np.mean(loss_holdout['seg']) < best_loss_holdout['seg']:
                        best_loss_holdout['seg'] = np.mean(loss_holdout['seg'])
                        print(f"New best Seg loss: {np.mean(loss_holdout['seg'])} at epoch: {epoch_val}.")
                    if self.args['logging']: wd.log({f'{split}/sr_loss_holdout':  best_loss_holdout['sr']})
                stime = time.time()
                if self.args['conditions']['sex']:
                    gt = self.datasets['val'].get_conditions(key='sex', normed=False).to(torch.int)
                    sex_pred_after = ((self.conditions['val'] >= 0.0) * 2 - 1).squeeze().to(torch.int)
                    sex_acc_mean = (sex_pred_after == gt).to(torch.float).mean().cpu().item()
                    # print mean and std
                    print(f"Accuracy sex prediction: {sex_acc_mean}")
                    if self.args['logging']: wd.log({f'{split}/sex_accuracy': sex_acc_mean})
                if self.args['conditions']['birth_age']:
                    gt = self.datasets['val'].get_conditions(key='birth_age', normed=False)
                    min_, max_ = self.datasets['val'].condition_bounds['birth_age']['min'], self.datasets['val'].condition_bounds['birth_age']['max']
                    pred = ((self.conditions['val'] / self.args['cond_scale'] + 1) / 2) * (max_ - min_) + min_
                    # mae = torch.nn.functional.l1_loss(pred, gt.unsqueeze(1)).item()
                    ba_mae_mean = torch.abs(pred - gt.unsqueeze(1)).mean().item()
                    std_abs_error = torch.abs(pred - gt.unsqueeze(1)).std().item()
                    print(f"MAE: {ba_mae_mean} +/- {std_abs_error}")
                    if self.args['logging']: wd.log({f'{split}/birth_age_mae': ba_mae_mean})
                # self.svr(key='scan_age')
                svr_acc, scan_age_acc_mean, pca_acc = self.svm_classifier(key='scan_age')

                loss_histories = self.train(self.dataloaders[split], scaler, split=split)
                self.log_loss_history(loss_histories, (epoch_train, epoch_val), stime, split=split)

        metrics = []
        for idx in range(len(self.datasets[split])):
            metrics.append(self.reconstruct_subject(self.datasets[split][idx],
                                                    (epoch_train, self.args['val_epochs'] - 1), split=split))
        metrics_mean = self.log_validation(self.args, metrics, epoch_train, split)
        metrics_mean['scan_age_acc'] = scan_age_acc_mean
        metrics_mean['sex_acc'] = sex_acc_mean
        metrics_mean['birth_age_mae'] = ba_mae_mean

        # if not self.args['test_time']: self.svm_classifier(key='scan_age')
        self.save_model(None, metrics_mean['dice'][0].mean())
        return metrics_mean

    def train(self, dataloader, scaler=None, split='train', holdout=False):
        loss_histories = {'sr': [], 'seg': [], 'lat_reg': [], 'contrastive': [], 'trafo': [], 'total': []}
        seg_weight = self.args['seg_weight'] if split == 'train' else float(self.args['seg_optim'])
        sr_weight = 1.0 if split == 'train' else 1.0 - float(self.args['seg_optim'])
        n_smpls = self.args['n_samples'][split]
        for batch in dataloader:
            batch = self.batch2device(batch, holdout)
            losses = self.train_inner(batch, sr_weight, seg_weight, n_smpls, split, scaler, holdout)
            for key, val in losses.items():
                loss_histories[key].append(np.mean(val))
        if self.scheduler is not None and not holdout:
            self.scheduler[split].step()
        return loss_histories

    def train_inner(self, batch, sr_weight, seg_weight, n_smpls, split, scaler, holdout=False):
        losses = {'sr': [], 'seg': [], 'lat_reg': [], 'contrastive': [], 'trafo': [], 'total': []}
        for step in range(0, batch['size'], n_smpls):
            coords_nrmd = batch['coords_nrmd'][step:step + n_smpls]
            values = batch['values'][step:step + n_smpls]
            latents = self.latents[split][batch['idcs'][step:step + n_smpls]]
            latents = torch.nn.functional.grid_sample(latents, coords_nrmd[:, None, None, None, :],
                                                            align_corners=True, mode='bilinear', padding_mode='border').squeeze()
            conds_nrmd = self.conditions[split] if len(self.conditions[split]) == 0 \
                else self.conditions[split][batch['idcs'][step:step + n_smpls]]

            latents = torch.concat((latents, conds_nrmd), dim=-1)
            trafos = self.trafos[split][batch['idcs'][step:step + n_smpls]] \
                if self.args['trafo_dim'][split] else None
            self.optim[split].zero_grad(set_to_none=True)
            with autocast(enabled=self.args['amp']), torch.set_grad_enabled(not holdout):
                output = self.models[split](coords_nrmd, latents, trafos)
                loss = self.criterion(output, values, latents, trafos, sr_weight, seg_weight)

            if not holdout: # do not optimize on holdout set
                if scaler is not None:
                    scaler.scale(loss['total']).backward()
                    scaler.step(self.optim[split])
                    scaler.update()
                else:
                    loss['total'].backward()
                    self.optim[split].step()

            for key, val in loss.items():
                losses[key].append(val.item())
        return losses

    def interpolate_subjects(self, subject1, subject2, ip_deg, split='train'):
        self.models[split].eval()
        new_affine, shape = scale_affine(subject1['affines'][0].clone(), self.args['spacing_rec'],
                                         self.datasets[split].bbox_vxl)
        coords = create_coordinate_grid(shape=shape, device=self.args['device'])

        with torch.no_grad(), autocast(enabled=self.args['amp']):
            trafos = self.trafos[split][subject1[f"idx_{split}"], None].expand(coords.shape[0], -1) if \
            self.args['trafo_dim'][split] else None
            latents1 = self.latents[split][subject1[f"idx_{split}"]]
            latents2 = self.latents[split][subject2[f"idx_{split}"]]
            latents_ip = latents1 + (latents2 - latents1) * ip_deg
            latents = latents_ip[None, ...].expand(coords.shape[0], -1, -1, -1, -1)
            latents = torch.nn.functional.grid_sample(latents, coords[:, None, None, None, :],
                                                      align_corners=True, mode='bilinear',
                                                      padding_mode='border').squeeze()
            conds = self.conditions[split][subject1[f"idx_{split}"]][None, ...].expand(coords.shape[0], -1) if \
                len(self.conditions[split]) > 0 else self.conditions[split]
            latents = torch.concat((latents, conds), dim=-1)
            modalities_rec = self.models[split].inference(coords, latents, img_shape=shape.tolist(), trafos=trafos)
            modalities_rec = modalities_rec[..., :self.sr_dims + 1]  # discard certainty maps for now
        del conds
        del latents
        torch.cuda.empty_cache()
        return self.compute_metrics(subject1, modalities_rec, new_affine, ip_deg, split)

    def reconstruct_subject(self, subject, epoch, split='train'):
        self.models[split].eval()
        new_affine, shape = scale_affine(subject['affines'][0].clone(), self.args['spacing_rec'],
                                         self.datasets[split].bbox_vxl)
        coords = create_coordinate_grid(shape=shape, device=self.args['device'])

        with torch.no_grad(), autocast(enabled=self.args['amp']):
            trafos = self.trafos[split][subject[f"idx_{split}"], None].expand(coords.shape[0], -1) if self.args['trafo_dim'][split] else None
            latents = self.latents[split][subject[f"idx_{split}"]]
            latents = latents[None, ...].expand(coords.shape[0], -1, -1, -1, -1)
            latents = torch.nn.functional.grid_sample(latents, coords[:, None, None, None, :],
                                                      align_corners=True, mode='bilinear', padding_mode='border').squeeze()
            conds = self.conditions[split][subject[f"idx_{split}"]][None, ...].expand(coords.shape[0], -1) if \
                len(self.conditions[split]) > 0 else self.conditions[split]
            latents = torch.concat((latents, conds), dim=-1)
            modalities_rec = self.models[split].inference(coords, latents, img_shape=shape.tolist(), trafos=trafos)
            modalities_rec = modalities_rec[..., :self.sr_dims+1] # discard certainty maps for now
        del conds
        del latents
        torch.cuda.empty_cache()
        return self.compute_metrics(subject, modalities_rec, new_affine, epoch, split)

    def get_condition_interval(self, key, normalize=False):
        min_, max_ = (self.datasets['val'].condition_bounds[key]['min'].to(self.args['device']),
                      self.datasets['val'].condition_bounds[key]['max'].to(self.args['device']))
        if len(min_.shape) == 0:
            interval = torch.linspace(min_, max_, self.args['cond_steps']).to(self.args['device'])
        else:
            interval = torch.stack([torch.linspace(min_[i], max_[i], self.args['cond_steps']).to(self.args['device'])
                                    for i in range(len(min_))], dim=1)
        if normalize:
            interval = (self.normalize(interval, (min_, max_), pre_fac=self.args['cond_scale'])
                        .to(torch.float).reshape(self.args['cond_steps'], -1))
        return min_, max_, interval

    def generate_conditional_atlas(self):
        s_id = None
        self.models['train'].eval()
        atlases = []
        atlas_affine, atlas_shape = scale_affine(torch.eye(4), self.args['spacing_atlas'],
                                                 self.datasets['train'].bbox_world)
        coords = create_coordinate_grid(shape=atlas_shape, device=self.args['device'])
        for (cond_key, cond_flag) in self.args['conditions'].items():
            if cond_flag:
                min_, max_, interval = self.get_condition_interval(cond_key, normalize=True)
                for cond_value in interval:
                    atlases.append(self.generate_temporal_atlas(cond_value * 1.1, coords, atlas_shape=atlas_shape, s_id=s_id))
        if len(atlases) == 0: # generate atlas without conditions
            for dname in self.args['datasets']:
                atlases.append(self.generate_temporal_atlas(torch.empty((0,)), coords, atlas_shape=atlas_shape, c_l=dname))
        atlases = torch.stack(atlases, dim=0)
        if self.args['save_cond_atlas']:
            save_atlas(atlases, atlas_affine, args=self.args, individually=False)
        self.models['train'].train()
        return atlases

    def generate_conditional_data(self):
        num_t = (self.args['constraints']['scan_age'][1] -self.args['constraints']['scan_age'][0]) // self.args['age_step']
        s_id = [(int(self.args['constraints']['scan_age'][0]) + 1), len(self.latents['train']) // num_t]
        self.models['train'].eval()
        atlases = []
        atlas_affine, atlas_shape = scale_affine(torch.eye(4), self.args['spacing_atlas'],
                                                 self.datasets['train'].bbox_world)
        coords = create_coordinate_grid(shape=atlas_shape, device=self.args['device'])
        for (cond_key, cond_flag) in self.args['conditions'].items():
            if cond_flag:
                min_, max_, interval = self.get_condition_interval(cond_key, normalize=True)
                for cond_value in interval:
                    atlases.append(self.generate_temporal_atlas(cond_value * 1.0, coords, atlas_shape=atlas_shape, s_id=s_id))
        if len(atlases) == 0: # generate atlas without conditions
            for dname in self.args['datasets']:
                atlases.append(self.generate_temporal_atlas(torch.empty((0,)), coords, atlas_shape=atlas_shape, c_l=dname))
        atlases = torch.stack(atlases, dim=0)
        if self.args['save_cond_atlas']:
            save_atlas(atlases, atlas_affine, args=self.args, individually=False)
        self.models['train'].train()
        return atlases

    def generate_temporal_atlas(self, conditions, coords, atlas_shape, c_l=None, s_id=None):
        temporal_atlas = []
        for age in range(int(self.args['constraints']['scan_age'][0]) + 1,
                         int(self.args['constraints']['scan_age'][1]), self.args['age_step']):
            age_nrmd = self.normalize(age, self.args['constraints']['scan_age'], pre_fac=1.0)
            with torch.no_grad(), autocast(enabled=self.args['amp']):
                mean_latent = self.generate_mean_latent(age_nrmd, cond_key='scan_age', cond_latents=c_l,
                                                        std=self.args['gauss_sigma'])
                if s_id is not None: mean_latent = self.latents['train'][(age - s_id[0]) * s_id[1] - 1]
                mean_latent = mean_latent.expand(coords.shape[0], -1, -1, -1, -1)
                mean_latent = torch.nn.functional.grid_sample(mean_latent, coords[:, None, None, None, :],
                                                          align_corners=True, mode='bilinear', padding_mode='border').squeeze()
                mean_latent = torch.concat((mean_latent,
                                            conditions.to(mean_latent.device).expand(coords.shape[0], -1)), dim=-1)
                temporal_atlas.append(
                    self.models['train'].inference(coords, mean_latent, img_shape=atlas_shape.tolist()).detach().cpu())
                del mean_latent
                torch.cuda.empty_cache()
        return torch.stack(temporal_atlas, dim=-1)

    def generate_mean_latent(self, cond_value_nrmd, cond_key, cond_latents=None, std=1.0):
        # Generate mean latent for a given condition value. Here in the sense that we only use similar latents,
        # e.g., latents of fetals for inference on a fetal subject. This stabilizes inference.
        if cond_latents and not self.args['test_time']:
            # TODO: switch to self.latent_attrs['train']['dataset'] instead of self.datasets['train'].subjects
            lat_idcs = (np.array([sub['dataset'] for sub in self.datasets['train'].subjects]) == cond_latents)
            latents = self.latents['train'][lat_idcs]
        else:
            latents = self.latents['train']
            lat_idcs = np.ones(latents.shape[0], dtype=bool)

        if cond_key == 'scan_age':
            conds_nrmd = self.latents_attrs['train']['scan_age_nrmd'][lat_idcs, None]
        else:
            conds_nrmd = self.datasets['train'].get_conditions(key=cond_key, normed=True)[lat_idcs, None]

        latents = latents[:, :self.args['latent_dim'][0]]
        weights = torch.exp(-((conds_nrmd - cond_value_nrmd) / std) ** 2)
        weights = weights / torch.sum(weights)
        weighted_mean_latent = torch.sum(latents * weights[:, :, None, None, None], dim=0, keepdim=True)
        return weighted_mean_latent.detach().clone()

    def _init_attributes(self):
        self.optim = {}
        self.scheduler = {} if self.args['s_gamma'] else None
        self.latents = {"train": torch.empty(size=(0,)), "val": torch.empty(size=(0,))}
        self.latents_attrs = {"train": {}, "val": {}}
        self.trafos = {"train": torch.empty(size=(0,)), "val": torch.empty(size=(0,))}
        self.conditions = {}
        self.args['cond_dims'] = sum(1 if isinstance(v, int) and v != 0 else len([x for x in v if x != 0])
        if isinstance(v, list) else 0 for v in self.args['conditions'].values())
        self.sr_dims = sum(self.args['out_dim'][:-1])

    def _init_optims(self, splits=('train', 'val')):
        for split in splits:
            params = [{'name': f'latents_{split}',
                       'params': self.latents[split],
                       'lr': self.args['lr_latent'][split],
                       'weight_decay': self.args['wd_latent'][split]}]
            if split == 'train':
                params.append({'name': 'model_sr',
                               'params': self.models[split].parameters(),
                               'lr': self.args['lr'][split],
                               'weight_decay': 0.0})
            if split == 'val':
                params.append({'name': 'conditions',
                                 'params': self.conditions[split],
                                 'lr': self.args['lr_cond'][split],
                                 'weight_decay': 0.0})

            self.optim[split] = torch.optim.AdamW(params)
            if self.args['trafo_dim'][split]:
                self.optim[split].add_param_group({'name': f'trafos_{split}',
                                                   'params': self.trafos[split],
                                                   'lr': self.args['lr_trafo'][split],
                                                   'weight_decay': 0.0})
            if self.args['s_gamma']:
                self.scheduler[split] = torch.optim.lr_scheduler.StepLR(self.optim[split], step_size=1,
                                                                        gamma=self.args['s_gamma'][split])

    def _init_latents(self, splits=('train', 'val')):
        for split in splits:
            if split == 'val':
                if self.args['val_init_by_mean']:
                    self.latents[split]  = torch.empty((self.n_subjects[split], *self.args['latent_dim'])).to(self.args['device'])
                    for i, subject in enumerate(self.datasets[split]):
                        scan_age = self.normalize(subject['conditions']['scan_age'], self.args['constraints']['scan_age'], pre_fac=1.0)
                        idx = subject[f'idx_{split}']
                        mean_latent = self.generate_mean_latent(scan_age, cond_key='scan_age',
                                                                cond_latents=subject['dataset'],
                                                                std=self.args['gauss_sigma'])
                        self.latents[split][idx] = mean_latent
                else:
                    self.latents[split] = torch.normal(0, 0.01, size=(self.n_subjects[split], *self.args['latent_dim'])).to(self.args['device'])
                self.latents[split] = nn.Parameter(self.latents[split])
                self.latents_attrs[split] = {'scan_age': self.datasets[split].get_conditions('scan_age', normed=False),
                                             'scan_age_nrmd': self.datasets[split].get_conditions('scan_age', normed=True),
                                            'dataset': np.array([sub['dataset'] for sub in self.datasets[split].subjects])}
            elif not self.args['test_time']: # self.latents['train'].shape != (self.n_subjects[split], *self.args['latent_dim']):
                self.latents[split] = nn.Parameter(torch.normal(0, 0.01, size=(self.n_subjects[split],
                                                                               *self.args['latent_dim'])).to(self.args['device']))
                self.latents_attrs[split] = {'scan_age': self.datasets[split].get_conditions('scan_age', normed=False),
                                             'scan_age_nrmd': self.datasets[split].get_conditions('scan_age', normed=True),
                                             'dataset': np.array([sub['dataset'] for sub in self.datasets[split].subjects])}


    def _init_trafos(self, splits=('train', 'val')):
        for split in splits:
            # self.trafos[split] = nn.Parameter(torch.normal(0, 0.01, size=(self.n_subjects[split],
            #                                                               self.args['trafo_dim'][split]))
            #                                   .to(self.args['device'])) if self.args['trafo_dim'][split] else None
            if split=='train' and self.trafos['train'].shape == (self.n_subjects[split], self.args['trafo_dim'][split]):
                continue
            self.trafos[split] = nn.Parameter(torch.zeros(size=(self.n_subjects[split], self.args['trafo_dim'][split]))
                                              .to(self.args['device'])) if self.args['trafo_dim'][split] else None

    def _init_conditions(self, splits=('train', 'val')):
        for split in splits:
            if self.args['test_time'] and split == 'train':
                continue
            conds = []
            for key in self.args['conditions'].keys():
                if self.args['conditions'][key]:
                    conds.append(self.datasets[split].get_conditions(key, normed=True).view(self.n_subjects[split], -1))
            if len(conds) > 0:
                self.conditions[split] = torch.concat(conds, dim=-1) * self.args['cond_scale']
            else:
                self.conditions[split] = torch.tensor(conds, device=self.args['device'])
            if split == 'val':
                self.conditions[split] = nn.Parameter(self.conditions[split])
                if split == 'val' and not self.args['val_init_by_mean']:  # in val, if alignment through latent optim., conds should be estimated, thus zero init. as param
                    # self.conditions[split] = nn.Parameter(0.0 * self.conditions[split])
                    self.conditions[split] = nn.Parameter(torch.normal(0,  0.01, #self.args['cond_scale'],
                                                                       size=self.conditions[split].shape).to(self.args['device']))

    def _load_datasets(self, bbox=None, condition_bounds=None):
        print("Loading datasets...")
        stime = time.time()
        self.datasets = {'train': dataset.Brains3D(self.args, split='train', bbox=bbox, condition_bounds=condition_bounds)}
        bbox = (self.datasets['train'].bbox_vxl, self.datasets['train'].bbox_world)
        condition_bounds = self.datasets['train'].condition_bounds
        self.datasets['val'] = dataset.Brains3D(self.args, bbox, condition_bounds, split='val')
        for split in ['train', 'val']:
            n_samples = len(self.datasets[split])
            if n_samples < self.args['batch_size'][split]:
                self.args['batch_size'][split] = n_samples
                print(f"Batch size larger than num samples. Batch size for {split} set to {n_samples}")

        self.dataloaders = {split: DataLoader(self.datasets[split], batch_size=self.args['batch_size'][split],
                                              collate_fn=self.datasets[split].collate_fn,
                                              shuffle=(split=='train'), pin_memory=True, num_workers=0,
                                              drop_last=(split=='train')) for split in ['train', 'val']}
        self.n_subjects = {split: len(self.datasets[split]) for split in ['train', 'val']}
        print("Datasets loaded. Subjects Train: {}, Val: {}. Time elapsed: {:.2f}".format(self.n_subjects['train'],
                                                                                          self.n_subjects['val'],
                                                                                          time.time() - stime))

    def _init_models(self, splits=('train', 'val')):
        bbox, condition_bounds = None, None
        if 'train' in splits:
            self.models = {'train': CINA(self.args).to(self.args['device'])}
            if self.args['load_model']:
                bbox, condition_bounds = self.load_model()
            self.models['train'] = torch.compile(self.models['train'], disable=not self.args['compile'])
        if 'val' in splits:
            self.models['val'] = cp.deepcopy(self.models['train'])
            self.models['val'].eval()
        return bbox, condition_bounds

    def batch2device(self, batch, holdout=False):
        if holdout:  # replace batch values with holdout values
            for key, val in batch.items():
                if '_holdout' in key:
                    key = key.replace('_holdout', '')
                    batch[key] = val
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(self.args['device'])
        return batch

    def compute_metrics(self, subject, modalities_rec, new_affine, epoch, split):
        reg_type = 'Translation' # "SyN" if self.args['val_init_by_mean'] else "Rigid"
        bg_label = self.args['label_names'].index('BG')
        # num_mods = len(self.args['modalities']['names'])
        metrics = {'psnr': None, 'ssim': None, 'ncc': None, 'dice': [], 'modalities_rec': modalities_rec,
                   'new_aff': new_affine, 'modalities_ref': subject['modalities'], 'ref_aff': subject['affines'][0],
                   'sub_id': f'{subject["sub_id"]}_{subject["ses_id"]}'}

        (metrics['modalities_rec'], metrics['new_aff'], metrics['psnr'],
         metrics['ssim'], metrics['ncc'], metrics['dice']) = compare2ref(self.args, modalities_rec, subject['modalities'],
                                                                           aff_rec=new_affine,
                                                                           aff_ref=subject['affines'][0],
                                                                           reg_type=reg_type, bg_label=bg_label,
                                                                           num_classes=len(self.args['label_names']),
                                                                           crop2bbox=True)

        print(f"{split}, epoch {epoch}: Subject: {subject['sub_id']}_{subject['ses_id']}, "
              f"PSNR: {metrics['psnr']}, SSIM: {metrics['ssim']}, NCC: {metrics['ncc']}, "
              f"DICE: {metrics['dice'].tolist()}, DICE_MEAN: {np.mean(metrics['dice'].tolist())}")
        return metrics

    def log_loss_history(self, loss_histories, epoch, stime, split='train'):
        sr_mean = torch.mean(torch.tensor(loss_histories['sr']))
        seg_mean = torch.mean(torch.tensor(loss_histories['seg']))
        lat_reg_mean = torch.mean(torch.tensor(loss_histories['lat_reg']))
        cont_mean = torch.mean(torch.tensor(loss_histories['contrastive']))
        total_mean = torch.mean(torch.tensor(loss_histories['total']))
        lat_magnitude = torch.mean(torch.linalg.vector_norm(self.latents[split], dim=-1))
        trafo_loss = torch.mean(self.trafos[split] ** 2) if self.trafos[split] is not None else 0
        print(f'{split}: epoch {epoch}, loss_sr: {sr_mean:.4f}, loss_seg: {seg_mean:.4f}, lat_reg: {lat_reg_mean:.4f}, '
              f'loss_contrastive: {cont_mean:.4f}, lat_magnitude: {lat_magnitude:.4f}, '
              f'loss_trafo: {trafo_loss:.4f}, ' f'loss_total: {total_mean:.4f}, time: {time.time() - stime:.2f}')

        if self.args['logging']:
            wd.log({'{}/loss_sr'.format(split): torch.mean(torch.tensor(loss_histories['sr'])),
                    '{}/loss_seg'.format(split): torch.mean(torch.tensor(loss_histories['seg'])),
                    '{}/loss_lat_reg'.format(split): torch.mean(torch.tensor(loss_histories['lat_reg'])),
                    '{}/loss_contrastive'.format(split): torch.mean(torch.tensor(loss_histories['contrastive'])),
                    '{}/lat_magnitude'.format(split): lat_magnitude,
                    '{}/loss_trafo'.format(split): trafo_loss,
                    '{}/loss_total'.format(split): torch.mean(torch.tensor(loss_histories['total'])),
                    'epoch': epoch})

    @staticmethod
    def log_validation(args, metrics_dict, epoch, split='train'):
        metrics = {}
        for i, mod in enumerate(args['modalities']['names'][:-1]):
            metrics[f'psnr_{mod}'] = [torch.mean(torch.tensor([entry['psnr'][i] for entry in metrics_dict])),
                                      torch.std(torch.tensor([entry['psnr'][i] for entry in metrics_dict]))]
            metrics[f'ssim_{mod}'] = [torch.mean(torch.tensor([entry['ssim'][i] for entry in metrics_dict])),
                                        torch.std(torch.tensor([entry['ssim'][i] for entry in metrics_dict]))]
            metrics[f'ncc_{mod}'] = [torch.mean(torch.tensor([entry['ncc'][i] for entry in metrics_dict])),
                                        torch.std(torch.tensor([entry['ncc'][i] for entry in metrics_dict]))]
            print(f"{split} validation images: "
                  f"PSNR_{mod}: {metrics[f'psnr_{mod}'][0]} +/- {metrics[f'psnr_{mod}'][1]}, "
                  f"SSIM_{mod}: {metrics[f'ssim_{mod}'][0]} +/- {metrics[f'ssim_{mod}'][1]}, "
                  f"NCC_{mod}: {metrics[f'ncc_{mod}'][0]} +/- {metrics[f'ncc_{mod}'][1]}")

        metrics['dice'] = (torch.mean(torch.stack([entry['dice'] for entry in metrics_dict]), dim=0),
                          torch.std(torch.stack([entry['dice'] for entry in metrics_dict]), dim=0))
        print(f"DICE: {metrics['dice'][0].tolist()}, DICE_STD: {metrics['dice'][1].tolist()}, "
              f"DICE_MEAN: {metrics['dice'][0].mean()}, DICE_MEAN_STD: {metrics['dice'][1].mean()}")


        num_mods = len(args['modalities']['names'])
        if args['logging']:
            num_classes = len(args['label_names'])
            fig, axs = plt.subplots(len(metrics_dict), num_mods*2, figsize=(15, 5 * len(metrics_dict)))
            axs_flat = axs.flatten()
            ax_counter = 0
            for h, sub in enumerate(metrics_dict):
                for i in range(num_mods):
                    mod_rec = sub['modalities_rec'][..., i].cpu().numpy()
                    mod_ref = sub['modalities_ref'][i].cpu().numpy()
                    mod_rec = np.flip(mod_rec[:, :, mod_rec.shape[2] // 2]).T
                    mod_ref = np.flip(mod_ref[:, :, mod_ref.shape[2] // 2]).T
                    cmap = 'gray' if i < len(sub['modalities_ref']) - 1 else 'viridis'
                    vmax_rec = 1 if i < len(sub['modalities_ref']) - 1 else num_classes - 1
                    vmax_ref = mod_ref.max() if i < len(sub['modalities_ref']) - 1 else num_classes - 1
                    axs_flat[ax_counter].imshow(mod_rec, cmap=cmap, vmin=0, vmax=vmax_rec)
                    axs_flat[ax_counter+1].imshow(mod_ref, cmap=cmap, vmin=0, vmax=vmax_ref)
                    axs_flat[ax_counter].set_title(f'{args["modalities"]["names"][i]}_rec {sub["sub_id"]}')
                    axs_flat[ax_counter+1].set_title(f'{args["modalities"]["names"][i]}_ref {sub["sub_id"]}')
                    axs_flat[ax_counter].axis('off')
                    axs_flat[ax_counter+1].axis('off')
                    ax_counter += 2

            fig.tight_layout()
            for i, mod in enumerate(args['modalities']['names'][:-1]):
                wd.log({f'{split}/psnr_{mod}': metrics[f'psnr_{mod}'][0]})
                wd.log({f'{split}/ssim_{mod}': metrics[f'ssim_{mod}'][0]})
                wd.log({f'{split}/ncc_{mod}': metrics[f'ncc_{mod}'][0]})
            wd.log({f'{split}/dice': metrics['dice'][0].mean(), 'epoch': epoch, f'{split}/images': fig})
            plt.close(fig)

        if args['save_imgs'][split]:
            for h, sub in enumerate(metrics_dict):
                for i in range(len(sub['modalities_ref'])):
                    mod_rec, mod_ref = sub['modalities_rec'][..., i], sub['modalities_ref'][i]
                    name_rec = f'{split}/{sub["sub_id"]}_{args["modalities"]["names"][i]}_ep={epoch}_rec.nii.gz'
                    name_ref = f'{split}/{sub["sub_id"]}_{args["modalities"]["names"][i]}_ep={epoch}_ref.nii.gz'
                    aff_rec, aff_ref = sub['new_aff'], sub['ref_aff']
                    save_img(mod_rec.detach().cpu().numpy(), aff_rec, args['output_path'], filename=name_rec)
                    save_img(mod_ref.detach().cpu().numpy(), aff_ref, args['output_path'], filename=name_ref)
        return metrics

    @staticmethod
    def normalize(value, min_max, pre_fac=1.0):
        return pre_fac * ((value - min_max[0]) / (min_max[1] - min_max[0]) * 2 - 1)

    def save_model(self, epoch=None, target_metric=-9999):
        if self.args['save_model']:
            state_dict = {'model_train': self.models['train'].state_dict(),
                          'optim_train': self.optim['train'].state_dict(),
                          'latents_train': self.latents['train'],
                          'latents_attrs_train': self.latents_attrs['train'],
                          'conditions_train': self.conditions['train'],
                          'trafos_train': self.trafos['train'],
                          'scan_age_min': self.args['constraints']['scan_age'][0],
                          'scan_age_max': self.args['constraints']['scan_age'][1],
                          'bbox': (self.datasets['train'].bbox_vxl, self.datasets['train'].bbox_world),
                          'condition_bounds': self.datasets['train'].condition_bounds}
            filename = f"implicit_atlas_ep={epoch}.pt" if epoch is not None else "implicit_atlas_latest.pt"
            os.makedirs(self.args['output_path'], exist_ok=True)
            torch.save(state_dict, os.path.join(self.args["output_path"], filename))
            print("Model saved at: ", os.path.join(self.args["output_path"], filename))
            if target_metric > self.args["best_target_metric"]:
                self.args['best_target_metric'] = target_metric
                torch.save(state_dict, os.path.join(self.args['output_path'], "implicit_atlas_best.pt"))
                print("Best model saved at: ", os.path.join(self.args['output_path'], "implicit_atlas_best.pt"))

    def load_model(self, epoch=None):
        print("-----------Loading model...----------- \n"
              "Ensure model was trained on the same age range!")
        bbox = None
        filename = f"implicit_atlas_ep={epoch}.pt" if epoch is not None else "implicit_atlas_best.pt"
        filename = f"implicit_atlas_ep={epoch}.pt" if epoch is not None else "implicit_atlas_latest.pt"
        path_model = os.path.join(self.args['root_dir'], self.args['output_dir'], self.args['load_model'], filename)
        state_dict = torch.load(path_model)
        state_dict_train = state_dict['model_train']
        # replace '_orig_mod.' with '' in state_dict keys
        state_dict_train = {key.replace('_orig_mod.', ''): val for key, val in state_dict_train.items()}
        self.models['train'].load_state_dict(state_dict_train)
        # self.optim['train'] = torch.optim.AdamW(state_dict['optim_train'])
        # self.optim['train'].load_state_dict(state_dict['optim_train']['state'])

        print("Loading latents from state_dict...")
        self.latents['train'] = state_dict['latents_train']
        self.latents_attrs['train'] = state_dict['latents_attrs_train']
        print("Loading conditions from state_dict...")
        self.conditions['train'] = state_dict['conditions_train']
        print("Loading transformations from state_dict...")
        self.trafos['train'] = state_dict['trafos_train']
        bbox = state_dict['bbox']
        if 'condition_bounds' in state_dict.keys():
            condition_bounds = state_dict['condition_bounds']
            print("Condition bounds loaded from state_dict...")
        else:
            condition_bounds = None
            print("Condition bounds not found in state_dict.")
        print("Model loaded. \n------------------------------------")
        return bbox, condition_bounds

    def svr(self, key='scan_age'):
        svr = SVR(kernel='rbf', gamma='auto', epsilon=0.1)
        sc_X = StandardScaler()
        sc_Y = StandardScaler()
        X_train = self.latents['train'].detach().cpu().numpy().reshape(self.latents['train'].shape[0], -1)
        Y_train = self.latents_attrs['train'][key].cpu().numpy()
        # Y_train = self.datasets['train'].get_conditions(key, normed=False).detach().cpu().numpy()
        X_train_tf = sc_X.fit_transform(X_train)
        Y_train_tf = sc_Y.fit_transform(Y_train.reshape(-1,1))

        X_val = self.latents['val'].detach().cpu().numpy().reshape(self.latents['val'].shape[0], -1)
        Y_val = self.datasets['val'].get_conditions(key, normed=False).detach().cpu().numpy()
        X_val_tf = sc_X.fit_transform(X_val)
        Y_val_tf = sc_Y.fit_transform(Y_val.reshape(-1,1))

        svr.fit(X_train_tf, Y_train_tf.squeeze())
        Y_pred_tf = svr.predict(X_val_tf)
        Y_pred = sc_Y.inverse_transform(Y_pred_tf.reshape(-1,1))

        mae = np.abs(Y_val-Y_pred).mean()
        print("SVR MAE: ", mae)
        wd.log({f'{key}_svr_mae': mae})

    def svm_classifier(self, key='sex', val_ep=0):
        # Support Vector Machine classifier
        plt.close('all')
        clf = make_pipeline(StandardScaler(), SVC(gamma='scale')) if key == 'sex' \
            else make_pipeline(StandardScaler(), SVR())
        nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2, random_state=42))
        nca1 = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=1, random_state=42))
        knn = KNeighborsClassifier(n_neighbors=min(min(len(self.latents['val']), len(self.latents['train'])), 5))
        pca = PCA(n_components=1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=3)

        # Fit classifier
        targets = self.datasets['train'].get_conditions(key, normed=False).detach().cpu().numpy()
        targets = self.latents_attrs['train'][key].cpu().numpy()
        targets_rounded = np.round(targets).astype(int)
        lats_train = self.latents['train'].detach().cpu().numpy().reshape(self.latents['train'].shape[0], -1)
        # conds_train = self.conditions['train'].detach().cpu().numpy()
        # lats_train = np.concatenate((lats_train, conds_train), axis=1)

        gt = self.datasets['val'].get_conditions(key, normed=False).detach().cpu().numpy()
        gt_rounded = np.round(gt).astype(int)
        lats_val = self.latents['val'].detach().cpu().numpy().reshape(self.latents['val'].shape[0], -1)
        # conds_val = self.conditions['val'].detach().cpu().numpy()
        # lats_val = np.concatenate((lats_val, conds_val), axis=1)

        tsne_results = tsne.fit_transform(np.concatenate((lats_train, lats_val), axis=0))
        tsne_res_train = tsne_results[:len(lats_train)]
        tsne_res_val = tsne_results[len(lats_train):]
        targets_train_nrmd = (self.latents_attrs['train'][key+'_nrmd'].cpu().numpy()+1) / 2
        targets_val_nrmd = (self.datasets['val'].get_conditions(key, normed=True).detach().cpu().numpy()+1) / 2

        # plot tsne_results with colors of targets, plot train and val as dots and squares
        fig, ax = plt.subplots()
        ax.scatter(tsne_res_train[:, 0], tsne_res_train[:, 1], c=targets_train_nrmd, cmap='viridis', marker='o',
                   label='Train')
        ax.scatter(tsne_res_val[:, 0], tsne_res_val[:, 1], c=targets_val_nrmd, cmap='viridis', marker='s', label='Val')
        ax.set_title('t-SNE of latents colored by scan age')
        fig.suptitle('t-SNE of latents colored by scan age, train=o, val=s')
        plt.colorbar(ax.collections[0], ax=ax)
        ax.legend()
        # fig_path = os.path.join(self.args['output_path'], f'{key}_{val_ep}_tsne.png')
        # plt.savefig(fig_path)
        if self.args['logging']:
            wd.log({f'{key}_{val_ep}_tsne': wd.Image(fig)})
        plt.close(fig)


        num_classes = len(np.unique(targets_rounded))
        nca_acc = 0
        nca_std = 0
        if lats_train.shape[0] > num_classes:
            nca.fit(lats_train, targets_rounded)
            nca1.fit(lats_train, targets_rounded)
            knn.fit(nca.transform(lats_train), targets_rounded)
            nca_predictions = knn.predict(nca.transform(lats_val))
            nca_acc = np.sum(nca_predictions == gt_rounded) / len(gt_rounded) if key == 'sex' \
                else np.mean(np.abs(nca_predictions - gt_rounded.astype(int)))
            nca_std = np.std(np.abs(nca_predictions - gt_rounded.astype(int)))

        clf.fit(lats_train, targets_rounded)
        lats_pca = pca.fit_transform(np.concatenate((lats_train, lats_val), axis=0))
        lats_pca_train = lats_pca[:len(targets)]
        lats_pca_val = lats_pca[len(targets):]
        m, b = np.polyfit(lats_pca[:len(targets)].squeeze(), targets_rounded, 1)

        # plot pca results with targets as y values, val orange, train blue and log to wandb
        fig, ax = plt.subplots()
        ax.scatter(lats_pca_train, targets_rounded, c='tab:blue', marker='o')
        ax.scatter(lats_pca_val, gt_rounded, c='tab:orange', marker='s')
        ax.plot(lats_pca, m * lats_pca + b, c='red')
        ax.set_ylim(bottom=min(targets_rounded.min(), gt_rounded.min())-1, top=max(targets_rounded.max(), gt_rounded.max())+1)
        fig.suptitle('PCA of latents colored by scan age, train=o, val=s')
        # fig_path = os.path.join(self.args['output_path'], f'{key}_{val_ep}_pca.png')
        # plt.savefig(fig_path)
        if self.args['logging']:
            wd.log({f'{key}_{val_ep}_pca': wd.Image(fig)})

        # plt.plot(lats_pca, m * lats_pca + b, c='red')
        # plt.title('PCA of latents colored by scan age, train=o, val=s')
        # plt.show()
        plt.close(fig)


        # Predict
        svr_predictions = clf.predict(lats_val)
        lats_val_pca = pca.transform(lats_val)
        pca_predictions = lats_val_pca * m + b

        svr_acc = np.sum(svr_predictions == gt_rounded) / len(gt_rounded) if key == 'sex' \
            else np.mean(np.abs(svr_predictions - gt_rounded))
        svr_acc_std = np.std(np.abs(svr_predictions - gt_rounded))
        pca_acc = np.sum(pca_predictions == gt_rounded) / len(gt_rounded) if key == 'sex' \
            else np.mean(np.abs(pca_predictions - gt_rounded))
        pca_acc_std = np.std(np.abs(pca_predictions - gt_rounded))

        print(f"Accuracy of SVM classifier for {key}: {svr_acc} +/- {svr_acc_std}")
        print(f"Accuracy of NCA classifier for {key}: {nca_acc} +/- {nca_std}")
        print(f"Accuracy of PCA classifier for {key}: {pca_acc} +/- {pca_acc_std}")
        if self.args['logging']:
            wd.log({f'{key}_accuracy': svr_acc, f'{key}_accuracy_nca': nca_acc, f'{key}_accuracy_pca': pca_acc})
        return svr_acc, nca_acc, pca_acc

    def _seed(self):
        torch.manual_seed(self.args['seed'])
        torch.cuda.manual_seed(self.args['seed'])
        np.random.seed(self.args['seed'])
        # torch.backends.cudnn.deterministic = True
