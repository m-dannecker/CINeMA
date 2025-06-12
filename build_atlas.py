
import wandb as wd
import pandas as pd
import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from torch.cuda.amp import autocast, GradScaler
from models.inr_decoder import INR_Decoder, LatentRegressor
from data_loading.dataset import Data
from utils import *


class AtlasBuilder:
    """
    Class to build an atlas from a training dataset.
    """
    def __init__(self, args):
        self.args = args
        self.device = args['device']
        self.loss_criterion = Criterion(args).to(args['device'])
        self._init_atlas_training()
        self.train_on_data()

    def train_on_data(self):
        if len(self.args['load_model']['path']) > 0: self.validate(epoch_train=0) 
        loss_hist_epochs = []
        epoch_iterator = tqdm(range(self.args['epochs']['train']), desc="Training (epochs)", leave=True)
        for epoch in epoch_iterator:
            if self.args['optimizer']['re_init_latents']: self.re_init_latents()
            loss = self.train_epoch(epoch, split='train')
            loss_hist_epochs.append(loss)
            self.validate(epoch) 
            self._update_scheduler(split='train')
            epoch_iterator.set_postfix(loss=f"{np.mean(loss_hist_epochs):.4f}")
        
        return np.mean(loss_hist_epochs)

    def train_epoch(self, epoch, split):
        self.inr_decoder[split].train() if split == 'train' else self.inr_decoder[split].eval()
        loss_hist_batches = []
        batch_iterator = tqdm(self.dataloaders[split], 
                              desc=f"Training ({split}) (epochs)", leave=False)

        for batch in batch_iterator:
            loss = self.train_batch(batch, epoch, split)
            loss_hist_batches.append(loss)
            batch_iterator.set_postfix(loss=f"{np.mean(loss_hist_batches):.4f}")
        return np.mean(loss_hist_batches)

    def train_batch(self, batch, epoch, split='train'):
        loss_hist_samples = []
        n_smpls = self.args['n_samples']
        seg_weight = self.args['optimizer']['seg_weight'] if split == 'train' else 0.0
        coords_batch, values_batch, conditions_batch, idx_df_batch = to_device(batch)
        sample_iterator = tqdm(range(0, idx_df_batch.shape[0], n_smpls),
                               desc=f"Training ({split}) (samples in batch)", leave=False)
        
        start_time = time.time()
        for i, smpls in enumerate(sample_iterator):
            self.optimizers[split].zero_grad()
            coords = coords_batch[smpls:smpls + n_smpls]
            values = values_batch[smpls:smpls + n_smpls]
            idx_df = idx_df_batch[smpls:smpls + n_smpls].squeeze()
            # during validation we let the model predict the conditions
            conditions = conditions_batch[smpls:smpls + n_smpls] if split == 'train' else self.conditions_val[idx_df]

            with autocast(enabled=self.args['amp']):
                values_p = self.inr_decoder[split](coords, self.latents[split], conditions,
                                            self.transformations[split][idx_df], idcs_df=idx_df)
                loss = self.loss_criterion(values_p, values, self.transformations[split][idx_df], 
                                           seg_weight=seg_weight)

            if self.args['amp']:    
                self.grad_scalers[split].scale(loss['total']).backward()
                self.grad_scalers[split].step(self.optimizers[split])
                self.grad_scalers[split].update()
            else:
                loss['total'].backward()
                self.optimizers[split].step()

            loss_hist_samples.append(loss['total'].item())
            # Update tqdm every few steps to reduce overhead
            if i % 100 == 0 or i == len(sample_iterator) - 1:
                end_time = time.time()
                print(f"Time taken for {i} samples: {end_time - start_time:.2f} seconds")
                start_time = end_time
                log_loss(loss, epoch, split, self.args['logging'])
                sample_iterator.set_postfix(loss=f"{np.mean(loss_hist_samples):.4f}")
        
        return np.mean(loss_hist_samples)
    
    def validate(self, epoch_train):
        """
        Validate the model on the validation set, including:
        - optionally generate atlas and save to disk
        - generate training subjects and compute eval metrics, optionally save to disk
        - generate validation subjects and compute eval metrics, optionally save to disk
        - analyze latent space to predict attributes like sex, scan age, birth age
        - save model state
        """
        if self.args['generate_cond_atlas']: self.generate_atlas(epoch_train, n_max=100)
        metrics_train = self.generate_subjects_from_df(idcs_df=[0, 2, 3], epoch=epoch_train, split='train')
        log_metrics(self.args, metrics_train, epoch_train, df=self.datasets['train'].df, split='train')

        # --------- Start Actual Validation ---------
        if (epoch_train+1) % self.args['validate_every'] == 0 or epoch_train == self.args['epochs']['train'] - 1:
            self._init_validation() # reinitialize validation model to avoid information leakage
            for epoch_val in range(self.args['epochs']['val']):
                self.train_epoch(epoch=epoch_val, split='val') # fit latent codes and tfs to validation set
                self._update_scheduler(split='val') #TODO: update more often than every epoch
                self.analyze_latent_space(epoch_train, epoch_val=epoch_val)
            metrics_val = self.generate_subjects_from_df(idcs_df=range(len(self.datasets['val'])), 
                                                        epoch=epoch_val, split='val')
            log_metrics(self.args, metrics_val, epoch_train, df=self.datasets['val'].df, split='val')
        
        self.save_state(epoch_train)

    def generate_subject_from_latent(self, latent_vec, condition_vector, transformation=None, split='train'):
        """
        Generates a subject from a latent vector, a condition vector and optional transformation parameters.
        """
        grid_coords, grid_shape, affine = generate_world_grid(self.args, device=self.device)
        with torch.no_grad():
            volume_inf = self.inr_decoder[split].inference(grid_coords, latent_vec, condition_vector, 
                                                    grid_shape, transformation)
        return volume_inf

    def generate_subjects_from_df(self, idcs_df=None, epoch=0, split='train'):
        """
        (Re)Generate subjects from the current dataset via dataframe indices.
        """
        metrics = []
        for idx_df in idcs_df:
            df_row_dict = self.datasets[split].df.iloc[idx_df].to_dict()
            grid_coords, grid_shape, affine = generate_world_grid(self.args, device=self.device)
            with torch.no_grad():
                transformations = self.transformations[split][idx_df, None]
                conditions = self.datasets[split].load_conditions(df_row_dict).to(self.device)
                volume_inf = self.inr_decoder[split].inference(grid_coords, self.latents[split][idx_df:idx_df+1], 
                                                        conditions, grid_shape, transformations)
            if self.args['compute_metrics']: # compute metrics and save images if enabled
                metrics.append(compute_metrics(self.args, volume_inf, affine, df_row_dict, epoch, split))
            elif self.args['save_imgs'][split]: # save images if enabled
                save_subject(self.args, volume_inf, affine, df_row_dict, epoch, split)
        
        return metrics

    def generate_atlas(self, epoch=0, n_max=100):
        """
        Generate temporal atlas for each condition combination in self.args['atlas_gen']['conditions'].
        """
        self.inr_decoder['train'].eval()
        grid_coords, grid_shape, affine = generate_world_grid(self.args, device=self.device)
        temp_steps = self.args['atlas_gen']['temporal_values']
        atlas_list = []
        with torch.no_grad():
            for temp_step in temp_steps:
                temp_step_normed = normalize_condition(self.args, 'scan_age', temp_step)
                mean_latent = self.get_mean_latent('scan_age', temp_step_normed, n_max=n_max)
                condition_vectors = generate_combinations(self.args, self.args['atlas_gen']['conditions'])
                cond_list = []
                for c_v in condition_vectors:
                    c_v = torch.tensor(c_v, dtype=torch.float32).to(self.device)
                    values_p = self.inr_decoder['train'].inference(grid_coords, mean_latent, c_v, 
                                                                   grid_shape, None)
                    seg = values_p[:, :, :, -1]
                    seg[seg==4] = 0
                    values_p[:, :, :, -1] = seg
                    cond_list.append(values_p.detach().cpu())
                    # free up GPU memory
                    torch.cuda.empty_cache()
                atlas_list.append(torch.stack(cond_list, dim=-1))
        atlas_list = torch.stack(atlas_list, dim=-1) # [x, y, z, num_modalities, num_conditions, t]
        save_atlas(self.args, atlas_list, affine, temp_steps, condition_vectors, epoch=epoch)
        return atlas_list
    
    def get_mean_latent(self, condition_key, condition_mean, n_max=100, split='train'):
        """
        Regress gaussian weighted latent code from subjects weighted by distance to condition mean
        of the condition with condition_key. Weights are clipped to the closest n_max subjects.
        sigma is the standard deviation of the gaussian distribution used to weight the latents
        emperically we want +/- 2 stds (covering 95% of the weights) to span +/- "gaussian_span" weeks of scan age, e.g. 0.75 weeks.
        Therefore:
        - Full range of condition values is [-1, 1], i.e. 2. 
        - Full range of scan age is c_max - c_min = c_range, e.g. 46 - 37 = 9 for term neonates.
        - The ratio of condition values to weeks is 2 / c_range = c_ratio, e.g. 2 / 9 = 0.222 units per week.
        ==> 2 std = 0.75 weeks = 0.75 * c_ratio e.g. = 0.165 units.
        ==> sigma = 1 std = 0.5 * 0.75 weeks * c_ratio, e.g. = 0.0825 units for term neonates.
        # Finally, we scale the sigma by the condition scale factor in the config, as scan age is actually normalized to [-cond_scale, cond_scale]
        """
        c_ratio = 2 / (self.args['dataset']['conditions'][condition_key]['max'] - self.args['dataset']['conditions'][condition_key]['min'])
        span_weeks = self.args['atlas_gen']['gaussian_span']
        sigma = 0.5 * span_weeks * c_ratio
        sigma = sigma * self.args['atlas_gen']['cond_scale']

        latents = self.latents[split]
        condition_values, df_idcs = self.datasets[split].get_condition_values(condition_key, normed=True, device=self.device)
        assert len(condition_values) == len(latents), "Condition values (all entries from the dataframe) \
                                                       and latents must have the same length!"
        weights = torch.exp(-(condition_values - condition_mean)**2 / (2*(sigma**2)))
        n_max = min(n_max, len(weights))
        weights[torch.argsort(weights, descending=True)[n_max:]] = 0
        weights = weights / torch.sum(weights)
        weights = weights[:, None, None, None, None] # [n_subjects, *4D]
        mean_latent = torch.sum(latents * weights, dim=0, keepdim=True)
        return mean_latent
    
    def analyze_latent_space(self, epoch, epoch_val=0):
        args_LA = self.args['latent_anaylsis']
        args_C = self.args['dataset']['conditions']
        if not args_LA['activate']: return  # skip if not activated in config
        # conduct latent space analysis including
        # - birth age prediction from condition
        # - scan age prediction from latent_code
        if args_LA['predict_ext_condition'] != 'none':
            if args_C['birth_age'] and args_LA['predict_ext_condition'] == 'MAE':
                self.predict_ext_condition('birth_age', epoch, epoch_val)
            elif args_C['sex'] and args_LA['predict_ext_condition'] == 'CrossEntropy':
                self.predict_ext_condition('sex', epoch, epoch_val)
        
        if args_LA['predict_scan_age'] != 'none':
            self.predict_cond_value(epoch, epoch_val, cond_key='scan_age')
        if args_LA['predict_birth_age'] != 'none':
            self.predict_cond_value(epoch, epoch_val, cond_key='birth_age')
        if args_LA['ba_regression']['activate']:
            self.regress_latent_condition('birth_age', epoch, epoch_val)

    def predict_cond_value(self, epoch, epoch_val=0, k=5, cond_key='scan_age'):
        # Predict scan_age using either NCA, PCA, SVR
        # 1. train regression network on training latents 
        # 2. after each epoch, compute regression on validation latents
        # 3. compute metrics
        # 4. log metrics
        train_data = self.latents['train'].detach().cpu().clone().numpy()
        train_data = train_data.reshape(train_data.shape[0], -1)
        train_labels = self.datasets['train'].get_condition_values(cond_key, normed=False, device=self.device)[0].cpu().numpy()
        train_labels_rnd = np.round(train_labels).astype(int)
        val_data = self.latents['val'].detach().cpu().clone().numpy()
        val_data = val_data.reshape(val_data.shape[0], -1)
        val_labels = self.datasets['val'].get_condition_values(cond_key, normed=False, device=self.device)[0].cpu().numpy()
        val_labels_rnd = np.round(val_labels).astype(int)

        nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2, random_state=42))
        knn = KNeighborsClassifier(n_neighbors=min(min(len(self.latents['val']), len(self.latents['train'])), k))
        nca.fit(train_data, train_labels_rnd)
        knn.fit(nca.transform(train_data), train_labels_rnd)
        nca_predictions = knn.predict(nca.transform(val_data))
        nca_acc = np.mean(np.abs(nca_predictions - val_labels_rnd))
        nca_std = np.std(np.abs(nca_predictions - val_labels_rnd))

        print(f"Epoch_train {epoch}, Epoch_val {epoch_val}: Accuracy of NCA classifier for {cond_key}: {nca_acc:.3f} +/- {nca_std:.3f}")
        if self.args['logging']:
            wd.log({f'latent_anaylsis/{cond_key}_accuracy_nca': nca_acc})
        
    def predict_ext_condition(self, condition_key, epoch, epoch_val=0):
        # retrieve ground truth
        condition_values, df_idcs = self.datasets['val'].get_condition_values(condition_key, normed=False, device=self.device)
        cond_idx = list(self.args['atlas_gen']['conditions'].keys()).index(condition_key)
        condition_predictions = self.conditions_val[:, cond_idx]
        condition_predictions = denormalize_conditions(self.args, condition_key, condition_predictions)
        # compute metrics
        if self.args['latent_anaylsis']['predict_ext_condition'] == 'MAE':
            mae = torch.mean(torch.abs(condition_predictions - condition_values))
            value = mae.item()
            print(f"MAE for {condition_key}: {mae:.3f} at epoch {epoch}, epoch_val {epoch_val}")
        elif self.args['latent_anaylsis']['predict_ext_condition'] == 'CrossEntropy':
            ce = torch.nn.functional.cross_entropy(condition_predictions, condition_values)
            value = ce.item()
            print(f"Cross entropy for {condition_key}: {ce:.3f} at epoch {epoch}, epoch_val {epoch_val}")
        else:
            raise ValueError(f"Unknown metric {self.args['latent_anaylsis']['predict_ext_condition']}")
        if self.args['logging']:
            wd.log({f"latent_anaylsis/{condition_key}": value})

    def regress_latent_condition(self, condition_key, epoch_train=0, epoch_val=0):
        # 1. train regression network on training latents 
        # 2. after each epoch, compute regression on validation latents
        # 3. compute metrics
        # 4. log metrics
        train_data = self.latents['train'].detach().clone()
        train_labels = self.datasets['train'].get_condition_values(condition_key, normed=True, device=self.device)[0]
        val_data = self.latents['val'].detach().clone()
        val_labels = self.datasets['val'].get_condition_values(condition_key, normed=True, device=self.device)[0]
        regressor = LatentRegressor(self.args['inr_decoder']['latent_dim']).to(self.device)
        optimizer = optim.AdamW(regressor.parameters(), lr=self.args['latent_anaylsis']['ba_regression']['lr'],
                                weight_decay=0.0)
        batch_size = 32
        loss_fnc = torch.nn.L1Loss()
        regressor.train()
        regression_epochs = self.args['latent_anaylsis']['ba_regression']['epochs']
        best_score_val = float('inf')
        best_score_val_epoch = 0
        for epoch in range(regression_epochs):
            shuffle = np.random.permutation(len(train_data))
            train_data_sh = train_data[shuffle]
            train_labels_sh = train_labels[shuffle]
            loss_train_epoch = []
            for i in range(0, len(train_data_sh), batch_size):
                train_data_batch = train_data_sh[i:i+batch_size]
                train_labels_batch = train_labels_sh[i:i+batch_size]
                optimizer.zero_grad()
                pred_train = regressor(train_data_batch.squeeze()).squeeze()
                loss_train = loss_fnc(pred_train, train_labels_batch)
                loss_train.backward()
                optimizer.step()
                loss_train_epoch.append(loss_train.item())
            if epoch % 1 == 0:
                print(f"Epoch {epoch} - Loss: {np.mean(loss_train_epoch):.4f}")
                # 2. validate regression network on validation latents
                regressor.eval()
                with torch.no_grad():   
                    pred_val = regressor(val_data.squeeze())
                    loss_val = loss_fnc(pred_val, val_labels) 
                    print(f"Validation Loss: {loss_val.item():.4f}")
                    # compute metrics
                    pred_val_denormed = denormalize_conditions(self.args, condition_key, pred_val)
                    val_labels_denormed = denormalize_conditions(self.args, condition_key, val_labels)
                    mae = torch.mean(torch.abs(pred_val_denormed - val_labels_denormed))
                    print(f"MAE for {condition_key}: {mae:.3f} at epoch {epoch_train}, epoch_val {epoch_val}\n")
                    if mae < best_score_val:
                        best_score_val = mae
                        best_score_val_epoch = epoch
                    # log metrics   
                if self.args['logging']:
                    wd.log({f"latent_anaylsis/{condition_key}_regression_train": loss_train.item()})
                    wd.log({f"latent_anaylsis/{condition_key}_regression_val": loss_val.item()})
                    wd.log({f"latent_anaylsis/{condition_key}_regression_mae_val": mae.item()})
                regressor.train()
        print(f"Best MAE for {condition_key}: {best_score_val:.3f} at regression_epoch {best_score_val_epoch} for epoch_train {epoch_train}, epoch_val {epoch_val}")

    def save_state(self, epoch, split='train'):
        if self.args['save_model']:
            log_dir = self.args['output_dir']
            torch.save({
                'epoch': epoch,
                'latents': self.latents[split].cpu(),
                'transformations': self.transformations[split].cpu(),
                'inr_decoder': self.inr_decoder[split].state_dict(),
                'tsv_file': self.datasets[split].tsv_file,
                'dataset_df': self.datasets[split].df,
                'args': self.args
            }, os.path.join(log_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f'Saved model state to {os.path.join(log_dir, f"checkpoint_epoch_{epoch}.pth")}')
        else:
            print(f'Not saving model state as save_model is set to False')

    def load_checkpoint(self, chkp_path=None, epoch=None):  
        chkp_path = os.path.join(chkp_path, f'checkpoint_epoch_{epoch}.pth')
        if not os.path.exists(chkp_path):
            raise FileNotFoundError(f'State file {chkp_path} not found!')
        chkp = torch.load(chkp_path)
        # self.args = chkp['args']
        self._init_dataloading(chkp['tsv_file'], chkp['dataset_df'])
        self._init_inr(chkp['inr_decoder'], split='train')
        self._init_transformations(chkp['transformations'])
        self._init_latents(chkp['latents'])
        print(f'Loaded state from {chkp_path}')
    
    def _init_atlas_training(self):
        self.datasets, self.dataloaders = {}, {}
        self.inr_decoder, self.latents, self.transformations = {}, {}, {}
        self.optimizers, self.grad_scalers = {}, {}
        self.schedulers = {}
        chkp_path = self.args['load_model']['path']
        if len(chkp_path) > 0:
            self.load_checkpoint(chkp_path, self.args['load_model']['epoch'])
        else:
            self._init_dataloading(split='train')
            self._init_inr(split='train')
            self._init_transformations(split='train')
            self._init_latents(split='train')
        self._init_optimizer(split='train') # optimizer is not loaded from checkpoint
        self._init_dataloading(split='val')

    def _init_validation(self):
        self._seed()
        self._init_latents(split='val')
        self._init_transformations(split='val')
        self._init_optimizer(split='val')
        self.inr_decoder['val'] = copy.deepcopy(self.inr_decoder['train'])
        self.inr_decoder['val'].eval()

    def _init_dataloading(self, tsv_file=None, df_loaded=None, split='train'):
        shuffle = True if split == 'train' else False
        tsv_file =  pd.read_csv(self.args['dataset']['tsv_file'], sep='\t') if tsv_file is None else tsv_file
        self.datasets[split] = Data(self.args, tsv_file, split=split, df_loaded=df_loaded)
        self.dataloaders[split] = DataLoader(self.datasets[split], batch_size=self.args['batch_size'], 
                                             num_workers=self.args['num_workers'], shuffle=shuffle, 
                                             collate_fn=self.datasets[split].collate_fn, pin_memory=True)

        print(f"Initialized dataloader for {split} with {len(self.datasets[split])} subjects")

    def _init_inr(self, state_dict=None, split='train'):
        # get the number of active conditions
        self.args['inr_decoder']['cond_dims'] = sum([self.args['dataset']['conditions'][c] 
                                                     for c in self.args['dataset']['conditions']])
        self.inr_decoder[split] = INR_Decoder(self.args, self.device).to(self.device)
        if state_dict is not None:
            self.inr_decoder[split].load_state_dict(state_dict)

    def _init_transformations(self, tfs=None, split='train'):
        shape = (len(self.datasets[split]), max(self.args['inr_decoder']['tf_dim'], 6)) # at least 6 for rigid, 9 for rigid+scale
        tfs = torch.zeros(shape).to(self.device) if tfs is None else tfs.to(self.device)
        self.transformations[split] = nn.Parameter(tfs) if self.args['inr_decoder']['tf_dim'] > 0 else tfs # if tf_dim=0, set trafos to 0 and fix
        
    def _init_latents(self, lats=None, split='train'):
        shape = (len(self.datasets[split]), *self.args['inr_decoder']['latent_dim'])
        lats = torch.normal(0, 0.01, size=shape).to(self.device) if lats is None else lats.to(self.device)
        self.latents[split] = nn.Parameter(lats)
        if split == 'val': # need to initialize conditions as learnable parameters
            shape_cond_val = (len(self.datasets['val']), self.args['inr_decoder']['cond_dims'])
            self.conditions_val = nn.Parameter(torch.normal(0, 0.01, size=shape_cond_val).to(self.device))

    def re_init_latents(self, split='train'):
        self.latents[split].data.normal_(0, 0.01)
        self.transformations[split].data.zero_()
        self.optimizers[split].zero_grad()
        
    def _init_optimizer(self, split='train'):
        params = [{'name': f'latents_{split}',
                   'params': self.latents[split],
                   'lr': self.args['optimizer']['lr_latent'],
                   'weight_decay': self.args['optimizer']['latent_weight_decay']}]
        
        if self.args['inr_decoder']['tf_dim'] > 0:
            params.append({'name': f'transformations_{split}',
                           'params': self.transformations[split],
                           'lr': self.args['optimizer']['lr_tf'],
                           'weight_decay': self.args['optimizer']['tf_weight_decay']})
        if split == 'train':
            params.append({'name': f'inr_decoder',
                           'params': self.inr_decoder[split].parameters(),
                           'lr': self.args['optimizer']['lr_inr'],
                           'weight_decay': self.args['optimizer']['inr_weight_decay']})
        if split == 'val':
            params.append({'name': f'conditions_val',
                           'params': self.conditions_val,
                           'lr': self.args['optimizer']['lr_latent'],
                           'weight_decay': self.args['optimizer']['latent_weight_decay']})
        self.optimizers[split] = optim.AdamW(params)
        self.grad_scalers[split] = GradScaler() if self.args['amp'] else None
        if self.args['optimizer']['scheduler']['type'] == 'cosine':
            self.schedulers[split] = CosineAnnealingLR(self.optimizers[split], T_max=self.args['epochs'][split], 
                                                       eta_min=self.args['optimizer']['scheduler']['eta_min'])
        else:
            self.schedulers[split] = None

    def _update_scheduler(self, split='train'):
        if self.schedulers[split] is not None:
            self.schedulers[split].step()

    def _seed(self):
        torch.manual_seed(self.args['seed'])
        torch.cuda.manual_seed(self.args['seed'])
        np.random.seed(self.args['seed'])
    