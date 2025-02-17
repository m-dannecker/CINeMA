import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from data_loading.data_utils import *


class Brains3D(Dataset):
    def __init__(self, args, bbox=None, condition_bounds=None, split='train') -> None:
        self.args = args
        self.split = split
        self.subjects = []
        for dataset in self.args['datasets']:
            self.subjects.extend(load_subjects(self.args, dataset=dataset, start_idx=len(self.subjects), split=self.split))
        self.bbox_vxl, self.bbox_world = determine_bounding_box(self.subjects) if bbox is None else bbox
        self.condition_bounds = self.normalize_conditions(condition_bounds=condition_bounds)
        self._create_holdout_set()
        self.plot_histogramm('scan_age')
        self.plot_histogramm('birth_age')
        self.plot_histogramm('pnatal_age')
        # self.plot_volume_distribution(key='birth_age', label=0)


    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx):
        return self.subjects[idx]

    def normalize_conditions(self, condition_bounds=None):
        if condition_bounds is None:
            condition_bounds = {}
            cond_keys = ['scan_age'] + [key for key in self.subjects[0]['conditions'].keys() if self.args['conditions'][key]]
            for cond in cond_keys:
                if cond == 'rad_score':
                    continue
                elif cond == 'scan_age' or cond == 'birth_age':
                    min_ = torch.tensor(self.args['constraints'][cond][0])
                    max_ = torch.tensor(self.args['constraints'][cond][1])
                elif cond == 'label_vol':
                    min_, max_ = [], []
                    for i, label in enumerate(self.args['conditions']['label_vol']):
                        min_.append(torch.tensor([sub['conditions'][cond][i] for sub in self.subjects]).min())
                        max_.append(torch.tensor([sub['conditions'][cond][i] for sub in self.subjects]).max())
                    min_, max_ = torch.tensor(min_), torch.tensor(max_)
                else:
                    max_ = torch.tensor([sub['conditions'][cond] for sub in self.subjects]).max()
                    min_ = torch.tensor([sub['conditions'][cond] for sub in self.subjects]).min()
                condition_bounds[cond] = {'min': min_, 'max': max_}
        # normalize conditions if possible
        for cond in condition_bounds.keys():
            min_, max_ = condition_bounds[cond]['min'], condition_bounds[cond]['max']
            for sub in self.subjects:
                sub['conditions_normed'] = {} if 'conditions_normed' not in sub else sub['conditions_normed']
                sub['conditions_normed'][cond] = ((sub['conditions'][cond] - min_) / (max_-min_)) * 2 - 1
        return condition_bounds

    def collate_fn(self, subjects):
        batch = self.assemble_batch(subjects, holdout=False)
        batch_val = self.assemble_batch(subjects, holdout=True) if self.split == 'val' else {}
        batch = {**batch, **batch_val}
        return batch

    def assemble_batch(self, subjects, holdout=False):
        # if val, a hold-out set for early-stopping/validation during latent-optimization is being created
        suffix = '_holdout' if holdout else ''
        coords_batch = []
        values_batch = []
        conds_batch = []
        idcs_batch = []
        for sub in subjects:
            coords_batch.append(sub['coords'+suffix])
            values_batch.append(sub['values'+suffix])
            conds = []
            for key in self.args['conditions'].keys():
                if self.args['conditions'][key]:
                    conds.append(sub['conditions_normed'][key].view(1, -1).expand(sub['coords'+suffix].shape[0], -1))
            conds_batch.append(torch.cat(conds, dim=-1) if conds else torch.empty((sub['coords'+suffix].shape[0], 0)).to(torch.float))
            idx = sub[f'idx_{self.split}']
            idcs_batch.append(torch.ones_like(sub['coords'+suffix][:, 0]).to(torch.int64) * idx)
        coords = torch.cat(coords_batch, dim=0).to(torch.float)
        coords_normed = ((coords - self.bbox_vxl['min']) * self.bbox_vxl['divisor']) * 2 - 1
        values = torch.cat(values_batch, dim=0).to(torch.float)
        conds = torch.cat(conds_batch, dim=0).to(torch.float)
        idcs = torch.cat(idcs_batch, dim=0)
        perm = torch.randperm(coords.shape[0])
        return {'coords_nrmd'+suffix: coords_normed[perm], 'values'+suffix: values[perm],
                'conds_nrmd'+suffix: conds[perm], 'idcs'+suffix: idcs[perm], 'size'+suffix: coords.shape[0]}

    def _create_holdout_set(self):
        val_ratio = 0.1
        for sub in self.subjects:
            perm = torch.randperm(sub['coords'].shape[0])
            val_idcs = perm[:int(val_ratio * sub['coords'].shape[0])]
            train_idcs = perm[int(val_ratio * sub['coords'].shape[0]):]
            sub['coords_holdout'] = sub['coords'][val_idcs]
            sub['values_holdout'] = sub['values'][val_idcs]
            sub['coords'] = sub['coords'][train_idcs]
            sub['values'] = sub['values'][train_idcs]

    def get_scan_ages(self):
        scan_ages = [sub['conditions']['scan_age'] for sub in self.subjects]
        return torch.tensor(scan_ages).to(self.args['device'], torch.float)

    def get_conditions(self, key, normed=True):
        norm_key = 'conditions_normed' if normed else 'conditions'
        return torch.stack([sub[norm_key][key] for sub in self.subjects]).to(self.args['device'], torch.float)

    def plot_histogramm(self, key):
        fig = None
        if self.args['constraints'][key]:
            # plot histogramm of condition key over all subjects, save fig under output_path, and return fig
            fig, ax = plt.subplots()
            bins = int(self.args['constraints'][key][1] - self.args['constraints'][key][0] + 0.5)
            bins = np.arange(self.args['constraints'][key][0], self.args['constraints'][key][1] + 2, 1)-0.5
            # add borders to bins
            ax.hist([torch.round(sub['conditions'][key]).to(torch.int).numpy() for sub in self.subjects], bins=bins, edgecolor='black')
            ax.set_title(f'Histogramm of {key}')
            fig_name = f'{self.split}_histogramm_{key}.png'
            fig.savefig(os.path.join(self.args['output_path'], fig_name))
            plt.close(fig)
        return fig

    def plot_volume_distribution(self, key, label=0):
        if self.args['constraints'][key]:
            label=4
            res = 3
            scan_age_bins = np.arange(self.args['constraints']['scan_age'][0], self.args['constraints']['scan_age'][1] + 2, 1)
            birth_age_bins = np.arange(self.args['constraints']['birth_age'][0], self.args['constraints']['birth_age'][1] + 2, res)

            # add each subject in self.subjects to the corresponding bin
            scan_age_volumes = {key: [] for key in scan_age_bins}
            birth_age_volumes = {key: [] for key in birth_age_bins}
            for i, ba in enumerate(birth_age_bins):
                birth_age_volumes[ba] = {key: [] for key in scan_age_bins}
                for sub in self.subjects:
                    bas = torch.round(sub['conditions']['birth_age']).to(torch.int).numpy()
                    if bas <= ba and bas:
                        birth_age_volumes[ba][torch.round(sub['conditions']['scan_age']).to(torch.int).numpy()].append((sub['values'][:, -1]!=label).sum())
                birth_age_volumes[ba] = {key: np.mean(birth_age_volumes[ba][key]) if birth_age_volumes[ba][key] else np.nan for key in scan_age_bins}

            # plot for each birth_age the volume of label (y-axis) over scan_age (x-axis)
            fig, ax = plt.subplots()
            colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime', 'teal', 'indigo', 'maroon', 'navy']
            for i, ba in enumerate(birth_age_bins):
                ax.plot(scan_age_bins, [birth_age_volumes[ba][key] for key in scan_age_bins], label=f'birth_age={ba}', color=colors[i])
            ax.set_title(f'Volume distribution of label {label}')
            ax.set_xlabel('scan_age')
            ax.set_ylabel('volume')
            ax.legend()
            fig_name = f'{self.split}_brain_volume_distribution.png'
            fig.savefig(os.path.join(self.args['output_path'], fig_name))

