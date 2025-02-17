import math
from types import SimpleNamespace
import ants
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import nibabel as nib
import wandb as wd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import scipy.ndimage as ndi


def dict_to_simplenamespace(d):
    """ Recursively converts dictionary to SimpleNamespace. """
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_simplenamespace(value)
        return SimpleNamespace(**d)
    else:
        return d


class Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_classes = args['out_dim'][-1]-1  # number of classes for segmentation excluding background
        self.sr_dims = sum(args['out_dim'][:-1])
        self.criterion_sr = nn.MSELoss() if args['loss_metric'] == 'mse' else nn.L1Loss()
        self.criterion_seg = nn.CrossEntropyLoss()
        self.criterion_contrastive = CosineSimilarityContrastiveLoss(margin=args['contrastive_margin'])

    def forward(self, output, target, lats, trafos, sr_weight=1.0, seg_weight=1.0, positive_pairs=None):
        # if mm then mods are (T2, T1, Seg)
        sr_dims_start = 0 if not self.args['T1_optim'] else 1   # we optimize on T1 only
        sr_dims = self.sr_dims if not self.args['T2_optim'] else 1  # we optimize on T2 only
        loss = {'seg': 0, 'sr': sr_weight * self.criterion_sr(output[..., sr_dims_start:sr_dims], target[..., sr_dims_start:sr_dims]),
                'lat_reg': self.args['lat_l2_weight'] * torch.mean(lats ** 2) if self.args['lat_l2_weight'] > 0 else torch.tensor(0.0),
                'contrastive': torch.tensor(0.0), 'trafo': torch.tensor(0.0), 'total': 0.0}

        if self.args['segmentation']:
            loss['seg'] = seg_weight * self.criterion_seg(output[..., self.sr_dims:], target[..., -1].to(torch.int64))
        if positive_pairs is not None:
            loss['contrastive'] = self.args['contrastive_weight'] * self.criterion_contrastive(lats, positive_pairs[0],
                                                                                            positive_pairs[1])
        if trafos is not None:
            loss['trafo'] = self.args['trafo_weight'] * torch.mean(trafos ** 2)
        loss['total'] = loss['sr'] + loss['seg'] + loss['lat_reg'] + loss['contrastive'] + loss['trafo']
        return loss


class CosineSimilarityContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(CosineSimilarityContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, lats,  anchor, positive):
        anchor = lats[anchor]
        positive = lats[positive]
        # Calculate the cosine similarity between anchor and positive samples
        cosine_sim = F.cosine_similarity(anchor, positive, dim=-1)

        # Calculate the loss as 1 minus the cosine similarity for positive pairs,
        # and then apply a margin. The loss is zero when the cosine similarity is above the margin.
        if self.margin == 1:
            loss = -torch.log((cosine_sim + 1.0) / 2.0)
        else:
            loss = torch.relu(1 - cosine_sim - self.margin)  # Enforces similarity above margin

        return loss.mean()


def dice_score(predictions, targets, num_classes, bg_label, reduce=True):
    dice_scores = []
    for class_idx in range(1, num_classes):
        if class_idx == bg_label:  # skip background
            continue
        pred_binary = (predictions == class_idx).to(torch.int)
        target_binary = (targets == class_idx).to(torch.int)

        intersection = torch.sum(pred_binary * target_binary)
        total = torch.sum(pred_binary) + torch.sum(target_binary)

        dice_score = (2. * intersection + 1e-6) / (total + 1e-6)  # Adding a small constant to avoid division by zero
        dice_scores.append(dice_score)
    dice = torch.stack(dice_scores).mean().item() if reduce else torch.stack(dice_scores)
    return dice


def compare2ref(args, modalities_rec, modalities_ref, aff_rec=None, aff_ref=None, mask=True, crop2bbox=True, normalize=True,
                reg_type='Rigid', bg_label=4, num_classes=1, reduce=False):
    num_mods = modalities_rec.shape[-1]
    mytx = None
    modalities_reg_np = []
    modalities_ref_np = []
    for i in range(num_mods):
        mod = modalities_rec[..., i]
        mod_ref = modalities_ref[i]
        if isinstance(mod, torch.Tensor):
            mod = mod.detach().cpu().numpy()
        if isinstance(mod_ref, torch.Tensor):
            mod_ref = mod_ref.cpu().numpy()
        if not isinstance(mod, nib.Nifti1Image):
            mod = nib.Nifti1Image(mod, aff_rec)
        if not isinstance(mod_ref, nib.Nifti1Image):
            mod_ref = nib.Nifti1Image(mod_ref, aff_ref)
        if i == 0:
            mod_reg, mytx = reg_imgs(mod, mod_ref, reg_type=reg_type, interpolator='linear')
            nib.save(mod, os.path.join(args['output_path'], 'mod.nii.gz'))
            nib.save(mod_ref, os.path.join(args['output_path'], 'mod_ref.nii.gz'))

        elif i < num_mods-1:
            mod_reg, _ = reg_imgs(mod, mod_ref, mytx, reg_type, interpolator='linear')
        else:
            mod_reg, _ = reg_imgs(mod, mod_ref, mytx, reg_type, interpolator='nearestNeighbor')
        modalities_reg_np.append(mod_reg.get_fdata())
        modalities_ref_np.append(mod_ref.get_fdata())
    mask = modalities_ref_np[-1] > 0 if mask else None
    _, bbox = get_non_zero_bbox(mask) if crop2bbox else (None, None)
    mod_regs, psnrs, ssims, nccs, dices = [], [[] for _ in range(num_mods-1)], [[] for _ in range(num_mods-1)], [[] for _ in range(num_mods-1)], []
    for i, (mod_reg_np, mod_ref_np) in enumerate(zip(modalities_reg_np, modalities_ref_np)):
        if mask is not None:
            mod_reg_np = mod_reg_np * mask
            mod_ref_np = mod_ref_np * mask
        if crop2bbox:
            mod_reg_np_c = mod_reg_np[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], bbox[0][2]:bbox[1][2]]
            mod_ref_np_c = mod_ref_np[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], bbox[0][2]:bbox[1][2]]
        else:
            mod_reg_np_c = mod_reg_np
            mod_ref_np_c = mod_ref_np
        if normalize and i < len(modalities_reg_np)-1:
            mod_reg_np_c = (mod_reg_np_c - mod_reg_np_c.min()) / (mod_reg_np_c.max() - mod_reg_np_c.min())
            mod_ref_np_c = (mod_ref_np_c - mod_ref_np_c.min()) / (mod_ref_np_c.max() - mod_ref_np_c.min())
        if i < len(modalities_reg_np)-1:
            psnrs[i].append(psnr_metric(mod_reg_np_c, mod_ref_np_c, data_range=1.0))
            ssims[i].append(ssim_metric(mod_reg_np_c, mod_ref_np_c, data_range=1.0))
            nccs[i].append(compute_ncc(mod_reg_np_c, mod_ref_np_c))
        else:
            if args['harmonize_labels']: # this is for comparison with Makropoulos et al. 2015 and Serag et al. 2012
                mod_reg_np_c = harmonize_labels(mod_reg_np_c, args['dataset_name'])
                mod_ref_np_c = harmonize_labels(mod_ref_np_c, args['dataset_name'])
                num_classes = 8 if 'dhcp_neo' in args['dataset_name'] else 5
                bg_label = 0

            dices.append(dice_score(torch.from_numpy(mod_reg_np_c).to(torch.int),
                                    torch.from_numpy(mod_ref_np_c).to(torch.int),
                                    num_classes=num_classes, bg_label=bg_label, reduce=reduce))
        mod_regs.append(mod_reg_np_c)
    mod_regs = torch.from_numpy(np.stack(mod_regs, axis=-1))
    new_aff = mod_reg.affine

    # if mm then mods are (T2, T1, Seg)

    return (mod_regs, new_aff, [np.mean(psnrs[i]) for i in range(len(psnrs))],
            [np.mean(ssims[i]) for i in range(len(ssims))], [np.mean(nccs[i]) for i in range(len(nccs))], dices[0])


def reg_imgs(mov, fixed, mytx=None, reg_type='Rigid', interpolator='linear'):
    mov_ants = ants.from_nibabel(mov)
    fixed_ants = ants.from_nibabel(fixed)
    # normalize intensities
    if interpolator == 'linear':
        mov_ants = (mov_ants - mov_ants.min()) / (mov_ants.max() - mov_ants.min())
        fixed_ants = (fixed_ants - fixed_ants.min()) / (fixed_ants.max() - fixed_ants.min())
    if mytx is None and reg_type is not None: # first modality, hence we need to register
        mytx = ants.registration(fixed=fixed_ants, moving=mov_ants, type_of_transform=reg_type)
    if mytx is None: # no registration needed as we only need to resample
        # resample moving image to fixed image space
        mov_warped_ants = ants.resample_image_to_target(mov_ants, fixed_ants, interp_type=interpolator)
    else: # apply the transformation if available
        mov_warped_ants = ants.apply_transforms(fixed=fixed_ants, moving=mov_ants, transformlist=mytx['fwdtransforms'],
                                                interpolator=interpolator)
    mov_warped = ants.to_nibabel(mov_warped_ants)
    return mov_warped, mytx


def get_non_zero_bbox(img, dim=3):
    non_zero_indices = np.array(np.where(img != 0))
    min_indices = np.min(non_zero_indices, axis=1)
    max_indices = np.max(non_zero_indices, axis=1)
    if dim == 2:
        img_cropped = img[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1]]
    else:
        img_cropped = img[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2]]
    return img_cropped, np.array([min_indices, max_indices])


def create_coordinate_grid(shape, normed=True, device='cpu'):
    img = torch.ones(shape.tolist(), dtype=torch.float32, device=device)
    grid = torch.nonzero(img).float() - ((shape.to(device) - 1) / 2)
    grid = (grid / (shape.to(device) - 1) * 2) if normed else grid
    return grid


def scale_affine(affine, new_spacing, bbox):
    old_spacing = torch.sqrt(torch.sum(affine[:3, :3] ** 2, axis=0))
    scale = torch.tensor(new_spacing) / old_spacing
    bbox_size = (bbox['max'] - bbox['min'])
    shape = (torch.ceil_(bbox_size / scale).int())
    affine[:3, :3] = torch.matmul(affine[:3, :3], torch.diag(scale))
    affine[:3, 3] = affine[:3, 3] + (bbox['min'] * old_spacing)
    return affine, shape


def compute_ncc(prediction, reference):
    mean_pred = np.mean(prediction)
    mean_ref = np.mean(reference)
    numerator = np.sum((prediction - mean_pred) * (reference - mean_ref))
    denominator = np.sqrt(np.sum((prediction - mean_pred) ** 2) * np.sum((reference - mean_ref) ** 2))
    ncc = numerator / denominator
    return ncc.astype(np.float64)


def get_surface_from_boundary(boundary, spacing, use_grad=False, iters=3):
    # iterate over all non-zero elements in the boundary and check how many of the six direct neighbors are zero
    # count the number of the zero neighbors
    if use_grad:
        # get largest connected component with value zero
        lcc = ndi.label(boundary == 0)[0]
        lcc_sizes = np.bincount(lcc.ravel())
        lcc = lcc == np.argmax(lcc_sizes)
        bndry = np.ones_like(boundary)
        bndry[lcc] = 0
        bndry = ndi.morphology.binary_closing(bndry, iterations=iters)

        gradient = np.gradient(bndry.astype(int))
        grad_norm = np.linalg.norm(gradient, axis=0)
        non_zero_idcs = np.array(np.where(grad_norm != 0)).T
        # plt.imshow(grad_norm[..., grad_norm.shape[-1]//2], cmap='gray')
        # plt.show()
    else:
        non_zero_idcs = np.array(np.where(boundary != 0)).T

    surface = 0
    for idx in non_zero_idcs:
        num_non_zero_neighbors = np.sum(boundary[idx[0]-1:idx[0]+2, idx[1]-1:idx[1]+2, idx[2]-1:idx[2]+2] != 0)
        num_zero_neighbors = 27 - num_non_zero_neighbors
        surface += num_zero_neighbors
    return surface * np.prod(spacing)


def seg2distance_map(img_np, spacing, c, sdf=False):
    img = img_np == c
    distance = ndi.distance_transform_edt(img, sampling=spacing) * img

    if sdf:
        img_inv = np.logical_not(img)
        dist_outside = ndi.distance_transform_edt(img_inv, sampling=spacing)
        distance -= dist_outside
    return distance


def save_atlas(atlas, affine, args, individually=False):
    # atlas is of shape (conds, x, y, z, num_modalities, t) where conds is the conditional dimension
    num_mods = len(args['modalities']['names'])
    if isinstance(atlas, torch.Tensor):
        atlas = atlas.detach().cpu().numpy()
    for c in range(atlas.shape[0]):
        for i in range(num_mods):
            filename = (f'{args["modalities"]["names"][i]}_atlas_age={args["constraints"]["scan_age"][0]}'
                        f'-{args["constraints"]["scan_age"][1]}_cond={c}.nii.gz')
            if individually:
                for j in range(atlas.shape[-1]):
                    filename_ = filename.replace('.nii.gz', f'_t{j}.nii.gz')
                    save_img(atlas[c, ..., i, j], affine=affine, output_path=args['output_path'], filename=filename_)
            else:
                save_img(atlas[c, ..., i, :], affine=affine, output_path=args['output_path'], filename=filename)
            if i == num_mods-1 and args['save_certainty_maps']:
                for j, l_name in enumerate(args['label_names'][1:], start=2):
                    filename_ = filename.replace('Seg', 'CertaintyMaps/'+l_name)
                    save_img(atlas[c, ..., i+j, :], affine=affine, output_path=args['output_path'], filename=filename_)
    print('Atlas saved to {}'.format(args['output_path']))

    if args['logging']:
        if args['conditions']['birth_age']:
            fig_vols = compute_volume_by_birth_age(args, atlas[:, :, :, :, len(args['modalities']['names'])+1:, :], # +1 don't pass background
                                        spacing=np.array(args['spacing_atlas']),
                                        label_names=args['label_names'][1:], scan_age_min_max=args['constraints']['scan_age'],
                                        birth_age_min_max=args['constraints']['birth_age'])
            wd.log({f"Volume By Birth Age": [wd.Image(fig_vols)]})
            plt.close(fig_vols)

        conds, x, y, z, rows, cols = atlas.shape  # rows num_mods + num_labels (soft labels)
        rows = len(args['modalities']['names'])
        for cond in range(conds):
            fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 5))
            axs = axs.reshape(rows, cols)
            for c in range(atlas.shape[-1]):
                for r in range(rows):
                    cmap = "gray" if r < rows-1 else "viridis"
                    axs[r, c].imshow(np.flip(atlas[cond, :, :, z // 2, r, c].T), cmap=cmap)
                    axs[r, c].axis('off')
            fig.suptitle('Atlas scan_age: {}-{}'.format(args['constraints']['scan_age'][0], args['constraints']['scan_age'][1]))
            plt.tight_layout()
            wd.log({f"Atlas condition={cond}": [wd.Image(fig)]})
            plt.close(fig)


def compute_volume_by_birth_age(args, certainty_maps, spacing, label_names, scan_age_min_max, birth_age_min_max):
    # certainty_maps is of shape (birth_age, x, y, z, n_labels, t)
    # one plot per label
    fig, axs = plt.subplots(math.ceil(len(label_names)/2), 2, figsize=(7, 14))
    axs = axs.flatten()
    for ln, label_name in enumerate(label_names):
        volumes = {}
        ba_arange = np.linspace(birth_age_min_max[0], birth_age_min_max[1], args['cond_steps'])
        for ba, birth_age in enumerate(ba_arange):
            volumes[birth_age] = []
            sa_arange = np.arange(scan_age_min_max[0]+1, scan_age_min_max[1], args['age_step'])
            for sa, scan_age in enumerate(sa_arange):
                certainty_map = certainty_maps[ba, ..., ln, sa]
                volumes[birth_age].append(np.sum(certainty_map) * np.prod(spacing))
            # plot volume over scan_age for each birth_age
            axs[ln].plot(sa_arange, volumes[birth_age], label=f'ba={birth_age}')
        axs[ln].set_title(label_name)
        axs[ln].set_xlabel('scan_age')
        axs[ln].set_ylabel('volume')
        axs[ln].legend(fontsize='small', loc='upper left')
    plt.tight_layout()
    return fig


def compute_volumes(certainty_maps_np, spacings, label_names):
    volumes = {}
    for i, label_name in enumerate(label_names):
        label_volume = np.sum(certainty_maps_np == i) * np.prod(spacings)
        volumes[label_name] = label_volume
    return volumes


def save_img(img, affine, output_path, filename):
    if img.dtype == 'int64' or img.dtype == 'int32':
        img = img.astype(np.int16)
    elif img.dtype == 'float64':
        img = img.astype(np.float32)
    full_path = os.path.join(output_path, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    img_nii = nib.Nifti1Image(img, affine)
    nib.save(img_nii, full_path)


def embed2affine(embed):
    R = euler2rot(embed[..., :3])
    t = embed[..., 3:6]
    if embed.shape[-1] >= 9:  # add 3D scaling
        S = torch.diag_embed(1.0 + embed[..., 6:9])
        R = torch.matmul(R, S)
    if embed.shape[-1] == 12:  # add 3D shear
        S_x = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        S_y = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        S_z = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        S_x[..., 0, 1] = embed[..., 10]  # y
        S_x[..., 0, 2] = embed[..., 11]  # z
        S_y[..., 1, 0] = embed[..., 9]   # x
        S_y[..., 1, 2] = embed[..., 11]  # z
        S_z[..., 2, 0] = embed[..., 9]   # x
        S_z[..., 2, 1] = embed[..., 10]  # y
        R = torch.matmul(R, S_x)
        R = torch.matmul(R, S_y)
        R = torch.matmul(R, S_z)
    return R, t


def euler2rot(theta):
    c1 = torch.cos(theta[..., 0])
    s1 = torch.sin(theta[..., 0])
    c2 = torch.cos(theta[..., 1])
    s2 = torch.sin(theta[..., 1])
    c3 = torch.cos(theta[..., 2])
    s3 = torch.sin(theta[..., 2])
    r11 = c1*c3 - c2*s1*s3
    r12 = -c1*s3 - c2*c3*s1
    r13 = s1*s2
    r21 = c3*s1 + c1*c2*s3
    r22 = c1*c2*c3 - s1*s3
    r23 = -c1*s2
    r31 = s2*s3
    r32 = c3*s2
    r33 = c2
    R = torch.stack([r11, r12, r13, r21, r22, r23, r31, r32, r33], dim=-1)
    R = R.view(R.shape[:-1] + (3, 3))
    return R


def harmonize_labels(subject_seg, dataset):
    if 'dhcp_neo' in dataset:
        subject_seg[subject_seg == 4] = 0
        subject_seg[subject_seg == 8] = 6
        subject_seg[subject_seg == 9] = 7
    elif 'dhcp_fetal' in dataset or 'marsfet' in dataset:
        subject_seg[subject_seg == 4] = 0  # BG to 0
        subject_seg[subject_seg == 5] = 4  # Ventricles to 4
        # 3, 6, 7, 8, 9 to 3
        subject_seg[subject_seg == 3] = 3
        subject_seg[subject_seg == 6] = 3
        subject_seg[subject_seg == 7] = 3
        subject_seg[subject_seg == 8] = 3
        subject_seg[subject_seg == 9] = 3
    else:
        raise NotImplementedError("Atlas Type not implemented.")
    return subject_seg
