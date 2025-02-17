import random
import nibabel as nib
import numpy as np
import torch
import glob
import pandas as pd
import os
import scipy.ndimage as ndi
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy


def get_paths(args, dataset, split):
    data_path = args[dataset]['data_path']
    subjects_split = args[dataset]['subject_ids'][split]
    if 'dhcp_neo' in dataset and args['sample_as_mm']: # required for fair validation between mono and multi-modal to sample only subjects with both modalities even for mono test
        args[dataset]['file_ids'] = [ '*_desc-restore_T2w.nii.gz', '*_desc-restore_T1w.nii.gz', '*_desc-drawem9_dseg.nii.gz' ]
    all_paths = [
        sorted([p for p in glob.glob(os.path.join(data_path, id)) if any(sub in p for sub in subjects_split)])
        for id in args[dataset]['file_ids']]
    # print("All paths: ", all_paths)

    if 'dhcp' in dataset:
        common_basenames = set.intersection(*[
            {"_".join(os.path.basename(p).split("_")[:2]) for p in paths}
            for paths in all_paths])
        common_paths = [
            [p for p in paths if "_".join(os.path.basename(p).split("_")[:2]) in common_basenames]
            for paths in all_paths]
    elif dataset == 'marsfet':
        common_sr = {path.split('/latest_nesvor/')[1].split('haste')[0] for path in all_paths[0]}
        common_seg = {path.split('/haste/')[1].split('reo-SVR')[0] for path in all_paths[1]}
        common_basenames = set.intersection(common_sr, common_seg)
        common_paths = [
            [p for p in paths if any([common in p for common in common_basenames])]
            for paths in all_paths]
    elif dataset == 'marsfet_nnUNet':
        common_sr = {path.split('/default_reconst/')[1].split('_acq-haste')[0] for path in all_paths[0]}
        common_seg = {path.split('/combined/')[1].split('.nii.gz')[0] for path in all_paths[1]}
        common_basenames = set.intersection(common_sr, common_seg)
        common_paths = [
            [p for p in paths if any([common in p for common in common_basenames])]
            for paths in all_paths]
    else:
        common_paths = all_paths

    if 'dhcp_neo' in dataset and len(args['modalities']['names']) == 2 and args['sample_as_mm']: # required for fair validation between mono and multi-modal to sample only subjects with both modalities even for mono test
        args[dataset]['file_ids'] = ['*_desc-restore_T2w.nii.gz', '*_desc-drawem9_dseg.nii.gz']
        common_paths = [common_paths[0], common_paths[2]]

    common_paths = [list(p) for p in zip(*common_paths)]
    print(f"Found {len(common_paths)} subjects for {dataset} in {split} split.")
    return common_paths


def get_subject_paths(args, dataset, split, verbose=False):
    # if test time, make sure to load the test set instead of the validation set
    split_ = 'test' if args['test_time'] and split=='val' else split
    n_subjects = args[dataset]['n_subjects'][split]
    paths_modalities = get_paths(args, dataset, split_)
    tsv_file = pd.read_csv(args[dataset]['tsv_file'], sep='\t')
    subjects_selected = []
    for i, path_mods in enumerate(paths_modalities):
        cons_met, conditions = check_constraints(args, path_mods[0], path_mods[-1], tsv_file, dataset=dataset)
        if cons_met:
            if verbose: print("Constraints met for ", path_mods[0])
            subjects_selected.append({"modality_paths": path_mods, "conditions": conditions})
        else:
            if verbose: print("Constraints not met for ", path_mods[0])
    if n_subjects > len(subjects_selected):
        if verbose:
            print("Warning: n_subjects is larger than number of available subjects. "
                  "Setting n_subjects to {}".format(len(subjects_selected)))
        args[dataset]['n_subjects'][split] = len(subjects_selected)
    return subjects_selected


def check_constraints(args, img_path, seg_path, tsv_file, dataset):
    constraints = args['constraints']
    constraints_sub = {}
    if constraints['scan_age']:
        constraints_sub["scan_age"] = get_attribute(img_path, tsv_file, dataset, attribute_name='scan_age')
        if not constraints_met(args['constraints'], constraints_sub):
            return None, False
    if constraints['rad_score']:
        constraints_sub["rad_score"] = get_attribute(img_path, tsv_file, dataset, attribute_name='rad_score')
        if not constraints_met(args['constraints'], constraints_sub):
            return None, False
    if constraints['birth_age'] and 'dhcp_neo' in dataset:
        constraints_sub["birth_age"] = get_attribute(img_path, tsv_file, dataset, attribute_name='birth_age')
        if not constraints_met(args['constraints'], constraints_sub):
            return None, False
    if constraints['pnatal_age'] and 'birth_age' in constraints_sub:
        constraints_sub["pnatal_age"] = constraints_sub["scan_age"] - constraints_sub["birth_age"]
        if not constraints_met(args['constraints'], constraints_sub):
            return None, False
    if constraints['sex']:
        constraints_sub["sex"] = get_attribute(img_path, tsv_file, dataset, attribute_name='sex')
        if not constraints_met(args['constraints'], constraints_sub):
            return None, False
    if constraints['brain_vol']:
        constraints_sub["brain_vol"] = get_brain_volume(nib.load(seg_path), dataset_name=dataset)
        if not constraints_met(args['constraints'], constraints_sub):
            return None, False
    if constraints['label_vol']:
        constraints_sub["label_vol"] = get_label_volume(nib.load(seg_path), labels=constraints['label_vol'])
        if not constraints_met(args['constraints'], constraints_sub):
            return None, False
    if constraints['pathology']:
        constraints_sub['pathology'] = get_attribute(img_path, tsv_file, dataset, attribute_name='pathology')
        if not constraints_met(args['constraints'], constraints_sub):
            return None, False
    return True, constraints_sub


def get_conditions(args, subject, dset):
    tsv_file = pd.read_csv(args[dset]['tsv_file'], sep='\t')
    conditions = args['conditions']
    if conditions['scan_age'] and 'scan_age'not in subject["conditions"]:
        scan_age = get_attribute(subject['modality_paths'][0], tsv_file, dset, attribute_name='scan_age')
        subject["conditions"]["scan_age"] = scan_age
    if conditions['rad_score'] and 'rad_score' not in subject["conditions"]:
        rad_score = get_attribute(subject['modality_paths'][0], tsv_file, dset, attribute_name='radiology_score')
        subject["conditions"]["rad_score"] = rad_score
    if conditions['state'] and 'state' not in subject["conditions"]:
        state = get_attribute(subject['modality_paths'][0], tsv_file, dset, attribute_name='state')
        subject["conditions"]["state"] = state
    if conditions['birth_age'] and 'birth_age' not in subject["conditions"]:
        birth_age = get_attribute(subject['modality_paths'][0], tsv_file, dset, attribute_name='birth_age')
        subject["conditions"]["birth_age"] = birth_age
    if conditions['pnatal_age'] and 'pnatal_age' not in subject["conditions"]:
        pnatal_age = subject["conditions"]["scan_age"] - subject["conditions"]["birth_age"]
        subject["conditions"]["pnatal_age"] = pnatal_age
    if conditions['sex'] and 'sex' not in subject["conditions"]:
        sex = get_attribute(subject['modality_paths'][0], tsv_file, dset, attribute_name='sex')
        subject["conditions"]["sex"] = sex
    if conditions['brain_vol'] and 'brain_vol' not in subject["conditions"]:
        brain_vol = get_brain_volume(nib.load(subject['modality_paths'][-1]), dataset_name=dset)
        subject["conditions"]["brain_vol"] = brain_vol
    if conditions['label_vol'] and 'label_vol' not in subject["conditions"]:
        label_vol = get_label_volume(nib.load(subject['modality_paths'][-1]), labels=conditions['label_vol'],
                                     dataset_name=dset)
        subject["conditions"]["label_vol"] = label_vol
    if conditions['pathology']:
        if 'marsfet' in dset:
            pathology = get_attribute(subject['modality_paths'][0], tsv_file, dset, attribute_name='pathology')
            pathology = -1.0 if pathology == 'IRM_normale' else 1.0
            subject["conditions"]["pathology"] = pathology
        else:
            subject["conditions"]["pathology"] = -1.0
    return subject


def constraints_met(constraints, constraints_sub, verbose=False):  # constraints is a subset of conditions
    keys = constraints_sub.keys()
    for key in keys:
        try:
            if constraints[key] is not None:
                if key == 'pathology': # check if we want to sample this pathology
                    if constraints_sub[key] not in constraints[key]:
                        return False
                elif constraints_sub[key] < constraints[key][0] or constraints_sub[key] >= constraints[key][1]:
                    return False
        except AttributeError:
            if verbose: print("No constraint for key ", key)
    return True


def assemble_bin(args, dataset, subjects, idcs_subs, constraints, parent_bin=None):
    constraint = constraints.pop()
    parent_bin['child_bins'] = get_bins(args['constraints'], constraint=constraint)
    for _bin in parent_bin['child_bins']:
        idcs_suitable_subs = []
        for i in idcs_subs:
            if _bin['min'] <= subjects[i]["conditions"][constraint] < _bin['max']:
                idcs_suitable_subs.append(i)
        if len(constraints) > 0:  # nodes
            _bin = assemble_bin(args, dataset, subjects, idcs_suitable_subs, copy.copy(constraints), parent_bin=_bin)
            parent_bin['leafs'].extend(_bin['leafs'])
        else:  # leafs
            _bin['idcs'] = idcs_suitable_subs
            _bin['count'] = len(idcs_suitable_subs)
            parent_bin['leafs'].append(_bin)
        parent_bin['count'] += _bin['count']
        parent_bin['idcs'] += _bin['idcs']
    return parent_bin


def assemble_subjects_v2(args, dataset, subjects, constraints, n_subs, start_idx, split):
    # Convert list of dictionaries to a DataFrame for easy handling
    dict_converted = []
    for s in subjects:
        dict_converted.append({'modality_paths': s['modality_paths'],
                               'scan_age': torch.round(s['conditions']['scan_age']).to(torch.int).item(),
                               'conditions': s['conditions']})
        if "birth_age" in s['conditions']:
            dict_converted[-1]['birth_age'] = torch.round(s['conditions']['birth_age']).to(torch.int).item()
        if "pnatal_age" in s['conditions']:
            dict_converted[-1]['pnatal_age'] = torch.round(s['conditions']['pnatal_age']).to(torch.int).item()

    df = pd.DataFrame(dict_converted)
    sec_key = 'birth_age' if 'birth_age' in constraints else ''
    sampled = sample_uniformly(df, args[dataset]['n_subjects'][split], prim_key='scan_age', sec_key=sec_key,
                               samp_strat=args['sampling_strategy'])
    return sampled


def sample_uniformly(df, k, prim_key='scan_age', sec_key='birth_age', samp_strat='uniform'):
    rnd = np.floor if samp_strat == 'uniform' else np.ceil
    # Step 1: Sample uniformly from scan_age groups
    scan_age_groups = df.groupby(prim_key)
    # List of unique scan_age groups
    unique_scan_ages = list(scan_age_groups.groups.keys())
    # If k is larger than the number of unique scan ages, adjust it
    num_unique_scan_ages = len(unique_scan_ages)
    num_samples_per_scan_age = max(1, rnd(k/num_unique_scan_ages).astype(int))
    # assign priorities to birth age bins to later sample the ones with highest priority first, and decrease the priority after sampling
    ids_sampled = []
    sampled_dicts = []
    if sec_key:
        birth_ages = list(df.groupby(sec_key).groups.keys())
        priorities = [1000] * len(birth_ages)
        b_a_prios = list(zip(birth_ages, priorities))
        # Step 2: Within each scan_age group, sample uniformly by birth_age
        for scan_age in unique_scan_ages:
            group = scan_age_groups.get_group(scan_age)
            # Now sample within each scan_age group by birth_age
            # sort b_a_prios by priority
            b_a_prios = sorted(b_a_prios, key=lambda x: x[1], reverse=True)
            birth_age_groups = group.groupby(sec_key)
            # Randomly sample dictionaries within the birth_age groups
            ns_group = 0
            idx_prios = 0
            idx_break = 0
            idx_all_sampled = 0

            while ns_group < num_samples_per_scan_age:
                birth_age, prio = b_a_prios[idx_prios%len(b_a_prios)]
                if birth_age not in birth_age_groups.groups:
                    idx_prios += 1
                    idx_break += 1
                    if idx_break >= len(b_a_prios): # not enough samples in this scan_age group
                        break
                    continue
                idx_break = 0

                group_by_birth = birth_age_groups.get_group(birth_age)
                samples = group_by_birth.sample(n=len(group_by_birth), random_state=1)
                for i, sample in enumerate(samples.iterrows()):
                    if sample[0] not in ids_sampled:
                        sampled_dicts.append(sample[1].to_dict())
                        ids_sampled.append(sample[0])
                        ns_group += 1
                        b_a_prios[idx_prios%len(b_a_prios)] = (birth_age, prio-1)
                        break
                    elif i == len(samples)-1:
                        idx_all_sampled += 1
                idx_prios += 1
                if idx_all_sampled >= len(birth_age_groups)*num_samples_per_scan_age: # all birth ages sampled
                    break
    else:
        while len(sampled_dicts) < min((len(unique_scan_ages)*num_samples_per_scan_age), k):
            for scan_age in unique_scan_ages:
                group = scan_age_groups.get_group(scan_age)
                samples = group.sample(n=len(group), random_state=1)
                for sample in samples.iterrows():
                    if sample[0] not in ids_sampled:
                        sampled_dicts.append(sample[1].to_dict())
                        ids_sampled.append(sample[0])
                        break
                    if len(sampled_dicts) >= k:
                        break
    return sampled_dicts


def assemble_subjects(args, subjects, dataset, start_idx, split):
    # TODO: needs restructuring
    subjects_loaded = []
    constraints = ['scan_age']
    additional_cons = [key for key in args['conditions'].keys() if args['conditions'][key]
                       and (key != 'rad_score' and key != 'state')]
    constraints.extend(additional_cons)
    n_subs = args[dataset]['n_subjects'][split]
    if True:
        print(f"Sampling {n_subs} subjects from {len(subjects)} available subjects.")
        subs_sampled = assemble_subjects_v2(args, dataset, subjects, constraints, n_subs, start_idx, split)
        print(f"Sampled {len(subs_sampled)} of {args[dataset]['n_subjects'][split]} subjects.")
        args[dataset]['n_subjects'][split] = len(subs_sampled)
        for idx, sub in enumerate(subs_sampled):
            subjects_loaded.append(load_subject(args, sub, dataset, split, start_idx+idx))
    else:
        subs_sampled = []
        # constraints = ['scan_age', 'birth_age']
        if args['sampling_strategy'] == 'uniform_fillup':
            parent_bin = assemble_bin(args, dataset, subjects, np.arange(len(subjects)), constraints,
                                      parent_bin={'count': 0, 'idcs': [], 'leafs': []})
        elif args['sampling_strategy'] == 'uniform':
            pass
        elif args['sampling_strategy'] == 'random':
            random.shuffle(subjects)
            for i, subject in enumerate(subjects[:args['n_subjects']['split']]):
                subs_sampled.append(load_subject(args, subject, dataset, split, i))
        counter = 0
        num_leafs = len(parent_bin['leafs'])
        while len(subs_sampled) < n_subs:
            current_leaf = parent_bin['leafs'][counter % num_leafs]
            if current_leaf['count'] > 0:
                subs_sampled.append(current_leaf['idcs'].pop())
                current_leaf['count'] -= 1
            counter += 1
        args[dataset]['n_subjects'][split] = len(subs_sampled)
        for idx, i in enumerate(subs_sampled):
            subjects_loaded.append(load_subject(args, subjects[i], dataset, split, start_idx+idx))
        # print max and min birth age
        # bages = [sub['conditions']['birth_age'] for sub in subjects_assembled]
        # print("Max birth age: ", max(bages))
        # print("Min birth age: ", min(bages))

    return subjects_loaded


def get_bins(constraints, constraint='scan_age'):
    bins = []
    for i in range(constraints[constraint][0], constraints[constraint][1]):
        bins.append({'constraint': constraint, 'min': i, 'max': i + 1, 'count': 0, 'idcs': [], 'leafs': []})
    return bins


def load_subject(args, subject, dataset, split, idx, apply_mask=True):
    subject = get_conditions(args, subject, dset=dataset)
    modalities, affines = [], []
    mask = None
    for j, mod_path in enumerate(reversed(subject["modality_paths"])):
        mod_nii = nib.load(mod_path)
        mod_np = mod_nii.get_fdata().squeeze()
        if apply_mask:
            if j == 0:  # first of the reversed, i.e. last modality is segmentation
                mask = mod_nii.get_fdata().squeeze() > 0
            else: # not segmentation
                if 'marsfet' in dataset:
                    # clip upper 99th percentile
                    mod_np = np.clip(mod_np, 0, np.percentile(mod_np, 99.9))
            modalities.append(torch.from_numpy(mod_np.squeeze() * mask).to(args['device_dataset'], torch.float))
        else:
            modalities.append(torch.from_numpy(mod_np.squeeze()).to(args['device_dataset'], torch.float))
        affines.append(torch.from_numpy(mod_nii.affine).to(args['device_dataset'], torch.float))
    subject['modalities'] = modalities[::-1]  # reverse the reversed list
    subject['affines'] = affines[::-1]
    subject["modalities"][-1] = add_background_halo(args, subject["modalities"][-1], halo_width=1.5)
    subject["coords"], subject["values"] = coords_and_values(subject)
    subject["sub_id"], subject["ses_id"] = get_basename(subject['modality_paths'][0], dataset)
    subject[f"idx_{split}"] = idx
    subject['dataset'] = dataset
    return subject


def coords_and_values(subject):
    c_nz = torch.nonzero(subject["modalities"][-1] > 0)
    values = torch.cat([subject["modalities"][i][c_nz[:, 0], c_nz[:, 1], c_nz[:, 2]].flatten()[..., None]
                        for i in range(len(subject["modalities"]))], dim=-1)
    coords_center = (torch.tensor(subject["modalities"][0].shape) - 1.0) / 2
    coords_non_zero = c_nz - coords_center
    values_normed = values.clone()
    for i in range(len(subject["modalities"])-1):
        values_normed[:, i] = ((values_normed[:, i] - values_normed[:, i].min())
                               / (values_normed[:, i].max() - values_normed[:, i].min()))
    return coords_non_zero, values_normed


# add background halo to segmentation to allow masking of the brain in the postprsocessing step
def add_background_halo(args, seg, halo_width=1.5):
    bg_label = args['label_names'].index('BG')
    mask_bg = (seg>0).numpy().astype(np.float32)
    mask_bg = (ndi.gaussian_filter(mask_bg, sigma=halo_width) > 0.001).astype(np.uint8) * bg_label
    mask_bg[seg > 0] = seg[seg > 0]
    return torch.from_numpy(mask_bg).to(torch.float)

def determine_bounding_box(subjects):
    device = subjects[0]["coords"].device
    bbox = [[9999, 9999, 9999], [-9999, -9999, -9999]]
    bbox_world = [[9999, 9999, 9999], [-9999, -9999, -9999]]
    for i, subject in enumerate(subjects):
        coords = subject["coords"]
        coords_world = (subject["affines"][0][:3, :3] @ coords.to(torch.float).T).T
        for j in range(3):
            bbox[0][j] = min(bbox[0][j], coords[:, j].min())
            bbox[1][j] = max(bbox[1][j], coords[:, j].max())
            bbox_world[0][j] = min(bbox_world[0][j], coords_world[:, j].min())
            bbox_world[1][j] = max(bbox_world[1][j], coords_world[:, j].max())
    bbox = torch.tensor(bbox).to(device)
    bbox_world = torch.tensor(bbox_world).to(device)
    bbox = {'min': bbox[0], 'max': bbox[1], 'divisor': 1 / (bbox[1] - bbox[0])}
    bbox_world = {'min': bbox_world[0], 'max': bbox_world[1], 'divisor': 1 / (bbox_world[1] - bbox_world[0])}
    return bbox, bbox_world


def load_subjects(args, dataset, start_idx, split):
    subjects = get_subject_paths(args, dataset, split)
    subjects = assemble_subjects(args, subjects, dataset, start_idx, split)
    return subjects


def get_brain_volume(seg_nii, dataset_name):
    spacing = seg_nii.header.get_zooms()
    if 'dhcp' in dataset_name:
        brain_voxels = np.sum((seg_nii.get_fdata() > 0) & (seg_nii.get_fdata() != 4))
    elif dataset_name == 'feta' or 'marsfet' in dataset_name:
        brain_voxels = np.sum(seg_nii.get_fdata() > 0)
    else:
        raise ValueError("Dataset not supported.")
    brain_volume = brain_voxels * np.prod(spacing)
    return brain_volume


def get_label_volume(seg_nii, labels, dataset_name):
    brain_vol = get_brain_volume(seg_nii, dataset_name)
    label_volumes = []
    for label in labels:
        seg = seg_nii.get_fdata()
        spacing = seg_nii.header.get_zooms()
        # get label volume in mm^3
        label_volumes.append((np.sum(seg == label) * np.prod(spacing) / brain_vol))
    return torch.tensor(label_volumes)


def get_attribute(p_img, tsv_file, dataset_name, attribute_name):
    sub_id, ses_id = get_basename(p_img, dataset_name)
    if dataset_name == 'dhcp_fetal' or "dhcp_neo" in dataset_name:
        if attribute_name == "sex":
            attribute = tsv_file.loc[(tsv_file["subject_id"] == sub_id)][attribute_name].values[0]
            attribute = (int(attribute == 'M') - 0.5) * 2
        elif attribute_name == "birth_age":
            attribute = tsv_file.loc[(tsv_file["subject_id"] == sub_id)][attribute_name].values[0]
        elif attribute_name == "state":
            attribute = -1 if dataset_name == 'dhcp_fetal' else 1
        elif attribute_name == "pathology":
            attribute = "IRM_normale"
        else:
            attribute = tsv_file.loc[(tsv_file["subject_id"] == sub_id) &
                                     (tsv_file["session_id"] == int(ses_id[4:]))][attribute_name].values[0]
    elif dataset_name == 'feta':
        if attribute_name == 'scan_age' or attribute_name == 'rad_score':
            attribute = tsv_file.loc[(tsv_file["participant_id"] == sub_id)][attribute_name].values[0]
        else:
            attribute = None
    elif 'marsfet' in dataset_name:
        if attribute_name == 'scan_age' or attribute_name == 'pathology' or attribute_name == 'rad_score':
            attribute = tsv_file.loc[(tsv_file["marsfet_subject_id"] == sub_id)
                                     & (tsv_file["marsfet_session_id"] == ses_id)][attribute_name].values[0]
        if attribute_name == "sex":
            attribute = tsv_file.loc[(tsv_file["marsfet_subject_id"] == sub_id)
                                     & (tsv_file["marsfet_session_id"] == ses_id)][attribute_name].values[0]
            if attribute != 'M' and attribute != 'F':
                attribute = -2
            attribute = (int(attribute == 'M') - 0.5) * 2
        if attribute_name == "pathology":
            attribute = tsv_file.loc[(tsv_file["marsfet_subject_id"] == sub_id)
                                     & (tsv_file["marsfet_session_id"] == ses_id)][attribute_name].values[0]
    elif dataset_name == 'hcp':
        pass
    else:
        raise ValueError("Dataset not supported")

    if attribute is not None and type(attribute) is not str and type(attribute) is not type(torch.tensor):
        attribute = torch.tensor(attribute).to(torch.float)
    return attribute


def get_basename(path, dataset):
    sub_id, ses_id = None, None
    if dataset == 'dhcp_fetal' or "dhcp_neo" in dataset:
        basename = os.path.basename(path)[:-24]
        sub_id = basename.split('_')[0]
        ses_id = basename.split('_')[1]
    elif dataset == 'feta':
        sub_id = (os.path.basename(path)[:7], '')
        ses_id = None
    elif 'marsfet' in dataset:
        basename = os.path.basename(path)
        sub_id = basename.split('_')[0]
        ses_id = basename.split('_')[1]
        print(f"Found sub_id: {sub_id}, ses_id: {ses_id}")
    elif dataset == 'hcp':
        basename = path.split('/')[-3]
    else:
        raise ValueError("Dataset not supported")
    return sub_id, ses_id
