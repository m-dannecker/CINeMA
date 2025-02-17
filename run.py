import os
import sys
from datetime import datetime
import torch
import numpy as np
import wandb
import yaml
import argparse
import shutil
import json
from utils import dict_to_simplenamespace
from atlas import Atlas
os.environ["WANDB__SERVICE_WAIT"] = "500"

def parse_arguments():
    parser = argparse.ArgumentParser(description="IA Script")
    parser.add_argument("--config", type=str, default="./configs/config_atlas.yaml",
                        help="Path to YAML config file")
    # parser.add_argument("--logging", action="store_true", default=False, help="Enable logging with wandb")
    parser.add_argument("--sweep", action="store_true", default=False, help="Enable sweep")
    parser.add_argument("--logging", action="store_true", default=False, help="Log to wandb")
    parser.add_argument("--val_init_by_mean", action="store_true", default=False, help="val latent mean init")
    parser.add_argument("--test_time", action="store_true", default=False, help="val latent mean init")
    parser.add_argument("--dataset_name", type=str, help="dataset_name")
    parser.add_argument("--load_model", type=str, help="name of model to load")
    parser.add_argument("--sample_as_mm", action="store_true", default=False, help="Sample only subs with mm data available")
    parser.add_argument("--conditions__birth_age", type=int, help="Whether to use birth age as condition")
    parser.add_argument("--conditions__sex", type=int, help="Whether to use birth age as condition")
    parser.add_argument("--conditions__pnatal_age", type=int, help="Whether to use birth age as condition")
    # parser.add_argument("--compile", action="store_true", help="Compile the model")
    parser.add_argument("--mask_reconstruction", action="store_true", default=True, help="Mask reconstruction")
    parser.add_argument("--latent_dim", type=int, nargs='+', help="Latent dimension [c,x,y,z]")
    parser.add_argument("--lat_l2_weight", type=float, help="Latent regularization weight")
    parser.add_argument("--wd_latent__train", type=float, help="Train latent reg weight")
    parser.add_argument("--wd_latent__val", type=float, help="Val latent reg weight")
    parser.add_argument("--trafo_dim__train", type=int, help="Trafo dim train")
    parser.add_argument("--trafo_dim__val", type=int, help="Trafo dim val")
    parser.add_argument("--num_hidden_layers", type=int, help="Number of hidden layers")
    parser.add_argument("--hidden_size", type=int, help="Hidden size")
    parser.add_argument("--head_hidden_size", type=int, help="Hidden size")
    parser.add_argument("--head_num_layers", type=int, help="head_num_layers (min=1)")
    parser.add_argument("--first_omega", type=float, help="First omega")
    parser.add_argument("--hidden_omega", type=float, help="Hidden omega")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size__train", type=int, help="Batch size for training")
    parser.add_argument("--n_samples__train", type=int, help="Sample size for training")
    parser.add_argument("--batch_size__val", type=int, help="Batch size for validation")
    parser.add_argument("--n_samples__val", type=int, help="Sample size for validation")
    parser.add_argument("--val_epochs", type=int, help="Validation epochs")
    parser.add_argument("--val_every", type=int, help="Validation every n epochs")
    parser.add_argument("--seg_weight", type=float, help="seg_weight")
    parser.add_argument("--lr_latent__train", type=float, help="Learning rate latent train")
    parser.add_argument("--lr_latent__val", type=float, help="Learning rate latent val")
    parser.add_argument("--lr_trafo__train", type=float, help="Learning rate trafo train")
    parser.add_argument("--lr_trafo__val", type=float, help="Learning rate trafo val")
    parser.add_argument("--lr_cond__val", type=float, help="Learning rate latent condition during ttime")
    parser.add_argument("--lr", type=float, help="Learning rate SR")
    parser.add_argument("--cond_scale", type=float, help="cond_scale")
    parser.add_argument("--modulated_layers", type=int, nargs='+', help="Modulated layers")

    args = parser.parse_args()
    cmd_args = {k: v for k, v in vars(args).items() if v is not None}
    return cmd_args

def override_args(config_args, cmd_args):
    for key, value in cmd_args.items():
        key1, key2 = key.split("__") if "__" in key else (key, None)
        if key2 is None:
            if value is not None:
                config_args[key] = value
        else:
            if value is not None:
                config_args[key1][key2] = value
    return config_args

def main():
    cmd_args = parse_arguments()
    with open("./configs/config_atlas.yaml", 'r') as stream:
        yaml_args = yaml.safe_load(stream)
    if "dataset_name" in cmd_args:
        yaml_args['dataset_name'] = cmd_args['dataset_name']
    with open("./configs/config_data.yaml", 'r') as stream:
        data_args = yaml.safe_load(stream)[yaml_args['dataset_name']]
    for dname in data_args['datasets']:
        with open("./configs/subject_ids.yaml", 'r') as stream:
            data_args[dname]["subject_ids"] = yaml.safe_load(stream)[dname]["subject_ids"]

    args = {**yaml_args, **data_args}
    args = override_args(args, cmd_args)
    args['root_dir'] = os.path.join(os.getcwd().split('/ImplicitAtlas/')[0], 'ImplicitAtlas')
    print("Root dir set to: ", args['root_dir'])

    args['logging'] = True
    args['test_time'] = False



    if args['test_time']:
        if args['load_model']:
            args[args['datasets'][0]]['n_subjects']['train'] = 1
        else:
            args['test_time'] = False
    metrics = {}
    for seed in args['seeds']:
        args_run = args.copy()
        args_run['seed'] = seed
        # Seeding
        torch.manual_seed(args_run['seed'])
        torch.cuda.manual_seed(args_run['seed'])
        np.random.seed(args_run['seed'])
        # torch.backends.cudnn.deterministic = True

        # get node id from system
        job_id = os.getenv("SLURM_JOB_ID", "loc")[-3:]
        now = datetime.now()
        ttime = "ttime_" if args_run['test_time'] else ""
        # generate three digit random number
        rn = np.random.randint(0, 100)
        now = now.replace(microsecond=rn)
        # print first 2 decimals of microseconds
        dt_string = now.strftime(f"{ttime}{''.join(args_run['datasets'])}_%d%m%Y_%H%M%S_{job_id}_{seed}")
        args_run['setup_name'] = dt_string

        if args_run['logging']:
            wandb.init(config=args_run, project=args_run['project_name'], entity=args_run['wandb_entity'], name=dt_string)
        if args_run['sweep']:
            print("Sweep config: ", wandb.config)
            args_run = dict_to_simplenamespace(wandb.config.as_dict())

        # logging
        args_run['log_path'] = os.path.join(args_run['root_dir'], args_run['log_dir'], dt_string)
        args_run['output_path'] = os.path.join(args_run['root_dir'], args_run['output_dir'], dt_string)
        os.makedirs(args_run['output_path'], exist_ok=True)
        # copy all config files to output path for reproducibility
        shutil.copyfile("configs/config_atlas.yaml", os.path.join(args_run['output_path'], "config_atlas.yaml"))
        shutil.copyfile("configs/config_data.yaml", os.path.join(args_run['output_path'], "config_data.yaml"))
        shutil.copyfile("configs/subject_ids.yaml", os.path.join(args_run['output_path'], "subject_ids.yaml"))

        print("Log path: ", args_run['log_path'])
        print("Output path: ", args_run['output_path'])
        if not args_run['save_model']:
            print("-------------- WARNING: Model checkpoints will not be saved!!!! --------------")



        atlas = Atlas(args_run)
        if args_run['test_time']:
            metrics[seed] = atlas.inference()
        else:
            metrics[seed] = atlas.run()
        # close wandb
        wandb.finish()

    if args['logging']:
        if len(args['seeds']) == 1:
            # log each epoch
            epochs = list(metrics[args['seeds'][0]].keys())
            metrics_epochs = {}
            for epoch in epochs:
                metrics_epoch = {}
                for key in metrics[args['seeds'][0]][epoch].keys():
                    if key == 'dice':
                        metrics_epoch['dsc_mean'] = np.mean([np.mean(metrics[seed][epoch][key][0].numpy()) for seed in args['seeds']]).astype(np.float64)
                        metrics_epoch['dsc_mean_std'] = np.mean([np.mean(metrics[seed][epoch][key][1].numpy()) for seed in args['seeds']]).astype(np.float64)
                        metrics_epoch['dsc'] = np.array(metrics[seed][epoch][key][0].numpy()).tolist()
                        metrics_epoch['dsc_std'] = np.array(metrics[seed][epoch][key][1].numpy()).tolist()
                    else:
                        metrics_epoch[key] = np.mean([metrics[seed][epoch][key] for seed in args['seeds']]).astype(np.float64)
                metrics_epochs[epoch] = metrics_epoch
            # log to json
            with open(os.path.join(args_run['output_path'], f'metrics_epochs.json'), 'w') as f:
                json.dump(metrics_epochs, f, indent=4)
        else:
            metrics_mean_over_seeds = {}
            metrics_list_over_seeds = {}
            metrics_std_over_seeds = {}
            f_epoch = list(metrics[args['seeds'][0]].keys())[0]
            for key in metrics[args['seeds'][0]][f_epoch].keys():
                if key == 'dice':
                    metrics_mean_over_seeds[key] = np.mean([metrics[seed][f_epoch][key][0].numpy() for seed in args['seeds']], axis=0).tolist()
                    metrics_mean_over_seeds['dsc_mean'] = np.mean([metrics[seed][f_epoch][key][0].numpy() for seed in args['seeds']]).astype(np.float64)
                    metrics_std_over_seeds[key] = np.std([metrics[seed][f_epoch][key][0].numpy() for seed in args['seeds']], axis=0).tolist()
                    metrics_std_over_seeds['dsc_mean'] = np.std([np.mean(metrics[seed][f_epoch][key][0].numpy()) for seed in args['seeds']]).astype(np.float64)
                    metrics_list_over_seeds[key] = np.array([metrics[seed][f_epoch][key][0].numpy() for seed in args['seeds']]).tolist()
                    metrics_list_over_seeds['dsc_mean'] = np.array([np.mean(metrics[seed][f_epoch][key][0].numpy()) for seed in args['seeds']]).tolist()
                else:
                    metrics_mean_over_seeds[key] = np.mean([metrics[seed][f_epoch][key] for seed in args['seeds']]).astype(np.float64)
                    metrics_std_over_seeds[key] = np.std([metrics[seed][f_epoch][key] for seed in args['seeds']]).astype(np.float64)
                    metrics_list_over_seeds[key] = np.array([metrics[seed][f_epoch][key] for seed in args['seeds']]).tolist()
            # log to json
            with open(os.path.join(args_run['output_path'], 'metrics_over_seeds.json'), 'w') as f:
                json.dump({'mean': metrics_mean_over_seeds, 'std': metrics_std_over_seeds, 'list': metrics_list_over_seeds}, f, indent=4)


if __name__ == "__main__":
    main()
