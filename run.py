import os
from datetime import datetime
import yaml
import argparse
import wandb as wd
from build_atlas import AtlasBuilder
os.environ["WANDB__SERVICE_WAIT"] = "500"


def initial_setup(cmd_args=None):
    with open('./configs/config_atlas.yaml', 'r') as stream:
        args_atlas = yaml.safe_load(stream)
    with open('./configs/config_data.yaml', 'r') as stream:
        config_data = cmd_args['config_data'] if 'config_data' in cmd_args else args_atlas['config_data']
        args_data = {'dataset': yaml.safe_load(stream)[config_data]}
    args = {**args_data, **args_atlas}
    with open(args['dataset']['subject_ids'], 'r') as stream:
        args['dataset']['subject_ids'] = yaml.safe_load(stream)[args['dataset']['dataset_name']]['subject_ids']
    if cmd_args is not None:
        args = override_args(args, cmd_args)

    job_id = os.getenv("SLURM_JOB_ID", "loc")[-3:]
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dsetup = args['config_data']

    run_name =f"{dsetup}_{time_stamp}_{job_id}"
    args['output_dir'] = f"{args['output_dir']}/{run_name}"
    os.makedirs(args['output_dir'], exist_ok=True)
    print(f"Output directory: {args['output_dir']}")

    # save config files
    with open(os.path.join(args['output_dir'], 'config_data.yaml'), 'w') as f:
        yaml.dump(args_data, f)
    with open(os.path.join(args['output_dir'], 'config_atlas.yaml'), 'w') as f:
        yaml.dump(args_atlas, f)
    print(f"Saved config files to {args['output_dir']}")

    if args['inr_decoder']['out_dim'][0] != len(args['dataset']['modalities'])-1:
        print(f"WARNING: The number of output dimensions ({args['inr_decoder']['out_dim'][0]})" 
              f"might not match the number of modalities ({len(args['dataset']['modalities'])-1}).")
    if args['atlas_gen']['conditions'] is not None: # check if all atlas conditions are set True in the dataset config
        for key in args['atlas_gen']['conditions'].keys():
            if not args['dataset']['conditions'][key]:
                print(f"WARNING: The atlas condition {key} is not set True in the dataset config."
                    f"Turning off the atlas generation for {key}.")
                args['atlas_gen']['conditions'].pop(key)

    if args['logging']: # init weights and biases if logging is True
        wd.init(config=args, project=args['project_name'], 
                                entity=args['wandb_entity'], name=run_name)
    return args


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


def parse_cmd_args():
    parser = argparse.ArgumentParser(description="CINeMA Atlas Builder")
    parser.add_argument("--config_data", type=str, help="Configuration data")
    parser.add_argument("--seed", type=int, help="Seed")
    parser.add_argument("--inr_decoder__out_dim", type=int, nargs='+', help="Number of output dimensions [#modalities, #classes of segmentation]")
    parser.add_argument("--inr_decoder__tf_dim", type=int, help="Degrees of freedom for the transformation")
    parser.add_argument("--inr_decoder__cnn_kernel_size", type=int, help="Kernel size for the CNN for spatial modulation")
    parser.add_argument("--inr_decoder__latent_dim", type=int, nargs='+', help="Latent dimension [c,x,y,z]")
    parser.add_argument("--inr_decoder__hidden_size", type=int, help="Hidden size of the sr network")
    parser.add_argument("--inr_decoder__num_hidden_layers", type=int, help="Number of hidden layers of the sr network")
    parser.add_argument("--inr_decoder__modulated_layers", type=int, nargs='+', help="Modulated layers")
    parser.add_argument("--atlas_gen__cond_scale", type=float, help="Scale of the condition vector")
    args = parser.parse_args()
    cmd_args = {k: v for k, v in vars(args).items() if v is not None}
    return cmd_args


def main():
    cmd_args = parse_cmd_args()
    args = initial_setup(cmd_args)
    print(args['inr_decoder'])
    atlas_builder = AtlasBuilder(args)


if __name__ == "__main__":
    main()
