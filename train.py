import numpy as np
import argparse
import time
import random
import os
from os import path as osp
from termcolor import colored
import pickle
import yaml
from pathlib import Path
from model.net import DGCNet
import sys
sys.path.insert(0, "/home/boat/proxyISP/ProxyOpt/")
sys.path.insert(0, "/home/boat/proxyISP/")
sys.path.insert(0, "/home/boat/proxyISP/fast-openISP/")
sys.path.insert(0, "/home/boat/proxyISP/ProxyOpt/pytorch-msssim/")
print(os.getcwd())
print(sys.path)
from ISP_tools.ProxyISPDataset import ProxyISPDataset, EXPERIMENT_OUTPUT_PATH
from proxy_utils import extract_iteration
from ProxyOpt.model import U_Net
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from data.dataset import HomoAffTpsDataset
from utils.loss import L1LossMasked
from utils.optimize import train_epoch, validate_epoch
from tensorboardX import SummaryWriter

def load_proxy_model_and_dataset(proxydgc_config):
    PROXYOPT_BASE_PATH = Path("/home/boat/proxyISP/ProxyOpt/")
    print("proxydgc_config", proxydgc_config)    
    with open(proxydgc_config["config_path"], "r") as f:
        yaml_dict = yaml.safe_load(f)

    config = yaml_dict["config"]
    openisp_config = yaml_dict["openisp_config"]
    hyp_setting = yaml_dict["hyp_setting"]

    stage2_output_dir = Path("proxyopt_output") / proxydgc_config["experiment_name"]

    loaded_param_layer = None
    proxyopt_checkpoint_object = None

    checkpoint_dir = stage2_output_dir / "proxyopt_checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = list(os.scandir(checkpoint_dir))
        checkpoints = sorted(checkpoints, key = lambda x: int(x.name.split("_")[-1].split(".")[0]))
        checkpoints = checkpoints[::-1]
        print("checkpoints", [p.name for p in checkpoints])
        load_attempt = 0 # in case of corrupt file
        max_attempt = 5
        load_success = False
        if checkpoints.__len__() > 0:
            import pickle
            while load_attempt <= max_attempt and load_attempt < len(checkpoints) and not load_success:
                try:
                    checkpoint_path = checkpoints[load_attempt]
                    print("attempt loading", checkpoint_path.path)
                    with open(checkpoint_path.path, "rb") as f:
                        proxyopt_checkpoint_object = pickle.load(f)
                    loaded_param_layer = proxyopt_checkpoint_object["proxy_hype"]
                    train_proxy_from_it = int(checkpoint_path.name.split("_")[-1].split(".")[0])
                    load_success = True
                except Exception as e:
                    print("error", e)
                    load_attempt += 1
        if load_attempt == max_attempt:
            raise Exception(f"apptemted to load checkpoint exceed {load_attempt} times! which were failed!")

    output_dir = PROXYOPT_BASE_PATH / EXPERIMENT_OUTPUT_PATH / config["experiment_name"]
    checkpoint_dir = output_dir / "checkpoints"
    # checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_files = os.listdir(output_dir / "checkpoints")

    latest_file = max(checkpoint_files, key=lambda f: extract_iteration(f))

    latest_obj = torch.load(output_dir / "checkpoints" / latest_file)

    # config = latest_obj["config"]

    # Model and dataset initialization
    in_channels = 1
    if config["input_type"] == "stacked":
        in_channels = 4  # GRGB

    additional_conf = {
        # "target_image": ["/home/boat/proxyISP/data/s21fe_dataset/20240115_123915.dng"],
        # "target_image": ["/home/boat/proxyISP/data/s21fe_dataset/20240117_182706.dng"],
        "proxyopt_base_path": "/home/boat/proxyISP/ProxyOpt/"
    }

    if "target_images" in proxydgc_config:
        raw_images = [str(p) for p in Path(proxydgc_config["target_images"]).rglob("*.dng")]
        additional_conf["target_image"] = raw_images
        # print(additional_conf["target_image"])
        print("target images found", len(additional_conf["target_image"]))

    dataset = ProxyISPDataset(config, openisp_config, hyp_setting, additional_conf)

    raw, _, sample_hyp = dataset.__getitem__(0)
    param_number = sample_hyp.shape[-1]

    net = U_Net(in_channels, 3, step_flag=3, img_size=config["img_size"], param_number=param_number)
    net.load_state_dict(latest_obj["model_state_dict"])
    net = net.to("cuda")

    # Setup target and starting hyperparameters
    dataset.switch_stage2()
    net.img_size = dataset.target_size
    start_hyp = dataset.get_original_hyp(True, True, add_eps = False)
    net.load_param_layer(start_hyp)

    if loaded_param_layer is not None:
        net.param_layer = torch.tensor(loaded_param_layer).to("cuda")
        net.param_layer.requires_grad = True

    net.set_requires_param_layer_grad(True)

    return net, dataset, proxyopt_checkpoint_object

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='DGC-Net train script')
    # Paths
    parser.add_argument('--proxydgc-config', type=str, help='Path to the proxy-dgc-net config file',)
    parser.add_argument('--image-data-path', type=str, default='',
                        help='path to TokyoTimeMachine dataset and csv files')
    parser.add_argument('--metadata-path', type=str, default='./data/',
                        help='path to the CSV files')
    parser.add_argument('--csv-path-train', type=str, default=None,
                        help='path to the train CSV files (overrides metadata-path)')
    parser.add_argument('--csv-path-test', type=str, default=None,
                        help='path to the test CSV files (overrides metadata-path)')
    parser.add_argument('--model', type=str, default='dgc',
                        help='Model to use', choices=['dgc', 'dgcm'])
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--logs', type=str, default='./logs')
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='momentum constant')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--n_epoch', type=int, default=70,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=8,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='weight decay constant')
    parser.add_argument('--seed', type=int, default=1984,
                        help='Pseudo-RNG seed')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.isdir(args.snapshots):
        os.mkdir(args.snapshots)

    cur_snapshot = time.strftime('%Y_%m_%d_%H_%M')

    if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
        os.mkdir(osp.join(args.snapshots, cur_snapshot))

    with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # mean_vector = np.array([0.485, 0.456, 0.406])
    # std_vector = np.array([0.229, 0.224, 0.225])
    # normTransform = transforms.Normalize(mean_vector, std_vector)
    # dataset_transforms = transforms.Compose([
    #         transforms.ToTensor(),
    #         normTransform
    #     ])
    dataset_transforms = None

    pyramid_param = [15, 30, 60, 120, 240]
    weights_loss_coeffs = [1, 1, 1, 1, 1]
    weights_loss_feat = [1, 1, 1, 1]

    csv_file_train = osp.join(args.metadata_path,
                                            'csv',
                                            'homo_aff_tps_train.csv')
    csv_file_test = csv_file=osp.join(args.metadata_path,
                                            'csv',
                                            'homo_aff_tps_test.csv')
    if args.csv_path_train is not None:
        print("overriding default csv train and tests")
        csv_file_train = args.csv_path_train
        if args.csv_path_test is not None:
            csv_file_test = args.csv_path_test
        else:
            raise Exception("Train csv path is provided but test is not")
    
    proxydgc_config = None
    with open(args.proxydgc_config, "r") as f:
        proxydgc_config = yaml.safe_load(f)
    assert proxydgc_config != None
    proxy, proxy_isp_dataset, proxyopt_checkpoint = load_proxy_model_and_dataset(proxydgc_config)

    proxydgc_log_path = Path("proxydgc_logs") / proxydgc_config["experiment_name"]

    if not os.path.exists(proxydgc_log_path):
        os.makedirs(proxydgc_log_path)
    
    # create tensorbaord instance
    proxydgc_log_writer = SummaryWriter(str(proxydgc_log_path / "logs"))

    train_dataset = \
        HomoAffTpsDataset(image_path=args.image_data_path,
                          csv_file=csv_file_train,
                          transforms=dataset_transforms,
                          pyramid_param=pyramid_param)

    val_dataset = \
        HomoAffTpsDataset(image_path=args.image_data_path,
                          csv_file=csv_file_test,
                          transforms=dataset_transforms,
                          pyramid_param=pyramid_param)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.n_threads)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.n_threads)

    # Model
    if args.model == 'dgc':
        model = DGCNet()
        print(colored('==> ', 'blue') + 'DGC-Net created.')
    elif args.model == 'dgcm':
        model = DGCNet(mask=True)
        print(colored('==> ', 'blue') + 'DGC+M-Net created.')
    else:
        raise ValueError('check the model type [dgc, dgcm]')

    model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer
    optimizer = \
        optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=args.lr,
                   weight_decay=args.weight_decay)
    # Scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[2, 15, 30, 45, 60],
                                         gamma=0.1)
    # Criterions
    criterion_grid = L1LossMasked().to(device)
    criterion_match = None
    if args.model == 'dgcm':
        criterion_match = nn.BCEWithLogitsLoss().to(device)

    train_losses = []
    val_losses = []
    prev_model = None

    train_started = time.time()

    for epoch in range(args.n_epoch):
        scheduler.step()
        # Training one epoch
        train_loss = train_epoch(model,
                                 optimizer,
                                 train_dataloader,
                                 proxy_isp_dataset,
                                 proxydgc_config,
                                 proxydgc_log_writer,
                                 proxy,
                                 device,
                                 epoch,
                                 criterion_grid=criterion_grid,
                                 criterion_matchability=criterion_match,
                                 loss_grid_weights=weights_loss_coeffs)
        train_losses.append(train_loss)
        print(colored('==> ', 'green') + 'Train average loss:', train_loss)

        # Validation
        val_loss_grid = validate_epoch(model,
                                       val_dataloader,
                                       proxy_isp_dataset,
                                       proxydgc_config,
                                       proxydgc_log_writer,
                                       proxy,
                                       device,
                                       epoch,
                                       criterion_grid=criterion_grid,
                                       criterion_matchability=criterion_match,
                                       loss_grid_weights=weights_loss_coeffs)
        print(colored('==> ', 'blue') + 'Val average grid loss :',
              val_loss_grid)
        print(colored('==> ', 'blue') + 'epoch :', epoch + 1)
        val_losses.append(val_loss_grid)

        np.save(osp.join(args.snapshots, cur_snapshot, 'logs.npy'),
                [train_losses, val_losses])

        if epoch > args.start_epoch:
            '''
            We will be saving only the snapshot which
            has lowest loss value on the validation set
            '''
            cur_snapshot_name = osp.join(args.snapshots,
                                         cur_snapshot,
                                         'epoch_{}.pth'.format(epoch + 1))
            if prev_model is None:
                torch.save({'state_dict': model.module.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           cur_snapshot_name)
                prev_model = cur_snapshot_name
                best_val = val_loss_grid
            else:
                if val_loss_grid < best_val:
                    os.remove(prev_model)
                    best_val = val_loss_grid
                    print('Saved snapshot:', cur_snapshot_name)
                    torch.save({'state_dict': model.module.state_dict(),
                                'optimizer': optimizer.state_dict()},
                               cur_snapshot_name)
                    prev_model = cur_snapshot_name

    print(args.seed, 'Training took:', time.time()-train_started, 'seconds')
