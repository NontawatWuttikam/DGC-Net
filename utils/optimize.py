import numpy as np
import cv2

from tqdm import tqdm
import torch
import torch.nn.functional as F
import rawpy


def read_and_process_proxy(raw_path, proxy_isp_dataset, proxy, proxydgc_config):
    adaptivepool2d = torch.nn.AdaptiveAvgPool2d(proxydgc_config["pooled_size"])
    bayer = rawpy.imread(raw_path).raw_image
    #TODO make configurable
    bayer = bayer[540:2460, 1040:2960] # 1920, 1920
    # raw_image = raw_image[1680: 1680 + 640, 1180:1180 + 640] # 640, 640
    # raw_image = raw_image[0:640, 0:640]

    print("process raw with proxy hype:", proxy.return_param_value())
    raw_image = proxy_isp_dataset.preprocess_raw(bayer)

    raw_image = raw_image.to("cuda")

    input_image = proxy(raw_image[None, :, :, :])[0]
    print("MEMORY after proxy forward pass:",  '{:,}'.format(torch.cuda.memory_allocated()))

    proxy_output_image = input_image.cpu().detach()

    input_image = adaptivepool2d(input_image)

    input_image = input_image.clamp(0, 1)

    # input_image = input_image.mean(dim = 0)
    # open("temp_log/after_reduce_image_shape", "w").write(str(input_image.shape))

    # input_image = input_image.astype('float32') / 255.0
    return input_image, proxy_output_image, bayer

def collate_fn(batch):
    collated = {}
    if len(batch) == 0:
        return collated
    
    keys = batch[0].keys()
    
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            values = [sample[key] for sample in batch]
            values = torch.stack(values, dim=0)
        elif key == "correspondence_map_pyro":
            values = []
            for sz in range(len(batch[0][key])):
                scale_samples = []
                for sample in batch:
                    scale_samples.append(sample[key][sz])
                scale_samples  = torch.stack(scale_samples, dim=0)
                values.append(scale_samples)
            # values = torch.stack(values, dim=1)
        # if torch.is_tensor(values[0]):
        assert values is not None, f"Values for key {key} are None"
        collated[key] = values
            # print(f"Error stacking values for key: {key}", values)
        # else:
            # collated[key] = values  # e.g. list of strings, numbers, etc.

    
    return collated

def preprocess_sample_batch(dataset, proxy_isp_dataset, proxydgc_config, proxy, batch):
    output_dicts = []
    for transform_type, source_img_name, theta in zip(batch['transform_type'],
                                                            batch['source_img_name'],
                                                            batch['theta']):
        image, proxy_output_image, bayer = \
           read_and_process_proxy(
               source_img_name,
               proxy_isp_dataset,
               proxy,
               proxydgc_config
            )
        
        output_dict = dataset.process_sample(
            transform_type, image, theta
        )

        output_dicts.append(output_dict)
    
    output = collate_fn(output_dicts)
    return output
        

def train_epoch(net,
                optimizer,
                train_loader,
                proxy_isp_dataset,
                proxydgc_config,
                proxy,
                device,
                criterion_grid,
                criterion_matchability=None,
                loss_grid_weights=None,
                L_coeff=1):
    """
    Training epoch script
    Args:
        net: model architecture
        optimizer: optimizer to be used for traninig `net`
        train_loader: dataloader
        device: `cpu` or `gpu`
        criterion_grid: criterion for esimation pixel correspondence (L1Masked)
        criterion_matchability: criterion for mask optimization
        loss_grid_weights: weight coefficients for each grid estimates tensor
            for each level of the feature pyramid
        L_coeff: weight coefficient to balance `criterion_grid` and
            `criterion_matchability`
    Output:
        running_total_loss: total training loss
    """

    net.train()
    running_total_loss = 0
    running_match_loss = 0
    if loss_grid_weights is None:
        loss_grid_weights = [1, 1, 1, 1, 1]

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in pbar:

        # preprocessing mini-batch
        mini_batch = preprocess_sample_batch(
            train_loader.dataset,
            proxy_isp_dataset,
            proxydgc_config,
            proxy,
            mini_batch
        )

        # print("Mini batch", mini_batch)

        optimizer.zero_grad()

        # net predictions
        estimates_grid, estimates_mask = \
            net(mini_batch['source_image'].to(device),
                mini_batch['target_image'].to(device))

        if criterion_matchability is not None and estimates_mask is None:
            raise ValueError('Cannot use `criterion_matchability` \
                without mask estimates')

        Loss_masked_grid = 0
        EPE_loss = 0

        # grid loss components (over all layers of the feature pyramid):
        for k in range(0, len(estimates_grid)):
            # print("length of correspondence_map_pyro", len(mini_batch['correspondence_map_pyro'][0]))
            # exit(0)
            grid_gt = mini_batch['correspondence_map_pyro'][k].to(device)
            bs, s_x, s_y, _ = grid_gt.shape

            flow_est = estimates_grid[k].permute(0, 2, 3, 1)
            flow_target = grid_gt

            # calculating mask
            mask_x_gt = \
                flow_target[:, :, :, 0].ge(-1) & flow_target[:, :, :, 0].le(1)
            mask_y_gt = \
                flow_target[:, :, :, 1].ge(-1) & flow_target[:, :, :, 1].le(1)
            mask_gt = mask_x_gt & mask_y_gt

            # number of valid pixels based on the mask
            N_valid_pxs = mask_gt.view(1, bs * s_x * s_y).data.sum()

            # applying mask
            mask_gt = torch.cat((mask_gt.unsqueeze(3),
                                 mask_gt.unsqueeze(3)), dim=3).float()
            flow_target_m = flow_target * mask_gt
            flow_est_m = flow_est * mask_gt

            # compute grid loss
            Loss_masked_grid = Loss_masked_grid + \
                loss_grid_weights[k] * criterion_grid(flow_est_m,
                                                      flow_target_m,
                                                      N_valid_pxs)

        Loss_match = 0
        if estimates_mask is not None:
            match_mask_gt = \
                mini_batch['mask_x'][-1].to(device) & \
                mini_batch['mask_y'][-1].to(device)
            Loss_match = \
                criterion_matchability(estimates_mask.squeeze(1),
                                       match_mask_gt)

        Loss = Loss_masked_grid + L_coeff * Loss_match
        Loss.backward()

        optimizer.step()

        running_total_loss += Loss.item()
        if estimates_mask is not None:
            running_match_loss += Loss_match.item()
            pbar.set_description('R_total_loss: %.3f/%.3f | \
                Match_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                          Loss.item(),
                                          running_match_loss / (i + 1),
                                          Loss_match.item()))
        else:
            pbar.set_description(
                'R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                             Loss.item()))

    running_total_loss /= len(train_loader)
    return running_total_loss


def validate_epoch(net,
                   val_loader,
                   proxy_isp_dataset,
                   proxydgc_config,
                   proxy,
                   device,
                   criterion_grid,
                   criterion_matchability=None,
                   loss_grid_weights=None,
                   L_coeff=1):
    """
    Validation epoch script
    Args:
        net: model architecture
        val_loader: dataloader
        device: `cpu` or `gpu`
        criterion_grid: criterion for esimation pixel correspondence (L1Masked)
        criterion_matchability: criterion for mask optimization
        loss_grid_weights: weight coefficients for each grid estimates tensor
            for each level of the feature pyramid
        L_coeff: weight coefficient to balance `criterion_grid` and
            `criterion_matchability`
    Output:
        running_total_loss: total validation loss
    """

    net.eval()
    if loss_grid_weights:
        loss_grid_weights = [1, 1, 1, 1, 1]

    bilinear_coeffs = [16, 8, 4, 2, 1]
    aepe_arrays_240x240 = [[] for _ in bilinear_coeffs]
    running_total_loss = 0
    running_match_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, mini_batch in pbar:

            # preprocessing mini-batch
            mini_batch = preprocess_sample_batch(
                val_loader.dataset,
                proxy_isp_dataset,
                proxydgc_config,
                proxy,
                mini_batch
            )
            # net predictions
            estimates_grid, estimates_mask = \
                net(mini_batch['source_image'].to(device),
                    mini_batch['target_image'].to(device))

            if criterion_matchability is not None and estimates_mask is None:
                raise ValueError('Cannot use criterion_matchability \
                    without mask estimates')

            Loss_masked_grid = 0
            # grid loss components (over all layers of the feature pyramid):
            for k in range(0, len(estimates_grid)):
                grid_gt = mini_batch['correspondence_map_pyro'][k].to(device)
                bs, s_x, s_y, _ = grid_gt.shape

                flow_est = estimates_grid[k].permute(0, 2, 3, 1)
                flow_target = grid_gt

                # calculating mask
                mask_x_gt = flow_target[:, :, :, 0].ge(-1) & \
                    flow_target[:, :, :, 0].le(1)
                mask_y_gt = flow_target[:, :, :, 1].ge(-1) & \
                    flow_target[:, :, :, 1].le(1)
                mask_gt = mask_x_gt & mask_y_gt

                # number of valid pixels based on the mask
                N_valid_pxs = mask_gt.view(1, bs * s_x * s_y).data.sum()

                # applying mask
                mask_gt = torch.cat((mask_gt.unsqueeze(3),
                                     mask_gt.unsqueeze(3)), dim=3).float()
                flow_target_m = flow_target * mask_gt
                flow_est_m = flow_est * mask_gt

                # compute grid loss
                Loss_masked_grid = Loss_masked_grid + \
                    loss_grid_weights[k] * criterion_grid(flow_est_m,
                                                          flow_target_m,
                                                          N_valid_pxs)

            # matchability mask loss
            Loss_match = 0
            if estimates_mask is not None:
                match_mask_gt = mini_batch['mask_x'][-1].to(device) & \
                    mini_batch['mask_y'][-1].to(device)
                Loss_match = criterion_matchability(estimates_mask.squeeze(1),
                                                    match_mask_gt)

            Loss = Loss_masked_grid + L_coeff * Loss_match

            running_total_loss += Loss.item()
            if estimates_mask is not None:
                running_match_loss += Loss_match.item()
                pbar.set_description('R_total_loss: %.3f/%.3f | \
                    Match_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                              Loss.item(),
                                              running_match_loss / (i + 1),
                                              Loss_match.item()))
            else:
                pbar.set_description(
                    'R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                                 Loss.item()))

    return running_total_loss / len(val_loader)
