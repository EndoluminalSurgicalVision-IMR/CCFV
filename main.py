import argparse
import numpy as np
import json
import torch
import json

import torch
from scipy.linalg import sqrtm
from utils.sliding_window_sampling import ms_sliding_window_sampling
from utils.get_model import get_model

from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    EnsureChannelFirstd
)
import json
import numpy as np
from monai.data import (
    DataLoader,
    Dataset
)

import random


def get_loader(args, configs):
    data_list = json.load(open(args.data_list_file))
    val_list = data_list['val']
    val_transforms = [
        LoadImaged(keys=["data", "seg"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["data", "seg"]),
        NormalizeIntensityd(
            keys=["data"], subtrahend=configs['mean'], divisor=configs['std'])
    ]
    val_transforms = Compose(val_transforms)
    val_ds = Dataset(data=val_list,
                     transform=val_transforms)
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            num_workers=4,
                            shuffle=False,
                            drop_last=False)

    return val_loader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(configs, test_loader, model):
    model.eval()
    print("begin evaluation!")
    layers = configs["layers"]
    class_feature_dict = {layer: {j: [] for j in range(
        configs["num_classes"]+1)} for layer in layers}
    global_feat_dict = {layer: [] for layer in layers}
    with torch.no_grad():
        for idx, val_data in enumerate(test_loader):
            print("evaluating index:{}".format(idx))
            try:
                val_data["data"] = val_data["data"].permute(
                    0, 1, 4, 3, 2).cuda()
                data_device = "cuda"
            except RuntimeError as e:     # cuda out of memory
                print(
                    "cuda out of memory occurred! Try to transfer the data to cpu!")
                val_data["data"] = val_data["data"].permute(
                    0, 1, 4, 3, 2).to(torch.device('cpu'))
                data_device = "cpu"

            """inference"""
            if data_device == "cuda":
                # import pdb; pdb.set_trace()
                data_seg = val_data["seg"].permute(0, 1, 4, 3, 2).cuda()
                data_seg = torch.nn.functional.interpolate(
                    val_data['seg'], size=val_data["data"].shape[2:], mode="nearest")
                layers = configs['layers']
                feature_dict = ms_sliding_window_sampling(layers, configs["sample_num"],
                                                          val_data["data"], data_seg, [configs['roi_z'],
                                                                                       configs['roi_y'], configs['roi_x']],
                                                          configs['sw_batch_size'],
                                                          model,
                                                          overlap=configs['infer_overlap'], mode=configs[
                    'window_mode'])
                for layer in class_feature_dict.keys():
                    global_feat = np.concatenate(
                        [feat for feat in feature_dict[layer].values()], axis=0)
                    global_feat = np.mean(global_feat, axis=0)
                    global_feat_dict[layer].append(global_feat)
                    for lb in class_feature_dict[layer].keys():
                        class_feature_dict[layer][lb].append(
                            feature_dict[layer][lb])
    ccfv = 0
    for layer in class_feature_dict.keys():
        w_distance = 0.0
        global_feature = np.array(global_feat_dict[layer])
        var_f = cal_variety(global_feature)

        for lb in class_feature_dict[layer].keys():
            if lb == 0:
                continue
            length = len(class_feature_dict[layer][lb])
            for i in range(length):
                for j in range(i+1, length):
                    w_distance += cal_w_distance(
                        class_feature_dict[layer][lb][i], class_feature_dict[layer][lb][j])
            w_distance += w_distance / (length * length / 2) / configs["num_classes"]
        ccfv += np.log(w_distance / var_f)

    print("ccfv:", ccfv)


def cal_w_distance(f1, f2):
    miu_f1 = np.mean(f1, axis=0)
    miu_f2 = np.mean(f2, axis=0)
    cov_f1 = np.cov(f1.T)
    cov_f2 = np.cov(f2.T)
    delta_miu = miu_f1 - miu_f2
    w_d = np.sum(delta_miu**2) + cov_f1.trace() + cov_f2.trace() - \
        2 * (sqrtm((cov_f1.dot(cov_f2)))).trace()

    return np.sqrt(abs(w_d))


def cal_variety(matrix):
    eps = 1e-3
    row_norms = np.sum(matrix * matrix, axis=1)
    pairwise_inner_products = np.dot(matrix, matrix.T)
    pairwise_distances_squared = np.expand_dims(
        row_norms, axis=1) + np.expand_dims(row_norms, axis=0) - 2 * pairwise_inner_products
    pairwise_distances = 1 / np.sqrt(pairwise_distances_squared+eps)
    return np.sum(np.triu(pairwise_distances, k=1)) / len(pairwise_distances) / (len(pairwise_distances)-1) * 2


def main():
    parser = argparse.ArgumentParser(description='PyTorch Evaluation')
    parser.add_argument('--data_list_file', type=str,
                        help='json file of data paths')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to the config file')
    parser.add_argument('--model_path', type=str, default='')
    args = parser.parse_args()

    setup_seed(0)
    print(args)
    configs = json.load(open(args.cfg, 'r'))
    print(configs)
    test_loader = get_loader(args, configs)
    model = get_model(args, configs)
    # print(model)
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)

    print(f'Total parameters count {pytorch_total_params}')
    model.cuda()

    ckp_path = args.model_path
    if ckp_path != '':
        try:
            pretrained_dict = torch.load(ckp_path)['state_dict']
            new_state_dict = {}
            for k, value in pretrained_dict.items():
                key = k
                if key.startswith('module.'):
                    key = key[7:]
                new_state_dict[key] = value
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in new_state_dict.items() if
                               (k in model_dict) and (model_dict[k].shape == new_state_dict[k].shape)}

            for k, v in model_dict.items():
                if k not in pretrained_dict.keys():
                    print(k)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f'loading checkpoint from {ckp_path}')
        except:
            print("Checkpoint does not exist!")
    evaluate(configs, test_loader, model)


if __name__ == '__main__':
    main()