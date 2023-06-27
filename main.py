import argparse
import numpy as np
import json
import torch
from utils.sliding_window_sampling import ms_sliding_window_sampling
from utils.get_model import get_model
from utils.ccfv import cal_variety, cal_w_distance
from utils.utility import get_loader, setup_seed, load_pretrained_model

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
    model.cuda()

    ckp_path = args.model_path
    if ckp_path != '':
        model = load_pretrained_model(ckp_path, model)
    else:
        print("Please provide the path to your checkpoint!")
    evaluate(configs, test_loader, model)


if __name__ == '__main__':
    main()