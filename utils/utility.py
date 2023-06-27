import json
import torch
import numpy as np
import random

from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    EnsureChannelFirstd
)

from monai.data import (
    DataLoader,
    Dataset
)

import json
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


def load_pretrained_model(ckp_path, model):
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
    
    return model