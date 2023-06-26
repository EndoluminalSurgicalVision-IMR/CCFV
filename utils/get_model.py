
from Models.nets.generic_UNet import Generic_UNet, InitWeights_He
from Models.nets.unetr import UNETR
from Models.nets.swin_unetr import SwinUNETR
from Models.nets.vnet import VNet

import torch.nn as nn

def get_model(args, configs):
    if configs['model']['name'] == 'UNETR':
        model = UNETR(in_channels=1, out_channels=configs["num_classes"]+1, img_size=(
        configs['roi_z'], configs['roi_y'], configs['roi_x']), feature_size=48, norm_name='instance')
    elif configs['model']['name'] == 'S4D2W64':
        net_numpool = configs['model']['num_pool']
        model = Generic_UNet(configs['model']['num_input_channels'],
                configs['model']['base_num_features'],
                configs['num_classes']+1,
                configs['model']['num_pool'],
                configs['model']['conv_per_stage'],
                2,
                nn.Conv3d,
                nn.InstanceNorm3d, {'eps': 1e-5, 'affine': True},
                nn.Dropout3d, {'p': 0, 'inplace': True},
                nn.LeakyReLU, {
                    'negative_slope': 1e-2, 'inplace': True},
                configs['model']['deep_supervision'],
                configs['model']['dropout_in_localization'],
                lambda x: x,
                InitWeights_He(1e-2),
                configs['model']['pool_op_kernel_sizes'][:net_numpool],
                configs['model']['conv_kernel_sizes'][:net_numpool+1],
                False, True, True,
                configs['model']['max_num_features'])
    elif configs['model']['name'] == 'SWUNETR':
        model = SwinUNETR(in_channels=1, out_channels=configs["num_classes"]+1, img_size=(
        configs['roi_z'], configs['roi_y'], configs['roi_x']), feature_size=24, norm_name='instance')
    elif configs['model']['name'] == 'VNET':
        model = VNet(spatial_dims=3, in_channels=1, out_channels=configs['num_classes']+1)
    return model