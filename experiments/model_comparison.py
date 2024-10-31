import numpy as np
import torch
from shapecentral.networks import diffusion_net, delta_conv, point_net


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    n_params = np.sum([param.nelement() for param in model.parameters()])
    return size_all_mb, n_params


model1 = diffusion_net.layers.DiffusionNet(C_in=1, C_out=30, C_width=64, N_block=4,
                                           last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
                                           outputs_at='global_mean', dropout=True)

model2 = diffusion_net.layers.DiffusionNet(C_in=2, C_out=30, C_width=64, N_block=4,
                                           last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
                                           outputs_at='global_mean', dropout=True)

model3 = diffusion_net.layers.DiffusionNet(C_in=3, C_out=30, C_width=64, N_block=4,
                                           last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
                                           outputs_at='global_mean', dropout=True)

model16 = diffusion_net.layers.DiffusionNet(C_in=16, C_out=30, C_width=64, N_block=4,
                                            last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
                                            outputs_at='global_mean', dropout=True)

model64 = diffusion_net.layers.DiffusionNet(C_in=64, C_out=30, C_width=64, N_block=4,
                                            last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
                                            outputs_at='global_mean', dropout=True)


mb_size1, n_params1 = get_model_size(model1)
mb_size2, n_params2 = get_model_size(model2)
mb_size3, n_params3 = get_model_size(model3)
mb_size16, n_params16 = get_model_size(model16)
mb_size64, n_params64 = get_model_size(model64)

print(f'DiffusionNet, with 4 blocks of width 128, has sizes:\n'
      f'For C_in=1 : {n_params1}\n'
      f'For C_in=2 : {n_params2}\n'
      f'For C_in=3 : {n_params3}\n'
      f'For C_in=16 : {n_params16}\n'
      f'For C_in=64 : {n_params64}\n')

model1 = point_net.PointNetClassification(in_channels=1, n_classes=30)
model2 = point_net.PointNetClassification(in_channels=2, n_classes=30)
model3 = point_net.PointNetClassification(in_channels=3, n_classes=30)
model16 = point_net.PointNetClassification(in_channels=16, n_classes=30)
model64 = point_net.PointNetSegmentation(in_channels=64, n_classes=30)


mb_size1, n_params1 = get_model_size(model1)
mb_size2, n_params2 = get_model_size(model2)
mb_size3, n_params3 = get_model_size(model3)
mb_size16, n_params16 = get_model_size(model16)
mb_size64, n_params64 = get_model_size(model64)

print(f'PointNet++ has sizes:\n'
      f'For C_in=1 : {n_params1}\n'
      f'For C_in=2 : {n_params2}\n'
      f'For C_in=3 : {n_params3}\n'
      f'For C_in=16 : {n_params16}\n'
      f'For C_in=64 : {n_params64}\n')

model1 = delta_conv.models.DeltaNetSegmentation(in_channels=1, num_classes=260, conv_channels=[128, 128, 128, 128],
                                                mlp_depth=2, embedding_size=256, num_neighbors=20)
model2 = delta_conv.models.DeltaNetSegmentation(in_channels=2, num_classes=260, conv_channels=[128, 128, 128, 128],
                                                mlp_depth=2, embedding_size=256, num_neighbors=20)
model3 = delta_conv.models.DeltaNetSegmentation(in_channels=3, num_classes=260, conv_channels=[128, 128, 128, 128],
                                                mlp_depth=2, embedding_size=256, num_neighbors=20)
model16 = delta_conv.models.DeltaNetSegmentation(in_channels=16, num_classes=260, conv_channels=[128, 128, 128, 128],
                                                 mlp_depth=2, embedding_size=256, num_neighbors=20)
model64 = delta_conv.models.DeltaNetSegmentation(in_channels=64, num_classes=260, conv_channels=[128, 128, 128, 128],
                                                 mlp_depth=2, embedding_size=256, num_neighbors=20)


mb_size1, n_params1 = get_model_size(model1)
mb_size2, n_params2 = get_model_size(model2)
mb_size3, n_params3 = get_model_size(model3)
mb_size16, n_params16 = get_model_size(model16)
mb_size64, n_params64 = get_model_size(model64)

print(f'DeltaNet, with 4 blocks of width 128, and embedding size of 512 has sizes:\n'
      f'For C_in=1 : {n_params1}\n'
      f'For C_in=2 : {n_params2}\n'
      f'For C_in=3 : {n_params3}\n'
      f'For C_in=16 : {n_params16}\n'
      f'For C_in=64 : {n_params64}\n')

breakpoint()
