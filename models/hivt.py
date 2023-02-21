# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import LaplaceNLLLoss
from losses import SoftTargetCrossEntropyLoss
from metrics import ADE
from metrics import FDE
from metrics import MR
from models import GlobalInteractor
from models import LocalEncoder
from models import MLPDecoder
from utils import TemporalData

import numpy as np
import matplotlib.pyplot as plt
from argoverse.map_representation.map_api import ArgoverseMap
from Bezier import Bezier

class HiVT(pl.LightningModule):

    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 rotate: bool,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 num_temporal_layers: int,
                 num_global_layers: int,
                 local_radius: float,
                 parallel: bool,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 **kwargs) -> None:
        super(HiVT, self).__init__()
        self.save_hyperparameters()
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.step = 0

        self.local_encoder = LocalEncoder(historical_steps=historical_steps,
                                          node_dim=node_dim,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          num_temporal_layers=num_temporal_layers,
                                          local_radius=local_radius,
                                          parallel=parallel)
        self.global_interactor = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)
        self.decoder = MLPDecoder(local_channels=embed_dim,
                                  global_channels=embed_dim,
                                  future_steps=future_steps,
                                  num_modes=num_modes,
                                  uncertain=True)
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MR()

    def forward(self, data: TemporalData):
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None

        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        return y_hat, pi

    def training_step(self, data, batch_idx):
        y_hat, pi = self(data)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        t = np.arange(0,0.96,0.033)
        print(data.y.shape)
        for index in range(data.y.size(dim=0)):
            y_samples = data.y[index, [0,5,10,15,20,25,29],:].numpy()
            curve = torch.tensor(Bezier.Curve(t,y_samples), dtype=data.y.dtype)
            data.y[index,:,:] = curve
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        loss = reg_loss + cls_loss
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        y_hat, pi = self(data)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        y_hat_agent = y_hat[:, data['agent_index'], :, : 2]
        y_agent = data.y[data['agent_index']]
        '''visualization'''
        for scenario in range (y_agent.size(dim=0)):
            # print(f"scenario #{scenario} | agent #{data['agent_index'][scenario]}'s pi : {pi[data['agent_index'][scenario],:]}")
            # print(f"scenario #{scenario} | agent #{data['agent_index'][scenario]}'s uncertainty : {y_hat[:, data['agent_index'][scenario],0,2:]}")
            self.step += 1
            rotate_mat_agent = torch.empty(y_hat_agent.size(dim=0), 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'][data['agent_index'][scenario]])
            cos_vals = torch.cos(data['rotate_angles'][data['agent_index'][scenario]])
            rotate_mat_agent[:, 0, 0] = cos_vals
            rotate_mat_agent[:, 0, 1] = sin_vals
            rotate_mat_agent[:, 1, 0] = -sin_vals
            rotate_mat_agent[:, 1, 1] = cos_vals
            y_agent_offset = torch.matmul(y_agent[scenario].clone().detach(),torch.tensor([[cos_vals, sin_vals], [-sin_vals, cos_vals]]))
            offset = y_agent_offset - data['positions'][data['agent_index'][scenario],20:,:] 
            y_hat_agent_original = torch.bmm(y_hat_agent[:, scenario, :, : 2].clone().detach(),rotate_mat_agent) - offset
            origin = data['origin'][scenario].numpy()
            theta = data['theta'][scenario]
            rotate_mat = torch.tensor([[torch.cos(theta), torch.sin(theta)],
                                    [-torch.sin(theta), torch.cos(theta)]])
            avm = ArgoverseMap()
            city_name = data['city'][scenario]
            seq_lane_props = avm.city_lane_centerlines_dict[city_name]
            mask = torch.tensor(data['batch'] == scenario, dtype=torch.bool)
            positions_masked = data['positions'][mask]
            positions = torch.matmul(positions_masked, rotate_mat).numpy() + origin
            y_agent_original = torch.matmul(data['positions'][data['agent_index'][scenario],20:,:], rotate_mat).numpy() + origin
            y_agent_past = torch.matmul(data['positions'][data['agent_index'][scenario],:20,:], rotate_mat).numpy() + origin
            x = positions[:,:,0]
            y = positions[:,:,1]
            x_min = np.min(x)
            y_min = np.min(y)
            x_max = np.max(x)
            y_max = np.max(y)
            lane_centerlines = []
            for lane_id, lane_props in seq_lane_props.items():

                lane_cl = lane_props.centerline

                if (
                    np.min(lane_cl[:, 0]) < x_max
                    and np.min(lane_cl[:, 1]) < y_max
                    and np.max(lane_cl[:, 0]) > x_min
                    and np.max(lane_cl[:, 1]) > y_min
                ):
                    lane_centerlines.append(lane_cl)
            # for time in range (50):    
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max) 
            for lane_cl in lane_centerlines:
                plt.plot(
                    lane_cl[:, 0],
                    lane_cl[:, 1],
                    "--",
                    color="grey",
                    alpha=1,
                    linewidth=1,
                    zorder=0,
                )       
            cur_time = 19
            for actor in range(positions_masked.size(dim=0)):
                if actor == data['agent_index'][scenario] - data['av_index'][scenario] and not torch.equal(positions_masked[actor, cur_time, :],torch.tensor([0,0], dtype=torch.float)):
                    plt.scatter(x[actor, cur_time], y[actor, cur_time], s=50, color='r')  # agent
                elif not torch.equal(positions_masked[actor, cur_time, :],torch.tensor([0,0], dtype=torch.float)):
                    plt.scatter(x[actor, cur_time], y[actor, cur_time], s=10)
                elif actor == 0:
                    plt.scatter(x[actor, cur_time], y[actor, cur_time], s=50, color='g') # AV
            # for time in range(29):
            for num in range(y_hat_agent_original.size(dim=0)):
                y_hat_agent_plot = torch.matmul(y_hat_agent_original[num,:,:], rotate_mat) + origin
                # for time in range(29):
                plt.plot(y_hat_agent_plot[:,0].numpy(), y_hat_agent_plot[:,1].numpy(),"--", color="r", alpha=1, linewidth=0.5, zorder=0,)
            plt.plot(y_agent_original[:,0], y_agent_original[:,1],"--", color="g", alpha=1, linewidth=1, zorder=0,)
            plt.plot(y_agent_past[:,0], y_agent_past[:,1],"-", color="gold", alpha=1, linewidth=1, zorder=0,)
            plt.axis('off')
            print(torch.norm(y_hat_agent[:, scenario, -1] - y_agent[scenario, -1], p=2, dim=-1) < 2.0)
            if torch.any(torch.norm(y_hat_agent[:, scenario, -1] - y_agent[scenario, -1], p=2, dim=-1) < 2.0):
                plt.savefig(
                    f"C:/Users/8854373/Desktop/HiVT/val image/HiVT_128/{self.step}.png",
                    bbox_inches="tight",
                    pad_inches=0,
                )  
            else:
                plt.savefig(
                    f"C:/Users/8854373/Desktop/HiVT/val image/HiVT_128/miss{self.step}.png",
                    bbox_inches="tight",
                    pad_inches=0,
                )  
            print(f"{self.step} complete!")
            plt.close()
            # from moviepy.editor import ImageSequenceClip
            # import cv2
            # import os
            # list_video = [f"{x}.png" for x in range(50)]
            # for time in range(50):
            #     img = cv2.imread(f'{time}.png')
            #     img = cv2.resize(img, dsize=(540,390))
            #     cv2.imwrite(f'{time}.png', img)
            # clip = ImageSequenceClip(list_video, fps=2)
            # clip.write_videofile(f"{1}.mp4")
            # for time in range(50):
            #     os.remove(f'{time}.png')
        '''visualization'''
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]
        self.minADE.update(y_hat_best_agent, y_agent)
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HiVT')
        parser.add_argument('--historical_steps', type=int, default=20)
        parser.add_argument('--future_steps', type=int, default=30)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--embed_dim', type=int, required=True)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--num_global_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
