# from matplotlib import bezier
from cv2 import rotate
import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from utils import TemporalData
import argparse
import os
import shutil
import sys
from typing import List

import matplotlib.pyplot as plt

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence
from torch_geometric.utils import subgraph
from typing import List, Optional, Tuple
from Bezier import Bezier


class DistanceDropEdge(object):

    def __init__(self, max_distance: Optional[float] = None, is_circle = None) -> None:
        self.max_distance = max_distance
        self.circle = is_circle

    def __call__(self,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor,
                 heading_angle: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # modified
        if self.max_distance is None:
            return edge_index, edge_attr
        row, col = edge_index
        vector_angle = torch.atan2(edge_attr[:,1], edge_attr[:,0])
        angle_diff = vector_angle - heading_angle[edge_index[0,:]]
        if self.circle:
            mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        else:
            mask = torch.logical_and(torch.abs(torch.norm(edge_attr, p=2, dim=-1) * torch.sin(angle_diff)) < self.max_distance/2,torch.abs(torch.norm(edge_attr, p=2, dim=-1) * torch.cos(angle_diff)+self.max_distance/4) < self.max_distance/2)        
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_attr = edge_attr[mask]
        return edge_index, edge_attr

# for t in range(300):
pt = 36
data = TemporalData()
data = torch.load(f'D:/다운로드/Argoverse1.1/val/processed/{pt}.pt')
cur_time = 19
radius = 88
is_circle = False
drop_edge = DistanceDropEdge(radius ,is_circle)
origin = data['origin'].numpy()
theta = data['theta']
rotate_mat = torch.tensor([[torch.cos(theta), torch.sin(theta)],
                            [-torch.sin(theta), torch.cos(theta)]])
avm = ArgoverseMap()
city_name = data['city']
seq_lane_props = avm.city_lane_centerlines_dict[city_name]
positions = torch.matmul(data['positions'], rotate_mat).numpy() + origin
t = np.arange(0,0.96,0.033)
y_agent_samples = data['positions'][0,20:,:][[0,5,10,15,20,25,29],:].numpy()

# curve = torch.tensor(Bezier.Curve(t,y_agent_samples), dtype=rotate_mat.dtype)
# original_curve = torch.matmul(curve, rotate_mat).numpy() + origin

y_agent_original = torch.matmul(data['positions'][data['agent_index'],20:,:], rotate_mat).numpy() + origin
y_agent_past = torch.matmul(data['positions'][data['agent_index'],:20,:], rotate_mat).numpy() + origin
y_agent_des = y_agent_original[-1, :]

# for index in range(data.y.size(dim=0)):
#     y_samples = data.y[index,[0,5,10,15,20,25,29],:].numpy()
#     curve = torch.tensor(Bezier.Curve(t,y_samples), dtype=rotate_mat.dtype)
#     data.y[index,:,:] = curve
# original_curve_y = torch.matmul(data.y[0], rotate_mat).numpy() + origin
# print(original_curve_y - original_curve)

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
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
nearest_lane = np.array([10000,10000])                                                                                                      
for lane_cl in lane_centerlines:
    min_lane_cl = lane_cl[np.argmin(np.linalg.norm(np.subtract(lane_cl,y_agent_des), axis=1))]
    if np.linalg.norm(nearest_lane - y_agent_des) > np.linalg.norm(min_lane_cl - y_agent_des):
        nearest_lane = min_lane_cl
    plt.plot(
        lane_cl[:, 0],
        lane_cl[:, 1],
        "--",
        color="grey",
        alpha=1,
        linewidth=1,
        zorder=0,
    ) 
t = np.arange(0,0.96,0.033)

if np.linalg.norm(nearest_lane - y_agent_des) > 1.5:
    nodes = np.array([y_agent_original[0].tolist(),y_agent_original[5].tolist(),y_agent_original[10].tolist(),y_agent_original[15].tolist(),y_agent_original[20].tolist(),y_agent_original[25].tolist(),y_agent_original[29].tolist()])
else:
    nodes = np.array([y_agent_original[0].tolist(),y_agent_original[5].tolist(),y_agent_original[10].tolist(),y_agent_original[15].tolist(),y_agent_original[20].tolist(),y_agent_original[25].tolist(),nearest_lane.tolist()])

curve = Bezier.Curve(t,nodes)
print(curve.shape)
# plt.plot(curve[:,0], curve[:,1],"--", color="b", alpha=1, linewidth=1, zorder=0,)
# plt.plot(original_curve[:,0], original_curve[:,1],"--", color="r", alpha=1, linewidth=1, zorder=0,)
# plt.scatter(x[:,cur_time],y[:,cur_time],s=10)
plt.scatter(x[data['agent_index'],cur_time],y[data['agent_index'],cur_time],s=50, color='r')
plt.plot(y_agent_original[:,0], y_agent_original[:,1],"--", color="g", alpha=1, linewidth=1, zorder=0,)
plt.plot(y_agent_past[:,0], y_agent_past[:,1],"-", color="gold", alpha=1, linewidth=1, zorder=0,)
plt.axis('off')
plt.savefig(
    f"C:/Users/KimJihong/Desktop/HiVT/bezier/{pt}.png",
    bbox_inches="tight",
    pad_inches=0,
    ) 
plt.close()
print(f"{pt} complete!")