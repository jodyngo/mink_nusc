# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Dict
from torch.nn import functional as F
import numpy as np
import torch
from mmdet3d.models import Det3DDataPreprocessor
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from torch import Tensor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from segment_anything import SamPredictor, sam_model_registry
import time

@MODELS.register_module()
class MinkNuscDataPreprocessorTest(Det3DDataPreprocessor):
    def calculate_euclidean_distance(self, points: List[Tensor]) -> List[Tensor]:
        distances = []
        for point_cloud in points:
            # point_cloud shape torch.Size([38120, 4])
            xyz = point_cloud[:, :3]
            distance = torch.sqrt(torch.sum(xyz ** 2, dim=1))
            distances.append(distance)
        return distances

    def compute_distance(self, points: Tensor) -> Tensor:
        """Compute distance from points to a reference point (e.g., sensor origin)."""
        # Subtract each point's (x, y, z) coordinates from the reference point
        xyz = points[:, :3]
        distance = torch.sqrt(torch.sum(xyz ** 2, dim=1))
        return distance

    def simple_process(self, data: dict, training: bool = False) -> dict:
        """modify simple_process to include distance information so that these distances are pre_computed and stored with the data
        """
        start_time = time.time()
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            if self.voxel:
                # print("voxelize available")
                voxel_dict = self.voxelize(inputs['points'], data_samples)
                batch_inputs['voxels'] = voxel_dict
                batch_inputs['voxel_weights'] = voxel_dict['voxel_weights']

        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs
            
        end_time = time.time()
        print(f"Processing time simple_process: {end_time - start_time:.4f} seconds")

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    @torch.no_grad()
    def voxelize(self, points: List[Tensor],
                 data_samples: SampleList) -> Dict[str, Tensor]:
        """Apply voxelization to point cloud and compute voxel weights based on distance.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples (list[:obj:`Det3DDataSample`]): The annotation data
                of every sample. Add voxel-wise annotation for segmentation.
        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
            - voxel_weights (Tensor, optional): Weights of voxels based on distance.
        """

        voxel_dict = dict()

        if self.voxel_type == 'minkunet':
            voxels, coors, distances, voxel_weights = [], [], [], []
            voxel_size = points[0].new_tensor(self.voxel_layer.voxel_size)
            for i, (res, data_sample) in enumerate(zip(points, data_samples)):
                res_coors = torch.round(res[:, :3] / voxel_size).int()
                res_coors -= res_coors.min(0)[0]

                res_coors_numpy = res_coors.cpu().numpy()
                inds, point2voxel_map = self.sparse_quantize(
                    res_coors_numpy, return_index=True, return_inverse=True)
                point2voxel_map = torch.from_numpy(point2voxel_map).cuda()
                if self.training and self.max_voxels is not None:
                    if len(inds) > self.max_voxels:
                        inds = np.random.choice(
                            inds, self.max_voxels, replace=False)
                inds = torch.from_numpy(inds).cuda()
                if hasattr(data_sample.gt_pts_seg, 'pts_semantic_mask'):
                    data_sample.gt_pts_seg.voxel_semantic_mask \
                        = data_sample.gt_pts_seg.pts_semantic_mask[inds]
                res_voxel_coors = res_coors[inds]
                res_voxels = res[inds]

                voxel_centers = res_voxel_coors.float() * voxel_size
                distance = torch.norm(voxel_centers, dim=1)

                weights = self.sigmoid(distance) + 1

                if self.batch_first:
                    res_voxel_coors = F.pad(
                        res_voxel_coors, (1, 0), mode='constant', value=i)
                    data_sample.batch_idx = res_voxel_coors[:, 0]
                else:
                    res_voxel_coors = F.pad(
                        res_voxel_coors, (0, 1), mode='constant', value=i)
                    data_sample.batch_idx = res_voxel_coors[:, -1]
                data_sample.point2voxel_map = point2voxel_map.long()
                voxels.append(res_voxels)
                coors.append(res_voxel_coors)
                voxel_weights.append(weights)

            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)
            voxel_weights = torch.cat(voxel_weights, dim=0)

        else:
            raise ValueError(f'Invalid voxelization type {self.voxel_type}')

        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors
        voxel_dict['voxel_weights'] = voxel_weights

        return voxel_dict

    def sigmoid(self, distance, scale=1.0):
        return 1.0 / (1.0 + torch.exp(-scale * distance))

    def softmax(self, distance):
        exp_dist = torch.exp(-distance)
        return exp_dist / torch.sum(exp_dist, dim=0, keepdim=True)  # BS = 2 softmax mIOU: 0.4309 local minimal epoch 76

    def gaussian(self, distance, sigma=1.0):
        return torch.exp(-0.5 * (distance / sigma) ** 2)  # too small

    def inverse_distance_weighting(self, distance, epsilon=1e-8):
        return 1.0 / (distance + epsilon)

    def exponential(self, distance, alpha=1.0):
        return torch.exp(-alpha * distance)


