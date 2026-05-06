# 导入必要的库
import os
import pdb
from tqdm import tqdm  # 进度条显示
import time

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Quantize_kMeans():
    """
    k-means量化类，用于对3D高斯溅射模型的参数进行向量量化压缩
    
    该类实现了基于k-means聚类的参数量化，可以将连续的高斯参数
    映射到离散的聚类中心，从而实现模型压缩。
    """
    def __init__(self, num_clusters=100, num_iters=10):
        """
        初始化k-means量化器
        
        Args:
            num_clusters: 聚类数量
            num_iters: k-means算法迭代次数
        """
        self.num_clusters = num_clusters  # 聚类数量
        self.num_kmeans_iters = num_iters  # k-means迭代次数
        self.nn_index = torch.empty(0)  # 最近邻索引（每个数据点所属的聚类）
        self.centers = torch.empty(0)  # 聚类中心
        self.vec_dim = 0  # 向量维度
        self.cluster_ids = torch.empty(0)  # 聚类ID
        self.cls_ids = torch.empty(0)  # 类别ID
        self.excl_clusters = []  # 排除的聚类列表（过大的聚类）
        self.excl_cluster_ids = []  # 排除聚类的ID列表
        self.cluster_len = torch.empty(0)  # 每个聚类的长度
        self.max_cnt = 0  # 最大聚类大小
        self.n_excl_cls = 0  # 排除的聚类数量

    def get_dist(self, x, y, mode='sq_euclidean'):
        """
        计算x中所有向量与y中所有向量之间的距离
        
        Args:
            x: 输入向量组 (m, dim)
            y: 目标向量组 (n, dim)
            mode: 距离计算模式
            
        Returns:
            dist: 距离矩阵 (m, n)
        """
        if mode == 'sq_euclidean_chunk':
            # 分块计算欧几里得距离，避免内存溢出
            step = 65536
            if x.shape[0] < step:
                step = x.shape[0]
            dist = []
            for i in range(np.ceil(x.shape[0] / step).astype(int)):
                dist.append(torch.cdist(x[(i*step): (i+1)*step, :].unsqueeze(0), y.unsqueeze(0))[0])
            dist = torch.cat(dist, 0)
        elif mode == 'sq_euclidean':
            # 标准欧几里得距离计算
            dist = torch.cdist(x.unsqueeze(0).detach(), y.unsqueeze(0).detach())[0]
        return dist

    def update_centers(self, feat):
        """
        在非聚类分配迭代中使用缓存的最近邻索引更新聚类中心
        
        Args:
            feat: 特征向量
        """
        feat = feat.detach().reshape(-1, self.vec_dim) # [N, 3]
        # 在单次操作中更新除排除聚类外的所有聚类
        # 在末尾添加一个零填充的虚拟元素
        feat = torch.cat([feat, torch.zeros_like(feat[:1]).cuda()], 0)
        self.centers = torch.sum(feat[self.cluster_ids, :].reshape(
            self.num_clusters, self.max_cnt, -1), dim=1)
        if len(self.excl_cluster_ids) > 0:
            for i, cls in enumerate(self.excl_clusters):
                # 聚类中点的数量除法在下面所有聚类的一次性平均中完成
                # 这里只添加较大聚类中的额外元素
                self.centers[cls] += torch.sum(feat[self.excl_cluster_ids[i], :], dim=0)
        self.centers /= (self.cluster_len + 1e-6)  # 避免除零错误

    def update_centers_(self, feat, cluster_mask=None, nn_index=None, avg=False):
        """
        在聚类分配期间使用掩码矩阵乘法更新聚类中心
        掩码从距离矩阵获得
        
        Args:
            feat: 特征向量
            cluster_mask: 聚类掩码
            nn_index: 最近邻索引
            avg: 是否进行平均
            
        Returns:
            centers: 更新后的聚类中心
        """
        feat = feat.detach().reshape(-1, self.vec_dim)
        centers = (cluster_mask.T @ feat)
        if avg:
            # 注意：这里需要传入counts参数，当前实现中未使用
            pass  # self.centers /= counts.unsqueeze(-1)
        return centers

    def equalize_cluster_size(self):
        """
        通过添加虚拟元素使所有聚类的大小相同
        
        找到聚类中元素的最大数量，通过添加虚拟元素使所有聚类的大小
        等于最大聚类的大小。如果最大值太大，则排除它并考虑下一个最大的。
        对排除的聚类使用for循环，对剩余的聚类使用单次操作来更新聚类中心。
        """
        unq, n_unq = torch.unique(self.nn_index, return_counts=True) # (K), (K)
        # 找到最大聚类大小并排除超过阈值的聚类
        topk = 100
        if len(n_unq) < topk:
            topk = len(n_unq)
        max_cnt_topk, topk_idx = torch.topk(n_unq, topk)
        self.max_cnt = max_cnt_topk[0]
        idx = 0
        self.excl_clusters = []
        self.excl_cluster_ids = []
        # 排除过大的聚类（超过5000个元素）
        while(self.max_cnt > 5000):
            self.excl_clusters.append(unq[topk_idx[idx]])
            idx += 1
            if idx < topk:
                self.max_cnt = max_cnt_topk[idx]
            else:
                break
        self.n_excl_cls = len(self.excl_clusters)
        self.excl_clusters = sorted(self.excl_clusters)
        
        # 存储每个聚类的元素索引
        all_ids = []
        cls_len = []
        for i in range(self.num_clusters):
            cur_cluster_ids = torch.where(self.nn_index == i)[0]
            # 对于排除的聚类，只使用前max_cnt个元素与其他聚类一起进行平均
            # 单独对排除聚类的剩余元素进行平均
            cls_len.append(torch.Tensor([len(cur_cluster_ids)]))
            if i in self.excl_clusters:
                self.excl_cluster_ids.append(cur_cluster_ids[self.max_cnt:])
                cur_cluster_ids = cur_cluster_ids[:self.max_cnt]
            # 添加虚拟元素以使所有聚类具有相同大小
            all_ids.append(torch.cat([cur_cluster_ids, -1 * torch.ones((self.max_cnt - len(cur_cluster_ids)),
                                                                       dtype=torch.long).cuda()]))
        all_ids = torch.cat(all_ids).type(torch.long)
        cls_len = torch.cat(cls_len).type(torch.long)
        self.cluster_ids = all_ids
        self.cluster_len = cls_len.unsqueeze(1).cuda()
        self.cls_ids = self.nn_index

    def cluster_assign(self, feat, feat_scaled=None):
        """
        执行k-means聚类分配
        
        Args:
            feat: 输入特征向量
            feat_scaled: 缩放后的特征向量（可选）
        """
        # 使用k-means进行量化
        feat = feat.detach()
        feat = feat.reshape(-1, self.vec_dim) # 将特征重塑成(num_points, vec_dim) (N, 3)
        if feat_scaled is None: # 如果缩放后的特征为空，则使用原始特征
            feat_scaled = feat
            scale = feat[0] / (feat_scaled[0] + 1e-8) # 计算缩放因子
        if len(self.centers) == 0: # 如果聚类中心为空，则随机初始化聚类中心
            # 随机初始化聚类中心
            self.centers = feat[torch.randperm(feat.shape[0])[:self.num_clusters], :]

        # 开始k-means迭代
        chunk = True
        counts = torch.zeros(self.num_clusters, dtype=torch.float32).cuda() + 1e-6 # (K, 1)
        centers = torch.zeros_like(self.centers) # (K, 3)
        for iteration in range(self.num_kmeans_iters):
            # 分块处理以避免内存问题
            if chunk:
                self.nn_index = None
                i = 0
                chunk = 10000
                while True:
                    dist = self.get_dist(feat[i*chunk:(i+1)*chunk, :], self.centers) # (N, K)
                    curr_nn_index = torch.argmin(dist, dim=-1) # (N)
                    # 当到多个聚类的距离相同时，分配单个聚类
                    dist = F.one_hot(curr_nn_index, self.num_clusters).type(torch.float32)
                    curr_centers = self.update_centers_(feat[i*chunk:(i+1)*chunk, :], dist, curr_nn_index, avg=False)
                    counts += dist.detach().sum(0) + 1e-6
                    centers += curr_centers
                    if self.nn_index == None:
                        self.nn_index = curr_nn_index
                    else:
                        self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
                    i += 1
                    if i*chunk > feat.shape[0]:
                        break

            self.centers = centers / counts.unsqueeze(-1)
            # 重新初始化为0
            centers[centers != 0] = 0.
            counts[counts > 0.1] = 0.

        # 最终分配使用缩放后的特征
        if chunk:
            self.nn_index = None
            i = 0
            # chunk = 100000
            while True:
                dist = self.get_dist(feat_scaled[i * chunk:(i + 1) * chunk, :], self.centers)
                curr_nn_index = torch.argmin(dist, dim=-1)
                if self.nn_index == None:
                    self.nn_index = curr_nn_index
                else:
                    self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
                i += 1
                if i * chunk > feat.shape[0]:
                    break
        self.equalize_cluster_size()

    def rescale(self, feat, scale=None):
        """
        通过除以最大值将特征缩放到[-1, 1]范围内
        
        Args:
            feat: 输入特征
            scale: 缩放因子（可选）
            
        Returns:
            缩放后的特征
        """
        if scale is None:
            return feat / (abs(feat).max(dim=0)[0] + 1e-8)
        else:
            return feat / (scale + 1e-8)

    def forward_pos(self, gaussian, assign=False, mask=None):
        """
        对3D位置坐标进行k-means量化
        
        Args:
            gaussian: 高斯模型对象
            assign: 是否进行聚类分配（True）还是仅更新中心（False）
            mask: 可选的高斯点掩码，只对指定点进行量化
        """
        if self.vec_dim == 0:
            self.vec_dim = gaussian._xyz.shape[1]
        
        # 如果提供了掩码，只处理指定的高斯点
        if mask is not None:
            xyz_data = gaussian._xyz[mask]
        else:
            xyz_data = gaussian._xyz
            
        if assign:
            self.cluster_assign(xyz_data)
        else:
            self.update_centers(xyz_data)
        
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        
        # 聚类完成，不直接修改高斯模型属性
        # 量化值将通过GaussianModel的get_quantized_*方法按需计算

    def forward_dc(self, gaussian, assign=False, mask=None):
        """
        对颜色DC分量进行k-means量化
        
        Args:
            gaussian: 高斯模型对象
            assign: 是否进行聚类分配（True）还是仅更新中心（False）
            mask: 可选的高斯点掩码，只对指定点进行量化
        """
        if self.vec_dim == 0:
            self.vec_dim = gaussian._features_dc.shape[1] * gaussian._features_dc.shape[2]
        
        # 如果提供了掩码，只处理指定的高斯点
        if mask is not None:
            dc_data = gaussian._features_dc[mask]
        else:
            dc_data = gaussian._features_dc
            
        if assign:
            self.cluster_assign(dc_data)
        else:
            self.update_centers(dc_data)
        
        # 使用聚类中心替换原始DC分量
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        
        # 聚类完成，不直接修改高斯模型属性
        # 量化值将通过GaussianModel的get_quantized_*方法按需计算

    def forward_frest(self, gaussian, assign=False, mask=None):
        """
        对球谐函数剩余系数进行k-means量化
        
        Args:
            gaussian: 高斯模型对象
            assign: 是否进行聚类分配（True）还是仅更新中心（False）
            mask: 可选的高斯点掩码，只对指定点进行量化
        """
        if self.vec_dim == 0:
            self.vec_dim = gaussian._features_rest.shape[1] * gaussian._features_rest.shape[2]
        
        # 如果提供了掩码，只处理指定的高斯点
        if mask is not None:
            frest_data = gaussian._features_rest[mask]
        else:
            frest_data = gaussian._features_rest
            
        if assign:
            self.cluster_assign(frest_data)
        else:
            self.update_centers(frest_data)
        
        deg = gaussian._features_rest.shape[1]
        # 使用聚类中心替换原始球谐函数系数
        # sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        
        # 聚类完成，不直接修改高斯模型属性
        # 量化值将通过GaussianModel的get_quantized_*方法按需计算

    def forward_scale(self, gaussian, assign=False, mask=None):
        """
        对缩放参数进行k-means量化
        
        Args:
            gaussian: 高斯模型对象
            assign: 是否进行聚类分配（True）还是仅更新中心（False）
            mask: 可选的高斯点掩码，只对指定点进行量化
        """
        if self.vec_dim == 0:
            self.vec_dim = gaussian._scaling.shape[1] # [N, 3]
        
        # 如果提供了掩码，只处理指定的高斯点
        if mask is not None:
            scale_data = gaussian._scaling[mask]
        else:
            scale_data = gaussian._scaling
            
        if assign:
            self.cluster_assign(scale_data)
        else:
            self.update_centers(scale_data)
        
        # 使用聚类中心替换原始缩放参数
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        
        # 聚类完成，不直接修改高斯模型属性
        # 量化值将通过GaussianModel的get_quantized_*方法按需计算

    def forward_rot(self, gaussian, assign=False, mask=None):
        """
        对旋转参数进行k-means量化
        
        Args:
            gaussian: 高斯模型对象
            assign: 是否进行聚类分配（True）还是仅更新中心（False）
            mask: 可选的高斯点掩码，只对指定点进行量化
        """
        if self.vec_dim == 0:
            self.vec_dim = gaussian._rotation.shape[1]
        
        # 如果提供了掩码，只处理指定的高斯点
        if mask is not None:
            rot_data = gaussian._rotation[mask]
        else:
            rot_data = gaussian._rotation
            
        if assign:
            self.cluster_assign(rot_data)
        else:
            self.update_centers(rot_data)
        
        # 使用聚类中心替换原始旋转参数
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        
        # 聚类完成，不直接修改高斯模型属性
        # 量化值将通过GaussianModel的get_quantized_*方法按需计算

    def forward_scale_rot(self, gaussian, assign=False, mask=None):
        """
        将缩放和旋转参数组合进行单个k-means量化
        
        Args:
            gaussian: 高斯模型对象
            assign: 是否进行聚类分配（True）还是仅更新中心（False）
            mask: 可选的高斯点掩码，只对指定点进行量化
        """
        if self.vec_dim == 0:
            self.vec_dim = gaussian._rotation.shape[1] + gaussian._scaling.shape[1]
        
        # 如果提供了掩码，只处理指定的高斯点
        if mask is not None:
            scale_data = gaussian._scaling[mask]
            rot_data = gaussian._rotation[mask]
        else:
            scale_data = gaussian._scaling
            rot_data = gaussian._rotation
        
        # 对缩放和旋转参数进行归一化后拼接
        feat_scaled = torch.cat([self.rescale(scale_data), self.rescale(rot_data)], 1)
        feat = torch.cat([scale_data, rot_data], 1)
        
        if assign:
            self.cluster_assign(feat, feat_scaled)
        else:
            self.update_centers(feat)
        
        # 分别更新缩放和旋转参数
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        
        # 聚类完成，不直接修改高斯模型属性
        # 量化值将通过GaussianModel的get_quantized_*方法按需计算

    def forward_dcfrest(self, gaussian, assign=False, mask=None):
        """
        将DC分量和球谐函数剩余系数组合进行单个k-means量化
        
        Args:
            gaussian: 高斯模型对象
            assign: 是否进行聚类分配（True）还是仅更新中心（False）
            mask: 可选的高斯点掩码，只对指定点进行量化
        """
        if self.vec_dim == 0:
            self.vec_dim = (gaussian._features_rest.shape[1] * gaussian._features_rest.shape[2] +
                            gaussian._features_dc.shape[1] * gaussian._features_dc.shape[2])
        
        # 如果提供了掩码，只处理指定的高斯点
        if mask is not None:
            dc_data = gaussian._features_dc[mask]
            frest_data = gaussian._features_rest[mask]
        else:
            dc_data = gaussian._features_dc
            frest_data = gaussian._features_rest
        
        if assign:
            self.cluster_assign(torch.cat([dc_data, frest_data], 1))
        else:
            self.update_centers(torch.cat([dc_data, frest_data], 1))
        
        deg = gaussian._features_rest.shape[1]
        # 分别更新DC分量和球谐函数系数
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        
        # 聚类完成，不直接修改高斯模型属性
        # 量化值将通过GaussianModel的get_quantized_*方法按需计算

    def replace_with_centers(self, gaussian):
        """
        用聚类中心替换球谐函数特征
        
        Args:
            gaussian: 高斯模型对象
        """
        deg = gaussian._features_rest.shape[1]
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_rest = gaussian._features_rest - gaussian._features_rest.detach() + sampled_centers.reshape(-1, deg, 3)
