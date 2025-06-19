import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.distributed as dist

from ptls.frames.coles.losses.dist_utils import all_gather_and_cat

class ContrastiveLoss(nn.Module):
    """
    对比损失 (Contrastive Loss)

    该损失函数旨在拉近相似样本的距离，同时推开不相似样本的距离。
    其灵感来源于论文 "Signature verification using a siamese time delay neural network", NIPS 1993
    <mcurl name="论文链接" url="https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf"></mcurl>

    Coles (Contrastive Learning for Sequential Recommendation) 损失是对比学习在序列推荐领域的一种应用。
    其核心思想是：
    1. 对于同一个用户的不同会话（或同一个会话的不同子序列），其嵌入表示应该相互靠近（正样本对）。
    2. 对于不同用户的会话，其嵌入表示应该相互远离（负样本对）。
    3. 通过定义一个“边界值”（margin），使得负样本对的距离只有在小于该边界值时才产生损失，从而避免过度惩罚已经足够远离的负样本。
    """

    def __init__(self, margin, sampling_strategy, distributed_mode=False, do_loss_mult=False):
        """
        初始化对比损失函数。

        Args:
            margin (float): 对比损失的边界值。对于负样本对，只有当它们之间的距离小于此边界时，才会产生损失。
                            这有助于避免对已经足够远离的负样本进行不必要的惩罚。
            sampling_strategy: 用于生成正样本对和负样本对的策略对象。例如，可以基于用户ID或会话ID来定义相似性。
            distributed_mode (bool): 是否在分布式训练模式下运行。如果为True，则会进行跨设备的嵌入和目标聚合。
            do_loss_mult (bool): 在分布式模式下，是否将损失乘以世界大小 (world size)。
                                 这通常用于在分布式训练中保持损失的尺度一致性。
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = sampling_strategy
        self.distributed_mode = distributed_mode
        self.do_loss_mult = do_loss_mult

    def forward(self, embeddings, target):
        """
        计算对比损失。

        Args:
            embeddings (torch.Tensor): 样本的嵌入表示，形状通常为 (batch_size, embedding_dim)。
            target (torch.Tensor): 样本的标签或ID，用于确定正负样本对。例如，可以是用户ID。

        Returns:
            torch.Tensor: 计算得到的对比损失值。
        """
        # 如果处于分布式模式且已初始化，则进行跨设备数据同步
        if dist.is_initialized() and self.distributed_mode:
            dist.barrier()  # 确保所有进程都到达此点，防止死锁
            # 聚合所有设备上的嵌入和目标，以便在所有样本上计算损失
            embeddings = all_gather_and_cat(embeddings)
            # 调整目标ID，使其在所有进程聚合后仍然是唯一的
            target = target + (target.max()+1) * dist.get_rank()
            target = all_gather_and_cat(target)

        # 使用采样策略获取正样本对和负样本对的索引
        # positive_pairs: 形状为 (num_positive_pairs, 2)，每行包含一对正样本的索引
        # negative_pairs: 形状为 (num_negative_pairs, 2)，每行包含一对负样本的索引
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)

        # 计算正样本对的损失
        # 正样本对的损失是它们之间欧氏距离的平方。
        # 目标是使正样本对的距离尽可能小。
        positive_loss = F.pairwise_distance(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]).pow(2)

        # 计算负样本对的损失
        # 负样本对的损失是基于边界值 (margin) 的。
        # F.relu(margin - distance) 意味着只有当距离小于margin时才产生损失。
        # 目标是使负样本对的距离大于或等于margin。
        negative_loss = F.relu(
            self.margin - F.pairwise_distance(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])
        ).pow(2)
        
        # 将正样本损失和负样本损失拼接起来
        loss = torch.cat([positive_loss, negative_loss], dim=0)

        # 在分布式模式下，如果设置了do_loss_mult，则将损失乘以世界大小
        # 这是为了在多GPU/多节点训练时，确保损失的平均值与单GPU训练时保持一致。
        if dist.is_initialized() and self.do_loss_mult:
            loss_mult = dist.get_world_size()
        else:
            loss_mult = 1
            
        # 返回所有损失项的总和，并根据需要乘以损失乘数
        return loss.sum() * loss_mult