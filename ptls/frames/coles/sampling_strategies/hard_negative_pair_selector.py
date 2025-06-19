import torch

from ptls.frames.coles.sampling_strategies.pair_selector import PairSelector
from ptls.frames.coles.metric import outer_pairwise_distance


class HardNegativePairSelector(PairSelector):
    """
    困难负样本对选择器。
    该类继承自 `PairSelector`，用于在 Coles 损失计算中选择正样本对和困难负样本对。
    它会生成所有可能的正样本对，并为每个样本选择 `neg_count` 个最难的负样本。
    """

    def __init__(self, neg_count=1):
        """
        初始化 HardNegativePairSelector。

        Args:
            neg_count (int): 为每个样本选择的困难负样本数量，默认为 1。
        """
        super(HardNegativePairSelector, self).__init__()
        self.neg_count = neg_count

    def get_pairs(self, embeddings, labels):
        """
        根据嵌入和标签获取正样本对和硬负样本对。

        Args:
            embeddings (torch.Tensor): 形状为 (batch_size, embedding_dim) 的嵌入张量。
            labels (torch.Tensor): 形状为 (batch_size,) 的标签张量，用于标识样本的类别。

        Returns:
            tuple: 包含两个张量，第一个是正样本对的索引，第二个是负样本对的索引。
                   每个样本对的索引都是 (anchor_idx, positive/negative_idx)。
        """
        # 构建矩阵 x，其中 x_ij == 0 当且仅当 labels[i] == labels[j]
        # 这意味着 x_ij == 0 表示样本 i 和样本 j 属于同一类别。
        n = labels.size(0) # 批次大小
        x = labels.expand(n, n) - labels.expand(n, n).t()

        # 正样本对 (positive pairs)
        # 查找 x == 0 的位置，即标签相同的样本对。
        # torch.triu((x == 0).int(), diagonal=1) 用于获取上三角矩阵，排除对角线元素，避免重复和自指。
        # .nonzero(as_tuple=False) 返回非零元素的索引，即正样本对的索引。
        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)

        # 困难负样本挖掘 (hard negative mining)
        # 计算所有样本嵌入之间的两两距离。
        # .detach() 避免梯度回传到嵌入，因为距离计算仅用于样本选择。
        mat_distances = outer_pairwise_distance(embeddings.detach())  # 两两距离矩阵

        # 设置一个上界，用于将距离转换为“相似度”或“不相似度”的度量，以便进行 topk 选择。
        # 距离越大，表示越不相似，但我们希望选择“最难”的负样本，即距离最小的负样本。
        # 因此，通过 (upper_bound - mat_distances) 将距离转换为一个“相似度”分数，距离越小，分数越大。
        upper_bound = int((2 * n) ** 0.5) + 1
        # 过滤：只保留负样本对的距离。
        # (x != 0) 表示标签不相同的样本对，即负样本对。
        # 将非负样本对的距离设置为 0 (或一个非常小的值)，确保它们不会被选为硬负样本。
        mat_distances = ((upper_bound - mat_distances) * (x != 0).type(
            mat_distances.dtype))  # 过滤：只获取负样本对

        # 从负样本对中选择最难的 `neg_count` 个负样本。
        # .topk(k=self.neg_count, dim=0, largest=True) 沿着 dim=0 (列) 选择最大的 `neg_count` 个值及其索引。
        # 由于我们之前将距离转换为“相似度”分数 (upper_bound - distance)，所以 largest=True 实际上选择了原始距离最小的负样本。
        values, indices = mat_distances.topk(k=self.neg_count, dim=0, largest=True)
        # 构建负样本对的索引。
        # torch.arange(0, n, ...) 创建一个从 0 到 n-1 的序列，表示 anchor 样本的索引。
        # .repeat(self.neg_count) 将每个 anchor 索引重复 `neg_count` 次，因为每个 anchor 会有 `neg_count` 个负样本。
        # torch.cat(indices.unbind(dim=0)) 将 topk 返回的索引连接起来，这些索引是每个 anchor 对应的负样本的索引。
        # .t() 进行转置，使每行是一个 (anchor_idx, negative_idx) 对。
        negative_pairs = torch.stack([
            torch.arange(0, n, dtype=indices.dtype, device=indices.device).repeat(self.neg_count),
            torch.cat(indices.unbind(dim=0))
        ]).t()

        return positive_pairs, negative_pairs