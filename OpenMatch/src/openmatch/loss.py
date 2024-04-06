import torch
from torch import Tensor
from torch.nn import functional as F
from torch import distributed as dist


class SimpleContrastiveLoss:

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean'):
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)
        logits = torch.matmul(x, y.transpose(0, 1))
        return F.cross_entropy(logits, target, reduction=reduction)


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)
    
class SimpleClusterLoss:
    def _get_cluster_loss(self, q_reps, p_reps, target_per_query):
        """ No in-batch negatives for cluster
        """
        bs = q_reps.shape[0]

        all_scores = []
        for i in range(bs):
            start_idx = i * target_per_query
            end_idx = start_idx + target_per_query
            score = q_reps[i] @ p_reps[start_idx:end_idx].T
            all_scores.append(score)

        all_scores = torch.stack(all_scores)

        target = torch.zeros(all_scores.size(0),
                             device=all_scores.device,
                             dtype=torch.long)

        loss = F.cross_entropy(all_scores, target)
        return loss
    
    def __call__(self, q: Tensor, p: Tensor, cn0: Tensor, cn1: Tensor, cn2: Tensor, cn3: Tensor, cn4: Tensor, reduction: str = 'mean'):
        target_per_qry = p.size(0) // q.size(0)
        hn_target = torch.arange(0, 
                              q.size(0) * target_per_qry, 
                              target_per_qry, 
                              device=q.device, 
                              dtype=torch.long)
        hn_logits = torch.matmul(q, p.transpose(0, 1))
        hn_loss = F.cross_entropy(hn_logits, hn_target, reduction=reduction)

        cn0_loss = self._get_cluster_loss(q, cn0, target_per_qry)
        cn1_loss = self._get_cluster_loss(q, cn1, target_per_qry)
        cn2_loss = self._get_cluster_loss(q, cn2, target_per_qry)
        cn3_loss = self._get_cluster_loss(q, cn3, target_per_qry)
        cn4_loss = self._get_cluster_loss(q, cn4, target_per_qry)

        loss = hn_loss + cn0_loss + cn1_loss + cn2_loss + cn3_loss + cn4_loss

        return loss, [hn_loss, cn0_loss, cn1_loss, cn2_loss, cn3_loss, cn4_loss]

class DistributedClusterLoss(SimpleClusterLoss):
    def __init__(self, scale_loss: bool = True):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(self,  q: Tensor, p: Tensor, cn0: Tensor, cn1: Tensor, cn2: Tensor, cn3: Tensor, cn4: Tensor, **kwargs):
        # print("in distributedclusterloss __call__")
        dist_q = self.gather_tensor(q)
        dist_p = self.gather_tensor(p)
        dist_cn0 = self.gather_tensor(cn0)
        dist_cn1 = self.gather_tensor(cn1)
        dist_cn2 = self.gather_tensor(cn2)
        dist_cn3 = self.gather_tensor(cn3)
        dist_cn4 = self.gather_tensor(cn4)

        loss, all_loss = super().__call__(dist_q, dist_p, dist_cn0, dist_cn1, dist_cn2, dist_cn3, dist_cn4,  **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
            all_loss = [l * self.word_size for l in all_loss]
        return loss, all_loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


class MarginRankingLoss:
    def __init__(self, margin: float = 1.0):
        self.margin = margin
    
    def __call__(self, pos_scores: Tensor, neg_scores: Tensor):
        return torch.mean(F.relu(self.margin - pos_scores + neg_scores))


class SoftMarginRankingLoss:
    def __init__(self, margin: float = 1.0):
        self.margin = margin
    
    def __call__(self, pos_scores: Tensor, neg_scores: Tensor):
        return torch.mean(F.softplus(self.margin - pos_scores + neg_scores))


class BinaryCrossEntropyLoss:
    def __call__(self, pos_scores: Tensor, neg_scores: Tensor):
        return (F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) 
              + F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores)))


class CrossEntropyLoss:
    def __call__(self, pos_scores: Tensor, neg_scores: Tensor):
        return (F.cross_entropy(pos_scores, torch.ones(pos_scores.shape[0], dtype=torch.long).to(pos_scores.device)) 
              + F.cross_entropy(neg_scores, torch.zeros(neg_scores.shape[0], dtype=torch.long).to(pos_scores.device)))


rr_loss_functions = {
    "mr": MarginRankingLoss,
    "smr": SoftMarginRankingLoss,
    "bce": BinaryCrossEntropyLoss,
    "ce": CrossEntropyLoss,
}