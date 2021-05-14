import torch

from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction
from pytorch_metric_learning.utils import common_functions as c_f


class NPairsLoss(BaseMetricLossFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_to_recordable_attributes(name="num_pairs", is_stat=True)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, embeddings, labels, indices_tuple):
        anchor_idx, positive_idx = lmu.convert_to_pos_pairs_with_unique_labels(
            indices_tuple, labels
        )
        self.num_pairs = len(anchor_idx)
        if self.num_pairs == 0:
            return self.zero_losses()
        anchors, positives = embeddings[anchor_idx], embeddings[positive_idx]
        targets = c_f.to_device(torch.arange(self.num_pairs), embeddings)
        sim_mat = self.distance(anchors, positives)
        if not self.distance.is_inverted:
            sim_mat = -sim_mat
        return {
            "loss": {
                "losses": self.cross_entropy(sim_mat, targets),
                "indices": anchor_idx,
                "reduction_type": "element",
            }
        }

    def get_default_distance(self):
        return DotProductSimilarity()

    def forward(self, embeddings, labels, a, b, indices_tuple=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        c_f.check_shapes(embeddings, labels)
        labels = c_f.to_device(labels, embeddings)
        loss_dict = self.compute_loss(embeddings, labels, indices_tuple)
        length=len(loss_dict['loss']['losses'])
        l2norm=torch.norm(embeddings[loss_dict['loss']['indices']],2)/length
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        mlloss=self.reducer(loss_dict, embeddings, labels)
        return a*mlloss+b*l2norm