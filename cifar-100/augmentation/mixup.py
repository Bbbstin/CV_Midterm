import torch
import torch.nn as nn


class Mixup:
    def __init__(self, alpha=1.0) -> None:
        if alpha > 0:
            self.alpha = torch.Tensor([alpha])
        else:
            self.alpha = torch.Tensor([1.0])
        self.loss_fn = nn.CrossEntropyLoss()
        self.mixup_ratio = None

    def __call__(self, data, labels):
        self.mixup_ratio = (
            torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        )

        batch_size = labels.shape[0]
        indices = torch.randperm(batch_size).cuda()

        mixup_data = (1 - self.mixup_ratio) * data + self.mixup_ratio * data[indices, :]
        mixup_labels = labels[indices]

        return mixup_data, mixup_labels

    def criterion(self, predictions, labels, mixup_labels):
        loss = self.loss_fn(predictions, labels)
        mixup_loss = self.loss_fn(predictions, mixup_labels)

        return loss * (1 - self.mixup_ratio) + mixup_loss * self.mixup_ratio
