from pandas import cut
import torch
import torch.nn as nn


class CutMix:
    def __init__(self, alpha=1.0) -> None:
        if alpha > 0:
            self.alpha = alpha
        else:
            self.alpha = 1
        self.loss_fn = nn.CrossEntropyLoss()
        self.cutmix_ratio = None

    def __call__(self, data, labels):
        batch_size, channel, height, width = data.shape
        indices = torch.randperm(batch_size).cuda()

        low_h, high_h, low_w, high_w = self._rand_box(height, width)
        cutmix_data = data.clone()
        cutmix_data[:, :, low_h:high_h, low_w:high_w] = data[
            indices, :, low_h:high_h, low_w:high_w
        ]

        cutmix_labels = labels[indices]
        self.cutmix_ratio = (high_h - low_h) * (high_w - low_w) / (height * width)

        return cutmix_data, cutmix_labels

    def criterion(self, predictions, labels, cutmix_labels):
        loss = self.loss_fn(predictions, labels)
        cutmix_loss = self.loss_fn(predictions, cutmix_labels)

        return loss * (1 - self.cutmix_ratio) + cutmix_loss * self.cutmix_ratio

    def _rand_box(self, height, width):
        beta = torch.distributions.Beta(self.alpha, self.alpha).sample()
        cut_ratio = torch.sqrt(1 - beta).item()
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)
        center_h, center_w = (
            torch.randint(high=height, size=(1,)),
            torch.randint(high=width, size=(1,)),
        )
        low_h, high_h = (
            torch.clamp(center_h - cut_h // 2, 0, height).item(),
            torch.clamp(center_h + cut_h // 2, 0, height).item(),
        )
        low_w, high_w = (
            torch.clamp(center_w - cut_w // 2, 0, width).item(),
            torch.clamp(center_w + cut_w // 2, 0, width).item(),
        )

        return low_h, high_h, low_w, high_w
