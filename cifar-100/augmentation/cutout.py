import torch
import torch.nn as nn


class Cutout:
    def __init__(self, half_size=8) -> None:
        self.half_size = half_size
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self, data, labels):
        batch_size, channel, height, width = data.shape
        low_h, high_h, low_w, high_w = self._rand_box(height, width)

        cutout_data = data.clone()
        cutout_data[:, :, low_h:high_h, low_w:high_w] = 0

        return cutout_data, None

    def criterion(self, predictions, labels, cutout_labels=None):
        return self.loss_fn(predictions, labels)

    def _rand_box(self, height, width):
        center_h, center_w = (
            torch.randint(high=height, size=(1,)),
            torch.randint(high=width, size=(1,)),
        )
        low_h, high_h = (
            torch.clamp(center_h - self.half_size, 0, height).item(),
            torch.clamp(center_h + self.half_size, 0, height).item(),
        )
        low_w, high_w = (
            torch.clamp(center_w - self.half_size, 0, width).item(),
            torch.clamp(center_w + self.half_size, 0, width).item(),
        )

        return low_h, high_h, low_w, high_w
