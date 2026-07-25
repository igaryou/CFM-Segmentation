import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Cityscapes
from torchvision.transforms import functional as TF

# まず全部を background/void=19 にする
ID_TO_20CLASS = np.full(256, 19, dtype=np.uint8)

mapping = {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}
for k, v in mapping.items():
    ID_TO_20CLASS[k] = v


class Cityscapes20ClassDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        mode="fine",
        image_size=None,
        crop_size=None,
        augment=False,
        hflip_prob=0.5,
        color_jitter=False,
        color_jitter_brightness=0.2,
        color_jitter_contrast=0.2,
        color_jitter_saturation=0.2,
        color_jitter_hue=0.1,
        imagenet_normalize=False,
    ):
        self.image_size = image_size
        self.crop_size = crop_size
        self.num_classes = 20
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.color_jitter_enabled = color_jitter
        self.imagenet_normalize = imagenet_normalize

        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.color_jitter = transforms.ColorJitter(
            brightness=color_jitter_brightness,
            contrast=color_jitter_contrast,
            saturation=color_jitter_saturation,
            hue=color_jitter_hue,
        )
        self.ds = Cityscapes(
            root=root,
            split=split,
            mode=mode,
            target_type="semantic",
        )

    def __len__(self):
        return len(self.ds)

    def _random_crop(self, image, mask):
        crop_h, crop_w = self.crop_size
        _, h, w = image.shape
        if crop_h > h or crop_w > w:
            raise ValueError(
                f"crop_size {self.crop_size} must be <= image size {(h, w)}"
            )

        top = torch.randint(0, h - crop_h + 1, ()).item()
        left = torch.randint(0, w - crop_w + 1, ()).item()
        image = TF.crop(image, top, left, crop_h, crop_w)
        mask = TF.crop(mask, top, left, crop_h, crop_w)
        return image, mask

    def __getitem__(self, idx):
        image, target = self.ds[idx]

        image = TF.pil_to_tensor(image).float() / 255.0
        target_np = np.array(target, dtype=np.uint8)
        mask = torch.from_numpy(ID_TO_20CLASS[target_np]).long()   # (H, W), 0..19

        if self.augment and torch.rand(()) < self.hflip_prob:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])

        if self.image_size is not None:
            image = TF.resize(
                image,
                self.image_size,
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask = TF.resize(
                mask.unsqueeze(0),
                self.image_size,
                interpolation=TF.InterpolationMode.NEAREST,
            ).squeeze(0).long()

        if self.crop_size is not None:
            image, mask = self._random_crop(image, mask)

        if self.augment and self.color_jitter_enabled:
            image = self.color_jitter(image)
            image = image.clamp(0.0, 1.0)

        if self.imagenet_normalize:
            mean = self.imagenet_mean.to(device=image.device, dtype=image.dtype)
            std = self.imagenet_std.to(device=image.device, dtype=image.dtype)
            image = (image - mean) / std

        onehot = F.one_hot(mask, num_classes=self.num_classes)      # (H, W, 20)
        onehot = onehot.permute(2, 0, 1).float()                   # (20, H, W)

        return image, onehot, mask
