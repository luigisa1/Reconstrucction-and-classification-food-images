import torch
from torchvision.transforms import functional as TF
import numpy as np

class SynchronizedTransforms:
    def __init__(self, crop_size=32, padding=4, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.crop_size = crop_size
        self.padding = padding
        self.mean = torch.tensor(mean).view(3, 1, 1)  # Redimensionar para aplicar broadcasting
        self.std = torch.tensor(std).view(3, 1, 1)

    def apply(self, img, params):
        crop_x, crop_y, flip = params
        # Aplicar recorte
        img = TF.crop(img, crop_y, crop_x, self.crop_size, self.crop_size)
        # Aplicar volteo horizontal
        if flip:
            img = TF.hflip(img)
        # Normalizar
        img = (img - self.mean) / self.std
        return img

    def get_params(self, img):
        # Generar parámetros aleatorios para las transformaciones
        _, h, w = img.shape[-3:]  # Obtener dimensiones
        crop_y = np.random.randint(0, self.padding * 2)
        crop_x = np.random.randint(0, self.padding * 2)
        flip = np.random.rand() > 0.5
        return crop_x, crop_y, flip