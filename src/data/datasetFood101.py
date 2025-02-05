import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize

class DatasetFood101(Dataset):
    def __init__(self, path, mask_params=None, train=True, transform=None, sync_transform=None, noise_params=None):
        """
        Args:
        - path (str): Ruta al dataset.
        - mask_params (list of dict): Lista de parámetros para las máscaras. Cada dict debe contener:
        - 'size': Tamaño de la máscara (en píxeles).
        - 'count': Número de máscaras a aplicar.
        - train (bool): Si es True, usa los datos de entrenamiento.
        - transform (callable): Transformaciones adicionales para aplicar.
        - sync_transform (SynchronizedTransforms): Transformaciones sincronizadas entre original y enmascarada.
        """
        self.path = path
        self.mask_params = mask_params
        self.noise_params = noise_params
        self.transform = transform
        self.sync_transform = sync_transform
        self.train = train

        if train:
            if self.mask_params != None:
                print("Se va aproceder a aplicar máscaras a las imágenes")
                
            if self.noise_params != None:
                print("Se va aproceder a aplicar ruido a las imágenes")
                
        # Leer los datos de división de entrenamiento/prueba
        self.images, self.labels = self.load_metadata()

    def add_masks_and_noise(self, img, mask_params, noise_params):
        img = img.clone()
        _, h, w = img.shape

        # Añadir máscaras (cuadrados negros)
        if mask_params:
            for params in mask_params:
                size = params.get('size', 0)
                count = params.get('count', 0)
                for _ in range(count):
                    if size > 0:
                        y = np.random.randint(0, h - size + 1)
                        x = np.random.randint(0, w - size + 1)
                        img[:, y:y+size, x:x+size] = 0

        # Añadir ruido
        if noise_params:
            noise_type = noise_params.get('type', 'gaussian')
            if noise_type == 'gaussian':
                mean = noise_params.get('mean', 0)
                std = noise_params.get('std', 0.1)
                noise = torch.randn_like(img) * std + mean
                img = img + noise
                img = torch.clamp(img, 0, 1)  # Asegurarse de que los valores estén en el rango [0, 1]

            elif noise_type == 'salt_and_pepper':
                amount = noise_params.get('amount', 0.05)
                mask = torch.rand_like(img) < amount
                salt = (torch.rand_like(img) < 0.5) * 1.0  # Generar "sal" (píxeles blancos)
                pepper = 1 - salt  # Generar "pimienta" (píxeles negros)
                img[mask] = salt[mask] + pepper[mask]

        return img

    def load_metadata(self):
        """
        Carga la lista de imágenes y etiquetas desde los archivos de metadatos (train.txt o test.txt).
        """
        subset = "train" if self.train else "test"
        images = []
        labels = []
        
        meta_file = os.path.join(self.path, "meta", f"{subset}.txt")
        with open(meta_file, "r") as f:
            for line in f:
                # Cada línea tiene el formato "<class_name>/<image_id>.jpg"
                rel_path = line.strip()
                label = rel_path.split('/')[0]
                images.append(os.path.join(self.path, "images", rel_path + ".jpg"))
                labels.append(label)

        # Cargar las clases desde classes.txt para convertir etiquetas a índices
        class_file = os.path.join(self.path, "meta", "classes.txt")
        with open(class_file, "r") as f:
            classes = [line.strip() for line in f]
        label_to_index = {label: idx for idx, label in enumerate(classes)}

        # Convertir etiquetas a índices
        labels = [label_to_index[label] for label in labels]
        return images, labels

    def __getitem__(self, index):
        # Cargar imagen y etiqueta
        image_path = self.images[index]
        image = read_image(image_path).float() / 255.0  # Cargar imagen como tensor normalizado
        label = self.labels[index]
        
        # Redimensionar a 256x256
        image = resize(image, [256, 256])

        # Aplicar máscaras
        masked_img = self.add_masks_and_noise(image.clone(), self.mask_params, self.noise_params)

        # Transformaciones sincronizadas
        if self.sync_transform:
            params = self.sync_transform.get_params(image)
            image = self.sync_transform.apply(image, params)
            masked_img = self.sync_transform.apply(masked_img, params)

        # Transformaciones adicionales
        if self.transform:
            image = self.transform(image)
            masked_img = self.transform(masked_img)

        return image, masked_img, label

    def __len__(self):
        return len(self.images)

