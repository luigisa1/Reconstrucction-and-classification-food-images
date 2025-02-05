from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
from torch.utils.data import Subset
import random
from .datasetFood101 import DatasetFood101
from .sync_transforms import SynchronizedTransforms
import torchvision.transforms as T



def get_mask_params(index):
    if index == 1:
        # Definir parámetros para las máscaras
        mask_params = [
            {'size': 8, 'count': 60},  # Dos máscaras de 8x8 píxeles  # Tres máscaras de 4x4 píxeles
        ]
        
    else:
        raise ValueError(f"Config mask params '{index}' no está definido.")
    
    return mask_params   

test_transforms = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_dataset_by_name_short(dataset_name, path, idex_mask_paramas):
    """
    Carga el dataset basado en su nombre.
    """
    print("Intentando cargar dataset versión reducida para pruebas")
    
    if dataset_name == "food101":
        
        print(f"Cargando dataset: '{dataset_name}.")
        sync_transform = SynchronizedTransforms(crop_size=256, padding=4)
        dataset_train = DatasetFood101(path, mask_params=get_mask_params(idex_mask_paramas), train = True, sync_transform = sync_transform, transform= None, noise_params={"type": "salt_and_pepper", "amount": 0.05})
        dataset_test = DatasetFood101(path, mask_params=get_mask_params(idex_mask_paramas), train = False, sync_transform=None, transform = test_transforms, noise_params={"type": "salt_and_pepper", "amount": 0.05})
        # Reducir a la mitad aleatoriamente
        train_size = len(dataset_train) // 500
        indices = random.sample(range(len(dataset_train)), train_size)

        # Crear un subconjunto
        train_subset = Subset(dataset_train, indices)

        # Reducir a la mitad aleatoriamente
        test_size = len(dataset_test) // 500
        indices = random.sample(range(len(dataset_test)), test_size)

        # Crear un subconjunto
        test_subset = Subset(dataset_test, indices)
        
    else:
        raise ValueError(f"Dataset '{dataset_name}' no está definido.")
    return train_subset, test_subset


def get_dataset_by_name_long(dataset_name, path, idex_mask_paramas):
    """
    Carga el dataset basado en su nombre.
    """
    
    print("Intentando cargar dataset versión completa")
    
    if  dataset_name == "food101":
        
        print(f"Cargando dataset: '{dataset_name}.")
        sync_transform = SynchronizedTransforms(crop_size=256, padding=4)
        dataset_train = DatasetFood101(path, mask_params=get_mask_params(idex_mask_paramas), train = True, sync_transform = sync_transform, transform= None, noise_params={"type": "salt_and_pepper", "amount": 0.05})
        dataset_test = DatasetFood101(path, mask_params=get_mask_params(idex_mask_paramas), train = False, sync_transform=None, transform = test_transforms, noise_params={"type": "salt_and_pepper", "amount": 0.05})
        
    else: 
        raise ValueError(f"Dataset '{dataset_name}' no está definido.")
    return dataset_train, dataset_test


def get_datalaoders(batch_size, dataset_train, dataset_test):
    dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers=4, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers=4, pin_memory=True)
    return dataloader_train, dataloader_test
