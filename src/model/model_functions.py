import torch
from pathlib import Path
from src.model.custom_model import CustomModel
from src.model.hybrid_model import HybridModel
from src.model.fully_pretrained_model import FullyPretrainedModel

def create_model(model_name, unet_encoder, pretrained_transformer, classes):
    
    print(f"Cargando modelo: '{model_name}'")
    
    if model_name == "custom":
        model = CustomModel(classes)
        return model
    elif model_name == "hybrid":
        model = HybridModel(classes, pretrained_transformer)
        return model
    elif model_name == "fully_pretrained":
        model = FullyPretrainedModel(classes, unet_encoder, pretrained_transformer)
        return model
    else:
        raise ValueError(f"Model '{model_name}' no está definido.")
    
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)