from tqdm.auto import tqdm
import torch
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import wandb
from torch.nn import CrossEntropyLoss
from src.data.dataset_functions import get_dataset_by_name_short, get_dataset_by_name_long, get_datalaoders
from src.model.model_functions import create_model, save_model
import torch.nn.functional as F
from src.utils.logs_wandb import log_reconstructed_images,log_confusion_matrix_as_artifact, log_accuracy_train_test
import matplotlib.pyplot as plt

def train(config = None):

    with wandb.init(config=config, settings=wandb.Settings(console="off")):
        config = wandb.config
        
        # Comprobar combinaciones irrelevantes después del init
        model_type = config["models"]
        pretrained_transformer = config.get("pretrained_transformer", None)
        unet_encoder = config.get("unet_encoder", None)

        # Si el modelo es `custom`, no debe usar ningún componente preentrenado
        if model_type == "custom" and (pretrained_transformer != "deit_base_patch16_224" or unet_encoder != "efficientnet-b0"):
            print(f"Descartando configuración irrelevante para modelo `custom`: models={model_type}, unet_encoder={unet_encoder}, pretrained_transformer={pretrained_transformer}")
            wandb.finish()  # Finaliza inmediatamente este run
            return

        # Si el modelo es `hybrid`, solo el `unet_encoder` debe ser válido
        if model_type == "hybrid" and unet_encoder != "efficientnet-b0":
            print(f"Descartando configuración irrelevante para modelo `hybrid`: models={model_type}, unet_encoder={unet_encoder}, pretrained_transformer={pretrained_transformer}")
            wandb.finish()  # Finaliza inmediatamente este run
            return

        # Para `fully_pretrained`, todas las combinaciones son válidas
        print(f"Configuración válida: models={model_type}, unet_encoder={unet_encoder}, pretrained_transformer={pretrained_transformer}")

        # Nombre personalizado del Run
        # run_name = f"{config.dataset_name}_{config.lenght_dataset}_model_{config.models}_bs_{config.batch_size}_ep_{config.epochs}"
        if model_type == "custom": 
            run_name = f"{config.models}_ep_{config.epochs}"
        elif model_type == "hybrid":
            run_name = f"{config.models}_{config.pretrained_transformer}_ep_{config.epochs}"
        elif model_type == "fully_pretrained":
            run_name = f"{config.models}_{config.unet_encoder}_{config.pretrained_transformer}_ep_{config.epochs}"
            
        wandb.run.name = run_name  # Asigna el nombre del run
        print(f"Run Name: {run_name}")

        results = {
            "train_loss": [],
            "test_loss": [],
            "train_loss_mask": [],
            "test_loss_mask": [],
            "train_loss_labels": [],
            "test_loss_labels": [],
            "accuracy_train": [],
            "accuracy_test": []
        }
        # Cargar el dataset basado en el identificador
        if config.lenght_dataset == "reduced":
            dataset_train, dataset_test = get_dataset_by_name_short(config.dataset_name, config.data_path, config.mask_params)
        elif config.lenght_dataset == "complete":
            dataset_train, dataset_test = get_dataset_by_name_long(config.dataset_name, config.data_path, config.mask_params)

        dataloader_train, dataloader_test = get_datalaoders(batch_size = config.batch_size, dataset_train = dataset_train, dataset_test = dataset_test)

        model=create_model(config.models, config.unet_encoder, config.pretrained_transformer, config.num_classes)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        params_to_update = [param for param in model.parameters() if param.requires_grad]
        loss_fn_mask = MSELoss()
        loss_fn_labels = CrossEntropyLoss()
        optimizer = Adam(params_to_update, lr = 0.0001)
        
        autoencoder_frozen = False
        freeze_threshold = 0.0001
        
        mean = [0.485, 0.456, 0.406]  # Media para normalizar (ImagenNet)
        std = [0.229, 0.224, 0.225]   # Desviación estándar para normalizar


        for epoch in range(config.epochs):
            
  
            mean_loss_mask_train , mean_loss_labels_trian , mean_loss_train, accuracy_train = train_step(model,
                                    dataloader_train = dataloader_train,
                                    loss_fn_mask= loss_fn_mask,
                                    loss_fn_labels = loss_fn_labels,
                                    optimizer = optimizer,
                                    device = device)
            mean_loss_mask_test, mean_loss_labels_test, mean_loss_test, accuracy_test = test_step(model,
                                dataloader_test = dataloader_test,
                                loss_fn_mask= loss_fn_mask,
                                loss_fn_labels = loss_fn_labels,
                                optimizer = optimizer,
                                device = device,
                                epoch = epoch,
                                config = config,
                                mean=mean,
                                std=std)

            if not autoencoder_frozen and mean_loss_mask_test < freeze_threshold:
                
                print("Se congela el encoder y el decoder debido a que la pérdia indica que el entrenamiento ya es óptimo")
                for param in model.encoder.parameters():
                    param.requires_grad = False
                for param in model.decoder.parameters():
                    param.requires_grad = False

            print(f"Accuracy train: {accuracy_train:.4f}, Accuracy test: {accuracy_test:.4f}")
            print(f"Epoch: {epoch+1}, Train Loss: {mean_loss_train:.4f}, Test Loss: {mean_loss_test:.4f}")
            print(f"Train Loss Mask: {mean_loss_mask_train:.4f}, Test Loss Mask: {mean_loss_mask_test:.4f}")
            print(f"Train Loss Labels: {mean_loss_labels_trian:.4f}, Test Loss Labels: {mean_loss_labels_test:.4f}")  

            results["train_loss"].append(mean_loss_train)
            results["test_loss"].append(mean_loss_test)
            results["train_loss_mask"].append(mean_loss_mask_train)
            results["test_loss_mask"].append(mean_loss_mask_test)
            results["train_loss_labels"].append(mean_loss_labels_trian)
            results["test_loss_labels"].append(mean_loss_labels_test)
            results["accuracy_train"].append(accuracy_train)
            results["accuracy_test"].append(accuracy_test)

            # 2️. Log metrics from your script to W&B
            # Log all metrics together for unified charts in W&B
            wandb.log({
                "loss/train": mean_loss_train,
                "loss/test": mean_loss_test,
                "loss_mask/train": mean_loss_mask_train,
                "loss_mask/test": mean_loss_mask_test,
                "loss_labels/train": mean_loss_labels_trian,
                "loss_labels/test": mean_loss_labels_test,
                "accuracy/train": accuracy_train,
                "accuracy/test": accuracy_test
            })
            
        log_accuracy_train_test(results, config, wandb.run.name)
       

        save_filepath = run_name + ".pth"
        save_model(model=model,
                    target_dir=config.model_path,
                    model_name=save_filepath)



def train_step(model: torch.nn.Module,
               dataloader_train: torch.utils.data.DataLoader,
               loss_fn_mask: torch.nn.Module,
               loss_fn_labels: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Dict[str, List]:

    model.train()
    train_loss = 0.0
    train_loss_mask = 0.0
    train_loss_labels = 0.0
    all_preds = []
    all_labels = []

    for imgs, masked_imgs, labels in tqdm(dataloader_train):
        imgs = imgs.to(device, non_blocking=True)
        masked_imgs = masked_imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward
        img_predict, labels_predict = model(masked_imgs)

        loss_mask = loss_fn_mask(img_predict, imgs)
        train_loss_mask += loss_mask.item()

        loss_labels = loss_fn_labels(labels_predict, labels)
        train_loss_labels += loss_labels.item()

        loss = 10*loss_mask + loss_labels
        train_loss += loss.item()

        probs = F.softmax(labels_predict, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().tolist())  # Convierte a listas y añade
        all_labels.extend(labels.cpu().tolist())


        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = accuracy_score(all_labels, all_preds)
        mean_loss_mask = train_loss_mask / len(dataloader_train)
        mean_loss_labels = train_loss_labels / len(dataloader_train)
        mean_loss = train_loss / len(dataloader_train)

    return mean_loss_mask, mean_loss_labels, mean_loss, accuracy


def test_step(model: torch.nn.Module,
              dataloader_test: torch.utils.data.DataLoader,
              loss_fn_mask: torch.nn.Module,
              loss_fn_labels: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              epoch: int,
              config: Any,
              mean: List[float],  # Media utilizada para normalizar las imágenes
              std: List[float],   # Desviación estándar utilizada para normalizar las imágenes
              max_batches_to_log: int = 10  # Número máximo de batches a registrar
              ) -> Dict[str, List]:

    model.eval()
    test_loss = 0.0
    test_loss_mask = 0.0
    test_loss_labels = 0.0
    all_preds = []
    all_labels = []

    reconstruction_table = None  # Inicializa la tabla como None
    logged_batches = 0  # Contador de batches registrados en la tabla
    logged_first_batch = False  # Control para la funcionalidad adicional en el `if`
    
    with torch.no_grad():
        for batch_idx, (imgs, masked_imgs, labels) in enumerate(tqdm(dataloader_test)):
            imgs = imgs.to(device, non_blocking=True)
            masked_imgs = masked_imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward
            img_predict, labels_predict = model(masked_imgs)

            loss_mask = loss_fn_mask(img_predict, imgs)
            test_loss_mask += loss_mask.item()

            loss_labels = loss_fn_labels(labels_predict, labels)
            test_loss_labels += loss_labels.item()

            loss = loss_mask + loss_labels
            test_loss += loss.item()

            probs = F.softmax(labels_predict, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            # **Mantener el `if` existente**: Log de imágenes reconstruidas (solo primer batch de la última época)
            if (epoch + 1) == config.epochs and not logged_first_batch:
                log_reconstructed_images(
                    imgs, img_predict, masked_imgs, labels.cpu().tolist(), preds.cpu().tolist(), 
                    epoch, config_name=wandb.run.name, config=config
                )
                logged_first_batch = True  # Asegura que solo se haga una vez

            # Crear tabla si es la última época y no se han registrado más de `max_batches_to_log`
            if (epoch + 1) == config.epochs and logged_batches < max_batches_to_log:
                if reconstruction_table is None:  # Inicializar tabla la primera vez
                    reconstruction_table = wandb.Table(columns=[
                        "original_image", 
                        "masked_image", 
                        "reconstructed_image", 
                        "true_label", 
                        "predicted_label", 
                        "predicted_probability"
                    ])
                
                # Añadir datos de este batch a la tabla
                for idx, (orig, mask, recon, true, pred, prob) in enumerate(zip(imgs.cpu(), masked_imgs.cpu(), img_predict.cpu(), labels.cpu(), preds.cpu(), probs.cpu())):
                    # **Desnormalizar imágenes usando la media y desviación estándar**
                    def denormalize(image, mean, std):
                        for t, m, s in zip(image, mean, std):
                            t.mul_(s).add_(m)  # Deshacer normalización: x = x * std + mean
                        return image

                    orig_denorm = denormalize(orig.clone(), mean, std).permute(1, 2, 0).numpy()
                    orig_np = (orig_denorm * 255).clip(0, 255).astype("uint8")

                    mask_denorm = denormalize(mask.clone(), mean, std).permute(1, 2, 0).numpy()
                    mask_np = (mask_denorm * 255).clip(0, 255).astype("uint8")

                    recon_denorm = denormalize(recon.clone(), mean, std).permute(1, 2, 0).numpy()
                    recon_np = (recon_denorm * 255).clip(0, 255).astype("uint8")

                    # Recuperar la probabilidad de la clase predicha
                    predicted_probability = prob[pred].item()

                    # Añade la fila a la tabla sin la columna `id`
                    reconstruction_table.add_data(
                        wandb.Image(orig_np, caption="Original"),
                        wandb.Image(mask_np, caption="Masked"),
                        wandb.Image(recon_np, caption="Reconstructed"),
                        true.item(),
                        pred.item(),
                        predicted_probability  # Probabilidad asociada a la predicción
                    )

                logged_batches += 1  # Incrementa el contador de batches registrados

        # Calcular métricas finales
        accuracy = accuracy_score(all_labels, all_preds)
        mean_loss_mask = test_loss_mask / len(dataloader_test)
        mean_loss_labels = test_loss_labels / len(dataloader_test)
        mean_loss = test_loss / len(dataloader_test)

        # Registrar la tabla completa al final de la última época
        if reconstruction_table is not None:
            print(f"Logging table with {len(reconstruction_table.data)} rows to W&B...")
            wandb.log({f"{wandb.run.name}_reconstruction_table": reconstruction_table})
        else:
            print("Reconstruction table is empty, skipping log.")

    return mean_loss_mask, mean_loss_labels, mean_loss, accuracy
