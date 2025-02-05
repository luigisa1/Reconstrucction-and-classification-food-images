
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Establecer backend no interactivo
import torch
import wandb
import os

def log_accuracy_train_test(results, config, run_name):
    """
    Genera una gráfica de precisión (accuracy) para entrenamiento y prueba,
    la guarda en local en una ruta específica y la registra en W&B.

    Args:
        results (dict): Diccionario con las métricas de precisión.
        config (obj): Configuración del experimento (incluye `logs_path`).
        run_name (str): Nombre o identificador del run actual.
    """
    # Crear el directorio para guardar la imagen si no existe
    output_dir = os.path.join(config.logs_path, "accuracy_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Definir el nombre del archivo basado en el run_name
    local_file_path = os.path.join(output_dir, f"{run_name}.png")

    # Crear la gráfica con matplotlib
    train_steps = list(range(len(results["accuracy_train"])))
    plt.figure(figsize=(15, 8))
    plt.plot(train_steps, results["accuracy_train"], label="Train Accuracy", marker="o")
    plt.plot(train_steps, results["accuracy_test"], label="Test Accuracy", marker="o")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy over Steps - {run_name}")
    plt.legend()
    plt.grid()
    
    
    # Guardar la gráfica como archivo local
    plt.savefig(local_file_path)

    # Registrar la imagen en una sección personalizada en W&B con identificador
    wandb.log({f"Accuracy Plot": wandb.Image(local_file_path)})

    # Limpiar el gráfico para evitar conflictos en futuras gráficas
    plt.close()

    print(f"Gráfica guardada localmente en: {local_file_path}")


def load_class_names(file_path):
    """
    Cargar nombres de las clases desde un archivo.
    """
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return {idx: name for idx, name in enumerate(class_names)}

def log_reconstructed_images(images, reconstructed_images, masked_imgs, labels, preds, epoch, config_name, config, max_images=8):
    """
    Loguea imágenes originales y reconstruidas en W&B junto con las clases reales y predichas.
    """
    # Cargar nombres de las clases
    class_names_file = "/home/lgimenos98/Reconstruction_and_classification/data/food-101/meta/labels.txt"
    class_names = load_class_names(class_names_file)

    # Función para desnormalizar imágenes
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

    def denormalize(img):
        return img * std + mean

    # Desactivar modo interactivo
    plt.ioff()

    # Seleccionar un subconjunto de imágenes
    images = images[:max_images].cpu()
    reconstructed_images = reconstructed_images[:max_images].cpu()
    masked_imgs = masked_imgs[:max_images].cpu()
    labels = labels[:max_images]  # Clases reales
    preds = preds[:max_images]    # Clases predichas

    # Crear una figura comparativa
    fig, axes = plt.subplots(3, min(len(images), max_images), figsize=(15, 8))

    for i in range(min(len(images), max_images)):
        # Desnormalizar imágenes
        image = denormalize(images[i]).squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
        masked = denormalize(masked_imgs[i]).squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
        reconstructed = denormalize(reconstructed_images[i]).squeeze(0).permute(1, 2, 0).detach().numpy().clip(0, 1)
        
        # Obtener nombres de las clases reales y predichas
        real_label = class_names.get(labels[i], "Unknown")
        pred_label = class_names.get(preds[i], "Unknown")
        
        # Imágenes originales con clase real
        axes[0, i].imshow(image)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Original\nLabel: {real_label}")

        # Imágenes con máscara
        axes[1, i].imshow(masked)
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Masked")

        # Imágenes reconstruidas
        axes[2, i].imshow(reconstructed)
        axes[2, i].axis("off")
        axes[2, i].set_title(f"Reconstructed\nPred: {pred_label}")

    plt.tight_layout()

    # Guardar la figura localmente
    output_dir = os.path.join(config.logs_path, "reconstructed_images")
    os.makedirs(output_dir, exist_ok=True)  # Crear la carpeta si no existe
    fig_path = os.path.join(output_dir, f"{config_name}.png")
    plt.savefig(fig_path)
    plt.close(fig)

    # Loguear la figura como artifact
    artifact = wandb.Artifact(f"reconstructed_images_{config_name}", type="reconstructed_images")
    artifact.add_file(fig_path, name=f"reconstructed_images_epoch_{config_name}")
    wandb.log_artifact(artifact)
    
    # Registrar la imagen en una sección personalizada en W&B con identificador
    wandb.log({f"Reconstructed images": wandb.Image(fig_path)})

    # Cerrar la figura para evitar que se muestre
    plt.close(fig)


def log_confusion_matrix_as_artifact(all_labels, all_preds, num_classes, epoch, config_name, config):
    # Calcula la matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)

    # Genera el heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Epoch')

    # Guarda la imagen temporalmente
    fig_path = os.path.join(config.logs_path, f"{config_name}.png")
    plt.savefig(fig_path)
    plt.close()

    # Subir a W&B como Artifact
    artifact = wandb.Artifact(f"confusion_matrix_{config_name}", type="confusion_matrices")
    artifact.add_file(fig_path, name=f"confusion_matrix_{config_name}")
    wandb.log_artifact(artifact)