import wandb
import torch
from src.train.train_functions import train
import json
import argparse

def main():
    # Configurar argparse
    parser = argparse.ArgumentParser(description="Carga un archivo JSON para configuración de sweeps.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True, 
        help="Ruta al archivo JSON con la configuración del sweep."
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="prueba_datasets_epocas",
        help="Nombre del proyecto en Weights & Biases"
    )
    args = parser.parse_args()

    # Iniciar sesión en wandb
    wandb.login()

    # Cargar la configuración desde el archivo JSON
    try:
        with open(args.config_path, "r") as file:
            sweep_config = json.load(file)
    except FileNotFoundError:
        print(f"Archivo no encontrado en la ruta: {args.config_path}")
        return

    # Inicializar el sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project_name)

    try:
        # Ejecuta las combinaciones automáticamente
        wandb.agent(sweep_id, function=train)  
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("Error de memoria detectado. Limpiando caché de GPU...")
            torch.cuda.empty_cache()
            # Opcionalmente, reduce el batch_size aquí

if __name__ == "__main__":
    main()
