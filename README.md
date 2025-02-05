# Reconstrucción y Clasificación de Imágenes Food-101

Este proyecto se centra en la reconstrucción y clasificación de imágenes alteradas de la base de datos **Food-101**. Las imágenes han sido modificadas aplicándoles una máscara (pixeles negros) y ruido, para luego ser reconstruidas utilizando autoencoders. Posteriormente, estas imágenes reconstruidas son clasificadas mediante modelos transformer.

Se implementan y evalúan tres configuraciones de modelos para abordar este problema, además de integrar un servidor y una página web sencilla para la predicción interactiva.

---

## Modelos Implementados

En este proyecto se prueban tres enfoques principales para el pipeline de reconstrucción y clasificación:

1. **Custom**:
   - Todo el pipeline se construye manualmente, sin usar modelos preentrenados.
2. **Hybrid**:
   - El autoencoder se construye de manera manual, pero el transformer utilizado para la clasificación es un modelo preentrenado.
3. **Fully Pretrained**:
   - Tanto el autoencoder como el transformer son modelos preentrenados.

Los modelos preentrenados utilizados en las combinaciones son:
- `Unet` con diferentes encoders ` efficientnet-b0` , ` resnet50`
- Y los siguientes transfomers `deit_base_patch16_224`, `vit_base_patch16_224`

---

## Experimentación con WandB

Se utiliza **Weights & Biases (WandB)** para realizar experimentos y registrar métricas automáticamente. Los archivos de configuración en formato JSON (`sweep_config_prueba.json` y `sweep_config.json`) definen las combinaciones de parámetros y modelos a evaluar. 

Las métricas registradas incluyen:
- Pérdidas (Loss).
- Precisión (Accuracy).
- Reconstrucciones de imágenes.
- Tablas de inferencias para visualizar pruebas.

Además, se utiliza un agente de WandB para lanzar las combinaciones de parámetros y modelos.

---

## Servidor y Página Web

El proyecto incluye un servidor y una página web básica para ejecutar predicciones. Estos permiten cargar imágenes, realizar predicciones y mostrar los resultados reconstruidos y clasificados.

- **Servidor**: Implementado en `server/main.py`.
- **Página Web**: Implementada en `web/` con un frontend sencillo (`index.html`, `app.js`, `styles.css`).

Ejecuta el servidor y la página web desde bash:
### Inicia el servidor web:
```bash
/tu/ruta/web$ http-server --index index.html
```
### Inicia el servidor backend:
```bash
/tu/ruta/server$ uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```
Y conectate a laweb a través de http://127.0.0.1:8080/


---

## Entrenamiento

Para entrenar los modelos, se utiliza el script `train.py`. Antes de ejecutarlo, adapta los parámetros en el archivo de configuración JSON (`sweep_config_prueba.json` o `sweep_config.json`) según tus necesidades. 

Ejecuta el entrenamiento con:
```bash
python3 train.py --config_path "/tu/ruta/archivo_config.json" --project_name "Nombre_del_proyecto"


