import torch.nn as nn
import timm
import torch.nn.functional as F
from segmentation_models_pytorch import Unet


class FullyPretrainedModel(nn.Module):
    def __init__(self, num_classes, unet_encoder, pretrained_transformer):
        super(FullyPretrainedModel, self).__init__()
    
        # Modelo preentrenado para reconstrucción
        self.pretrained_unet = Unet(encoder_name=unet_encoder,  # Usa un encoder preentrenado
                                    encoder_weights="imagenet",
                                    in_channels=3,
                                    classes=3)  # Salida de 3 canales

        # Congelar parámetros del encoder (o partes específicas)
        for name, param in self.pretrained_unet.encoder.named_parameters():
            param.requires_grad = False  # Congela el encoder
        
        # Transformer preentrenado
        self.transformer = timm.create_model(pretrained_transformer, pretrained=True)
        in_features = self.transformer.head.in_features
        for param in self.transformer.parameters():
            param.requires_grad = False  # Congela el transformer

        # Reemplaza la cabeza del Transformer
        self.transformer.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Reconstrucción
        out_reconstruction = self.pretrained_unet(x)

        # Clasificación (redimensionar salida reconstruida)
        resized_image = F.interpolate(out_reconstruction, size=(224, 224), mode='bilinear', align_corners=False)
        out_classification = self.transformer(resized_image)

        return out_reconstruction, out_classification