from src.model.conv_blok import ConvBlock
import torch.nn as nn
import timm
import torch 
import torch.nn.functional as F

class HybridModel(nn.Module):
    """
    U-Net combinado con un Vision Transformer preentrenado.
    """
    def __init__(self, num_classes, pretrained_transformer):
        super(HybridModel, self).__init__()

        # Encoder
        self.encoder = nn.ModuleList([
            ConvBlock(3, 64),      # enc1
            ConvBlock(64, 128),    # enc2
            ConvBlock(128, 256),   # enc3
            ConvBlock(256, 512)    # enc4
        ])
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                ConvBlock(512 + 512, 512)  # dec4
            ),
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                ConvBlock(256 + 256, 256)  # dec3
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                ConvBlock(128 + 128, 128)  # dec2
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                ConvBlock(64 + 64, 64)    # dec1
            )
        ])
        self.final = nn.Conv2d(64, 3, kernel_size=1)
        
        # Vision Transformer preentrenado
        self.transformer = timm.create_model(pretrained_transformer, pretrained=True)
        in_features = self.transformer.head.in_features
        for param in self.transformer.parameters():
            param.requires_grad = False

        # Reemplaza la capa final para adaptarla al número de clases
        self.transformer.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Encoder
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
            x = F.max_pool2d(x, 2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, layer in enumerate(self.decoder):
            x = layer[0](x)  # upconv
            x = layer[1](torch.cat([x, features[-(i + 1)]], dim=1))  # concat + conv

        # Reconstrucción de la imagen
        out_reconstruction = self.final(x)

        # Clasificación con el Vision Transformer
        resized_image = F.interpolate(out_reconstruction, size=(224, 224), mode='bilinear', align_corners=False)
        out_classification = self.transformer(resized_image)

        return out_reconstruction, out_classification