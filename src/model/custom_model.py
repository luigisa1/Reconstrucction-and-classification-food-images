from src.model.conv_blok import ConvBlock
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from src.model.conv_blok import ConvBlock
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CustomModel(nn.Module):
    """
    Modelo personalizado con U-Net para reconstrucción y un Transformer para clasificación.
    """
    def __init__(self, num_classes, embed_dim=128, num_heads=4, num_layers=2, seq_length=49):
        super(CustomModel, self).__init__()

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

        # Transformer-based classifier
        self.embedding = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((seq_length, seq_length))
        )

        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, num_classes)  # Solo una capa lineal
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

        # Clasificación con el Transformer
        embeddings = self.embedding(out_reconstruction)
        embeddings = embeddings.flatten(start_dim=2).permute(2, 0, 1)  # Prepare for transformer (Seq, Batch, Embed)
        transformer_output = self.transformer_encoder(embeddings)

        # Use only the last token for classification
        last_token = transformer_output[-1]  # (batch_size, embed_dim)
        out_classification = self.classifier(last_token)

        return out_reconstruction, out_classification

