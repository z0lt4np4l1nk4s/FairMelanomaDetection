import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional

class ResNetClassificationModel(nn.Module):
    """
    A ResNet-based classifier model with custom dropout and a two-stage head.
    """

    def __init__(self, dropout_p: float = 0.5) -> None:
        """
        Initializes the ClassifierModel using a pretrained ResNet101 backbone.

        Args:
            dropout_p (float): Dropout probability applied in the classifier head.
        """

        super().__init__()
        base_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

        # Backbone layers
        self.layer0 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = nn.Sequential(base_model.layer3, nn.Dropout(p=0.3))
        self.layer4 = nn.Sequential(base_model.layer4, nn.Dropout(p=0.5))

        # Pooling and classifier head
        self.pool_dropout = nn.Dropout(p=dropout_p)
        in_features = base_model.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features // 2, 1)
        )

        # Layer grouping attributes (optional, useful for fine-tuning)
        self.HEAD_LAYERS = ['fc']
        self.LAST_BLOCK_LAYERS = ['layer4']
        self.PENULTIMATE_LAYERS = ['layer3']
        self.ALL_LAYERS = ['']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with raw logits.
        """

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.pool_dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def set_requires_grad(self, layers_to_unfreeze: Optional[List[str]] = None) -> None:
        """
        Sets requires_grad=True only for selected layers.

        Args:
            layers_to_unfreeze (List[str] or None): List of substrings of parameter names to unfreeze.
        """

        if layers_to_unfreeze is None:
            layers_to_unfreeze = []

        for name, param in self.named_parameters():
            param.requires_grad = any(layer_str in name for layer_str in layers_to_unfreeze)

class EfficientNetClassificationModel(nn.Module):
    """
    A classifier model based on EfficientNet-B4 using timm library.
    """
    def __init__(self, dropout_p: float = 0.30, drop_path_p: float = 0.15) -> None:
        """
        Initializes the EfficientNet-B4 model with customized head.

        Args:
            dropout_p (float): Dropout probability in the classifier.
            drop_path_p (float): Drop path (stochastic depth) probability in backbone.
        """
        super().__init__()

        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=True,
            drop_rate=dropout_p,       # classifier-level dropout
            drop_path_rate=drop_path_p # stochastic-depth inside MBConv blocks
        )

        # Replace the classifier head
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, 1)
        )

        self.HEAD_LAYERS = ['classifier']
        self.LAST_BLOCK_LAYERS = ['blocks.7']
        self.PENULTIMATE_LAYERS = ['conv_head', 'bn2']
        self.ALL_LAYERS = ['']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with logits.
        """
        return self.backbone(x)

    def set_requires_grad(self, layers_to_unfreeze: Optional[List[str]] = None) -> None:
        """
        Sets requires_grad=True only for selected layers.

        Args:
            layers_to_unfreeze (List[str] or None): List of substrings of parameter names to unfreeze.
        """
        if layers_to_unfreeze is None:
            layers_to_unfreeze = []

        for name, param in self.named_parameters():
            param.requires_grad = any(layer_name in name for layer_name in layers_to_unfreeze)
