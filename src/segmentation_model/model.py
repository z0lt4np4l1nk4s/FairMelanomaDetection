import torch
import segmentation_models_pytorch as smp
from config import DEVICE

class SegmentationModel(torch.nn.Module):
    """
    A wrapper for a U-Net segmentation model using the segmentation_models_pytorch library.

    Attributes:
        device (torch.device): Device to run the model on ('cpu' or 'cuda').
        model (torch.nn.Module): The U-Net segmentation model.
        encoder (torch.nn.Module): The encoder part of the U-Net model (for feature extraction).
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        decoder_attention_type: str = 'scse',
        dropout: float = 0.3,
    ) -> None:
        """
        Initializes the SegmentationModel with the specified configuration.

        Args:
            encoder_name (str): Name of the encoder backbone (e.g., 'resnet34').
            encoder_weights (str): Pretrained weights for the encoder (e.g., 'imagenet').
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            classes (int): Number of output segmentation classes.
        """
        
        super(SegmentationModel, self).__init__()

        # Set the computation device
        self.device = torch.device(DEVICE)

        # Instantiate the U-Net model
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_attention_type=decoder_attention_type,
            dropout=dropout,
        ).to(self.device)

        # Expose the encoder for external feature extraction if needed
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.head = self.model.segmentation_head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, classes, height, width).
        """

        # Move inputs to the appropriate device
        inputs = inputs.to(self.device)
        
        # Forward pass through the U-Net model
        return self.model(inputs)
