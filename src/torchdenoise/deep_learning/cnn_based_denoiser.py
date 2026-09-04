"""
"""

__all__ = []

import torch
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import Conv3d, Conv2d, MaxPool3d, MaxPool2d, ConvTranspose3d, ConvTranspose2d
from torch.nn import LSTM

import pytorch_lightning as pl


class SpatialCNN(nn.Module):
    """2D or 3D CNN for spatial feature extraction."""
    
    def __init__(self, in_channels: int, spatial_dim: int, filters: int = 64):
        """
        Initialize the spatial CNN.

        Parameters
        ----------
        in_channels : int
            Number of input channels (e.g., slices or contrasts).
        spatial_dim : int
            Dimensionality of the spatial data (2 or 3 for 2D+t or 3D+t respectively).
        filters : int, optional
            Number of filters in the first convolution layer, by default 64.
            
        """
        super().__init__()
        if spatial_dim == 2:
            self.conv1 = Conv2d(in_channels, filters, kernel_size=3, padding=1)
            self.pool = MaxPool2d(2)
            self.conv2 = Conv2d(filters, filters*2, kernel_size=3, padding=1)
            self.upconv = ConvTranspose2d(filters*2, filters, kernel_size=2, stride=2)
        else:
            self.conv1 = Conv3d(in_channels, filters, kernel_size=3, padding=1)
            self.pool = MaxPool3d(2)
            self.conv2 = Conv3d(filters, filters*2, kernel_size=3, padding=1)
            self.upconv = ConvTranspose3d(filters*2, filters, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = self.upconv(x)
        return x


class TemporalTransformer(nn.Module):
    """Transformer for capturing temporal correlations."""
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int):
        """
        Initialize the temporal transformer.

        Parameters
        ----------
        d_model : int
            Number of features in the input.
        nhead : int
            Number of heads in the multi-head attention.
        num_layers : int
            Number of layers in the transformer encoder.
        dim_feedforward : int
            Dimension of the feedforward network model.
            
        """
        super().__init__()
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x):
        # x shape: (batch, time, feature_dim)
        x = self.transformer(x)
        return x


class ConvLSTM(nn.Module):
    """ConvLSTM for capturing temporal correlations."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, batch_first: bool = True):
        """
        Initialize the ConvLSTM model.

        Parameters
        ----------
        input_size : int
            The number of features in the input.
        hidden_size : int
            The number of features in the hidden state of the LSTM.
        num_layers : int
            Number of recurrent layers.
        batch_first : bool, optional
            If True, then the input and output tensors are provided as (batch, time, feature), by default True.
        """
        super().__init__()
        self.lstm = LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)

    def forward(self, x):
        # x shape: (batch, time, feature_dim)
        output, _ = self.lstm(x)
        return output


class CNNTemporalDenoiser(pl.LightningModule):
    """Hybrid model with spatial CNN and temporal Transformer/ConvLSTM."""

    def __init__(self, in_channels: int, spatial_dim: int, use_transformer: bool = True, filters: int = 64,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256,
                 temporal_hidden_size: int = 64, num_temporal_layers: int = 2):
        """
        Initialize the hybrid CNN-Temporal Denoiser model.

        Parameters
        ----------
        in_channels : int
            Number of input channels for the spatial CNN.
        spatial_dim : int
            Dimensionality of the spatial data (2 for 2D+t, 3 for 3D+t).
        use_transformer : bool, optional
            If True, use the Transformer for temporal correlations, otherwise use ConvLSTM, by default True.
        filters : int, optional
            Number of filters in the spatial CNN, by default 64.
        d_model : int, optional
            Feature size for the Transformer, by default 64.
        nhead : int, optional
            Number of attention heads in the Transformer, by default 4.
        num_layers : int, optional
            Number of layers in the Transformer, by default 2.
        dim_feedforward : int, optional
            Feedforward dimension for the Transformer, by default 256.
        temporal_hidden_size : int, optional
            Hidden size for the ConvLSTM, by default 64.
        num_temporal_layers : int, optional
            Number of LSTM layers for the ConvLSTM, by default 2.
        """
        super().__init__()
        self.spatial_cnn = SpatialCNN(in_channels, spatial_dim, filters)
        self.use_transformer = use_transformer

        if use_transformer:
            self.temporal = TemporalTransformer(d_model, nhead, num_layers, dim_feedforward)
        else:
            self.temporal = ConvLSTM(d_model, temporal_hidden_size, num_temporal_layers)

    def forward(self, x):
        # Spatial feature extraction (2D+t or 3D+t)
        spatial_features = self.spatial_cnn(x)
        
        # Permute to (batch, time, feature_dim) for temporal processing
        time_dim = -1 if x.dim() == 4 else -2  # based on spatial dimension
        temporal_input = spatial_features.permute(0, time_dim, *range(1, time_dim))  
        
        # Temporal processing (using Transformer or ConvLSTM)
        temporal_output = self.temporal(temporal_input)

        return temporal_output