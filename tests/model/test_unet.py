import pytest
import torch
from satellite_trail_segmentation.model.unet import UNet

def test_unet_forward_shape(dummy_image, dummy_unet):
    """Test if the forward pass yields a tensor of the same shape as the input."""
    output = dummy_unet(dummy_image)
    assert output.shape == dummy_image.shape, "Output shape should match input shape."

def test_unet_invalid_ndim():
    """Test that the model rejects inputs that are not 4D."""
    model = UNet()
    bad_input = torch.randn(1, 64, 64)  # Missing batch dimension (3D)
    with pytest.raises(ValueError, match="Expected a 4D input tensor"):
        model(bad_input)

def test_unet_invalid_dimensions():
    """Test that the model rejects inputs not divisible by 16."""
    model = UNet()
    bad_input = torch.randn(2, 1, 65, 65)  # 65 is not divisible by 16
    with pytest.raises(ValueError, match="must be divisible by 16"):
        model(bad_input)

def test_unet_dropout_layer():
    """Test that dropout is applied when configured."""
    model_no_dropout = UNet(dropout=0.0)
    assert isinstance(model_no_dropout._dropout_layer(), torch.nn.Identity)

    model_with_dropout = UNet(dropout=0.5)
    assert isinstance(model_with_dropout._dropout_layer(), torch.nn.Dropout2d)
    assert model_with_dropout._dropout_layer().p == 0.5