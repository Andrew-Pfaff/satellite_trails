import pytest
import torch

from satellite_trail_segmentation.unet_model.attention_unet import AttentionGate, AttentionUNet
from satellite_trail_segmentation.unet_model.unet import UNet


@pytest.mark.parametrize("height,width", [(32, 48), (64, 32)])
def test_attention_unet_forward_shape(height, width):
    model = AttentionUNet(in_channels=1, out_channels=2, base_channels=4, dropout=0.1)
    x = torch.randn(2, 1, height, width)
    y = model(x)
    assert y.shape == (2, 2, height, width)
    assert y.dtype == x.dtype


def test_attention_unet_forward_returns_logits():
    model = AttentionUNet(in_channels=1, out_channels=1, base_channels=4, dropout=0.0)
    with torch.no_grad():
        model.final.weight.zero_()
        model.final.bias.fill_(2.0)
    model.eval()
    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    assert y.shape == (1, 1, 32, 32)
    assert torch.allclose(y, torch.full_like(y, 2.0))
    assert torch.all(y > 1.0)


def test_attention_unet_preserves_double_dtype():
    model = AttentionUNet(in_channels=1, out_channels=1, base_channels=4, dropout=0.0).double()
    x = torch.randn(1, 1, 32, 32, dtype=torch.float64)
    y = model(x)
    assert y.dtype == torch.float64


def test_attention_unet_invalid_input_rank():
    model = AttentionUNet()
    with pytest.raises(ValueError, match="4D"):
        model(torch.randn(1, 64, 64))


@pytest.mark.parametrize("shape", [(2, 1, 65, 64), (2, 1, 64, 65)])
def test_attention_unet_rejects_spatial_dimensions_not_divisible_by_16(shape):
    model = AttentionUNet()
    with pytest.raises(ValueError, match="divisible by 16"):
        model(torch.randn(*shape))


def test_attention_unet_rejects_invalid_channel_count():
    model = AttentionUNet(in_channels=1, out_channels=1, base_channels=4)
    with pytest.raises(RuntimeError, match="expected input"):
        model(torch.randn(2, 2, 32, 32))


def test_attention_unet_dropout_layer():
    assert isinstance(AttentionUNet(dropout=0.0)._dropout_layer(), torch.nn.Identity)
    assert isinstance(AttentionUNet(dropout=0.5)._dropout_layer(), torch.nn.Dropout2d)


def test_attention_unet_uses_average_pooling():
    assert isinstance(AttentionUNet().pool, torch.nn.AvgPool2d)


def test_attention_unet_leaky_relu_slope():
    activations = [module for module in AttentionUNet().modules() if isinstance(module, torch.nn.LeakyReLU)]

    assert activations
    assert all(module.negative_slope == 0.1 for module in activations)


def test_attention_unet_batchnorm_can_be_disabled():
    with_batchnorm = AttentionUNet(use_batchnorm=True)
    without_batchnorm = AttentionUNet(use_batchnorm=False)

    assert any(isinstance(module, torch.nn.BatchNorm2d) for module in with_batchnorm.modules())
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in without_batchnorm.modules())


def test_attention_gate_preserves_skip_shape():
    gate = AttentionGate(skip_channels=8, gating_channels=8)
    skip = torch.randn(2, 8, 16, 16)
    gating = torch.randn(2, 8, 16, 16)
    out = gate(skip, gating)
    assert out.shape == skip.shape


def test_attention_gate_coefficients_shape_and_range():
    gate = AttentionGate(skip_channels=8, gating_channels=8)
    skip = torch.randn(2, 8, 16, 16)
    gating = torch.randn(2, 8, 16, 16)

    coefficients = gate.attention_coefficients(skip, gating)

    assert coefficients.shape == (2, 1, 16, 16)
    assert torch.all(coefficients >= 0)
    assert torch.all(coefficients <= 1)
    assert (skip * coefficients).shape == skip.shape


def test_attention_gate_rejects_mismatched_spatial_dimensions():
    gate = AttentionGate(skip_channels=8, gating_channels=8)
    with pytest.raises(ValueError, match="matching spatial dimensions"):
        gate(torch.randn(2, 8, 16, 16), torch.randn(2, 8, 8, 8))


def test_attention_unet_parameter_count_close_to_baseline():
    unet_params = sum(parameter.numel() for parameter in UNet(base_channels=8).parameters())
    attention_params = sum(parameter.numel() for parameter in AttentionUNet(base_channels=8).parameters())

    assert attention_params >= unet_params
    assert attention_params / unet_params < 1.05
