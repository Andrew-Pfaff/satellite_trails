import pytest
import torch

from satellite_trail_segmentation.unet_model.unet import UNet


@pytest.mark.parametrize("height,width", [(32, 48), (64, 32)])
def test_unet_forward_shape(height, width):
    model = UNet(in_channels=1, out_channels=2, base_channels=4, dropout=0.1)
    x = torch.randn(2, 1, height, width)
    y = model(x)
    assert y.shape == (2, 2, height, width)
    assert y.dtype == x.dtype


def test_unet_forward_returns_logits():
    model = UNet(in_channels=1, out_channels=1, base_channels=4, dropout=0.0)
    with torch.no_grad():
        model.final.weight.zero_()
        model.final.bias.fill_(2.0)
    model.eval()
    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    assert y.shape == (1, 1, 32, 32)
    assert torch.allclose(y, torch.full_like(y, 2.0))
    assert torch.all(y > 1.0)


def test_unet_preserves_double_dtype():
    model = UNet(in_channels=1, out_channels=1, base_channels=4, dropout=0.0).double()
    x = torch.randn(1, 1, 32, 32, dtype=torch.float64)
    y = model(x)
    assert y.dtype == torch.float64


@pytest.mark.parametrize("shape", [(1, 64, 64), (2, 1, 65, 64), (2, 1, 64, 65)])
def test_unet_invalid_inputs(shape):
    model = UNet()
    x = torch.randn(*shape)
    with pytest.raises(ValueError):
        model(x)


def test_unet_rejects_invalid_channel_count():
    model = UNet(in_channels=1, out_channels=1, base_channels=4)
    with pytest.raises(RuntimeError, match="expected input"):
        model(torch.randn(2, 2, 32, 32))


def test_unet_dropout_layer():
    assert isinstance(UNet(dropout=0.0)._dropout_layer(), torch.nn.Identity)
    assert isinstance(UNet(dropout=0.5)._dropout_layer(), torch.nn.Dropout2d)


def test_unet_uses_average_pooling():
    assert isinstance(UNet().pool, torch.nn.AvgPool2d)


def test_unet_leaky_relu_slope():
    activations = [module for module in UNet().modules() if isinstance(module, torch.nn.LeakyReLU)]

    assert activations
    assert all(module.negative_slope == 0.1 for module in activations)


def test_unet_batchnorm_can_be_disabled():
    with_batchnorm = UNet(use_batchnorm=True)
    without_batchnorm = UNet(use_batchnorm=False)

    assert any(isinstance(module, torch.nn.BatchNorm2d) for module in with_batchnorm.modules())
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in without_batchnorm.modules())
