import pytest
import torch

from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier


def test_classifier_forward_shape(tiny_classifier):
    x = torch.randn(2, 1, 32, 32)
    y = tiny_classifier(x)
    assert y.shape == (2, 1)
    assert y.dtype == x.dtype


def test_classifier_forward_returns_logits():
    model = TrailClassifier(in_channels=1, kernel_size=3, base_channels=4, dropout=0.0)
    with torch.no_grad():
        model.head[3].weight.zero_()
        model.head[3].bias.fill_(2.0)
    model.eval()
    y = model(torch.randn(2, 1, 32, 32))
    assert torch.allclose(y, torch.full_like(y, 2.0))
    assert torch.all(y > 1.0)


def test_classifier_preserves_double_dtype():
    model = TrailClassifier(in_channels=1, kernel_size=3, base_channels=4, dropout=0.0).double()
    x = torch.randn(2, 1, 32, 32, dtype=torch.float64)
    y = model(x)
    assert y.dtype == torch.float64


@pytest.mark.parametrize("shape", [(1, 32, 32), (2, 1, 33, 32), (2, 1, 32, 33)])
def test_classifier_invalid_inputs(shape):
    model = TrailClassifier()
    x = torch.randn(*shape)
    with pytest.raises(ValueError):
        model(x)


def test_classifier_rejects_invalid_channel_count():
    model = TrailClassifier(in_channels=1, kernel_size=3, base_channels=4)
    with pytest.raises(RuntimeError, match="expected input"):
        model(torch.randn(2, 2, 32, 32))


def test_classifier_head_reflects_dropout(tiny_classifier):
    assert tiny_classifier.dropout == 0.2
    assert isinstance(tiny_classifier.head[2], torch.nn.Dropout)
    assert tiny_classifier.head[2].p == 0.2
