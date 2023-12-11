from conv import sketch_conv
import torch


def test_sketch_conv2d_dims():
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    rgb_input = torch.rand(16, 3, 128, 128)
    conv_layer = sketch_conv.SketchConv2d(3, 32, 3, 5, 8, 8, device=device)
    result = conv_layer.forward(rgb_input)
    assert result.shape[0] == 16
    assert result.shape[1] == 32
    assert result.shape[2] == 128 - 5 + 1
    assert result.shape[3] == 128 - 3 + 1
