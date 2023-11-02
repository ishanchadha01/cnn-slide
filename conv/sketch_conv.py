import torch
import torch.nn as nn
import math


class SketchConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_width,
        kernel_height,
        num_sketches,
        sketch_dimension,
        sketch_output=True,
        bias=True,
        device="cpu",
        *,
        num_groups=1,  # TODO: Fix later to accept values != 1
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.device = device
        self.bias = bias
        self.groups = num_groups
        self.num_sketches = num_sketches
        self.sketch_dim = sketch_dimension
        self.type1_sketches = [None] * self.num_sketches
        self.type1_signed_matrices = [None] * self.num_sketches

        if sketch_output and sketch_dimension >= out_channels:
            raise Exception(
                f"Sketch dimension must be less than output channels, \
                    output channels: {out_channels}, sketch dimension: {sketch_dimension}"
            )

        k = math.sqrt(
            self.groups / (self.in_channels * self.kernel_width * self.kernel_height)
        )
        self.weight_tensor = torch.empty(
            self.out_channels,
            self.in_channels // self.groups,
            kernel_height,
            kernel_width,
        ).uniform_(-1 * k, k)

        self.bias_tensor = (
            torch.empty(out_channels).uniform_(-1 * k, k) if self.bias else None
        )
        for idx in range(self.num_sketches):
            # TODO: does reordering args give better performance?
            self.type1_signed_matrices[idx] = (
                (
                    (
                        (
                            2
                            * torch.bernoulli(
                                torch.empty(self.out_channels, self.sketch_dim)
                            )
                        )
                        - 1
                    )
                    / self.sketch_dim
                )
                .detach()
                .T
            )  # sketch_dim x out_channels
            # self.type2_signed_matrices[idx] = u2
            self.type1_sketches[idx] = (
                torch.einsum(
                    "ijkl, im -> mjkl",
                    self.weight_tensor,
                    self.type1_signed_matrices[idx],
                )
                .permute(0, 2, 3, 1)
                .flatten(0, 2)
            )  # (in_channels x h x w) x sketch_dim

    def forward(self, x):
        B, _, H1, W1 = x.shape()
        H2 = H1 - self.kernel_height + 1
        W2 = W1 - self.kernel_width + 1
        input_unfolded = torch.nn.functional.unfold(
            x, (self.kernel_height, self.kernel_width)
        )  # batch x (in_channels * h * w) x L
        result = None
        for idx in range(self.num_sketches):
            u1 = self.type1_signed_matrices[idx]
            s1 = self.type1_sketches[idx]
            input_unfolded = torch.einsum("ijk, jl -> ilk", input_unfolded, s1)
            input_unfolded = torch.einsum("ijk, jl -> ilk", input_unfolded, u1).view(
                B, self.out_channels, H2, W2
            )
            if not result:
                result = input_unfolded / self.num_sketches
            else:
                result += input_unfolded / self.num_sketches
        result += self.bias_tensor[None, :, None, None]
        return result
