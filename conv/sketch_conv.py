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
        self.type2_sketches = [None] * self.num_sketches
        self.type2_signed_matrices = [None] * self.num_sketches

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
            u1 = (
                (
                    (
                        2
                        * torch.bernoulli(
                            torch.empty(in_channels // self.groups, self.sketch_dim)
                        )
                    )
                    - 1
                )
                / self.sketch_dim
            ).detach()
            self.type1_signed_matrices[idx] = u1
            u2 = (
                ((2 * torch.bernoulli(torch.empty(out_channels, self.sketch_dim))) - 1)
                / self.sketch_dim
            ).detach()
            self.type2_signed_matrices[idx] = u2
            self.type1_sketches[idx] = torch.einsum(
                "ijkl, jm -> imkl", self.weight_tensor, u1
            )  # TODO: does reordering args give better performance?
            self.type2_sketches[idx] = torch.einsum(
                "ijkl, im -> mjkl", self.weight_tensor, u2
            )
