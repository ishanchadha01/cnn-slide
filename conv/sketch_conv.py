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
        dilation=1,
        padding=0,
        stride=1,
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
        self.type1_signed_matrices = torch.empty(
            size=(num_sketches, self.sketch_dim, self.out_channels), requires_grad=False
        )
        self.type1_sketches = torch.empty(
            size=(
                num_sketches,
                self.in_channels * self.kernel_height * self.kernel_width,
                self.sketch_dim,
            )
        )
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

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
                        2
                        * torch.bernoulli(
                            torch.empty(self.out_channels, self.sketch_dim).uniform_(
                                0.5, 0.5
                            )
                        )
                    )
                    - 1
                )
                / self.sketch_dim
            ).T  # sketch_dim x out_channels
            # self.type2_signed_matrices[idx] = u2

            prod = (
                torch.einsum(
                    "ijkl, mi -> mjkl",
                    self.weight_tensor,
                    self.type1_signed_matrices[idx],
                )
                .permute(2, 3, 1, 0)
                .flatten(0, 2)
            )  # (in_channels x h x w) x sketch_dim

            self.type1_sketches[idx] = prod

    def forward(self, x):
        B, _, H1, W1 = x.shape
        H2 = H1 - self.kernel_height + 1
        W2 = W1 - self.kernel_width + 1
        input_unfolded = (
            torch.nn.functional.unfold(
                x,
                (self.kernel_height, self.kernel_width),
                self.dilation,
                self.padding,
                self.stride,
            )
            .unsqueeze(1)
            .expand(
                B,
                self.num_sketches,
                self.kernel_height * self.kernel_width * self.in_channels,
                -1,
            )
        )  # batch x num_sketches x (in_channels * h * w) x L
        # result = None
        input_unfolded = torch.einsum(
            "bnfe, nfs -> bnse", input_unfolded, self.type1_sketches
        )  # batch x num_sketches x sketch_dim x L
        input_unfolded = torch.einsum(
            "bnse, nso -> bnoe", input_unfolded, self.type1_signed_matrices
        ).view(B, self.num_sketches, self.out_channels, H2, W2)
        input_unfolded = (
            torch.mean(input_unfolded, dim=1) + self.bias_tensor[None, :, None, None]
        )

        # for idx in range(self.num_sketches):
        #     u1 = self.type1_signed_matrices[idx]
        #     s1 = self.type1_sketches[idx]
        #     input_unfolded = torch.einsum("ijk, jl -> ilk", input_unfolded, s1)
        #     input_unfolded = torch.einsum("ijk, jl -> ilk", input_unfolded, u1).view(
        #         B, self.out_channels, H2, W2
        #     )
        #     if not result:
        #         result = input_unfolded / self.num_sketches
        #     else:
        #         result += input_unfolded / self.num_sketches
        # result += self.bias_tensor[None, :, None, None]

        return input_unfolded
