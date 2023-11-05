import torch
import torch.nn as nn
import math
from conv.alsh_conv import ALSHConv, zero_fill_missing


class SketchConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_width,
        kernel_height,
        num_sketches,
        sketch_dimension,
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

        # TODO: currently sketching only occurs on the output dimension
        if sketch_dimension >= out_channels:
            raise Exception(
                f"Sketch dimension must be less than output channels, \
                    output channels: {out_channels}, sketch dimension: {sketch_dimension}"
            )

        k = math.sqrt(
            self.groups / (self.in_channels * self.kernel_width * self.kernel_height)
        )
        weight_tensor = torch.empty(
            self.out_channels,
            self.in_channels // self.groups,
            kernel_height,
            kernel_width,
        ).uniform_(-1 * k, k)

        self.bias_tensor = (
            torch.empty(out_channels).uniform_(-1 * k, k) if self.bias else None
        )
        # TODO: does reordering dimensions give better performance?
        for idx in range(self.num_sketches):
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

            prod = (
                torch.einsum(
                    "ijkl, mi -> mjkl",
                    weight_tensor,
                    self.type1_signed_matrices[idx],
                )
                .permute(2, 3, 1, 0)
                .flatten(0, 2)
            )  # (in_channels x h x w) x sketch_dim

            self.type1_sketches[idx] = prod

        weight_tensor

    def _forward_sketch_output(
        self,
        x,
        in_channels,
        kernel_height,
        kernel_width,
        dilation,
        padding,
        stride,
        num_sketches,
        bias,
        type1_sketches,
        type1_signed_matrices,
    ):
        B, _, H1, W1 = x.shape
        H2 = H1 - kernel_height + 1
        W2 = W1 - kernel_width + 1
        input_unfolded = (
            torch.nn.functional.unfold(
                x,
                (kernel_height, kernel_width),
                dilation,
                padding,
                stride,
            )
            .unsqueeze(1)
            .expand(
                B,
                num_sketches,
                kernel_height * kernel_width * in_channels,
                -1,
            )
        )  # batch x num_sketches x (in_channels * h * w) x L
        input_unfolded = torch.einsum(
            "bnfe, nfs, nso -> bnoe",
            input_unfolded,
            type1_sketches,  # bnse = batch x num_sketches x sketch_dim x L
            type1_signed_matrices,
        ).view(B, self.num_sketches, self.out_channels, H2, W2)
        input_unfolded = torch.mean(input_unfolded, dim=1) + bias[None, :, None, None]

        return input_unfolded

    def forward(self, x):
        return self._forward_sketch_output(
            x,
            self.in_channels,
            self.kernel_height,
            self.kernel_width,
            self.dilation,
            self.padding,
            self.stride,
            self.num_sketches,
            self.bias_tensor,
            self.type1_sketches,
            self.type1_signed_matrices,
        )


class ALSHSketchConv2d(SketchConv2d, ALSHConv):
    LAS = None  # static class variable to track last called layer's active set.

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_width,
        kernel_height,
        num_sketches,
        sketch_dimension,
        which_hash,
        hash_init_params,
        K,
        L,
        final_L,
        max_bits,
        bias=True,
        device="cpu",
        *,
        num_groups=1,  # TODO: Fix later to accept values != 1
        dilation=1,
        padding=0,
        stride=1,
    ):
        super().__init__(
            self,
            in_channels,
            out_channels,
            kernel_width,
            kernel_height,
            num_sketches,
            sketch_dimension,
            bias=bias,
            device=device,
            num_groups=num_groups,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        # TODO: In the original ALSH implementation, they use `out_channels`, could try in_channels since we are sketching on output dimension (diff in line 207) compared to original ALSH
        alsh_dim = self.kernel_height * self.kernel_width * self.in_channels
        self.init_ALSH(
            L,
            final_L,
            max_bits,
            which_hash,
            hash_init_params,
            K,
            alsh_dim + 2,
            out_channels,
            device=device,
        )
        self.cpu()

        # cache is used for modifying ALSH tables after an update.
        self.cache = None
        self.first = False
        self.last = False

    def use_naive(self):
        """
        when first and last are true, it fills output with zeros rather than
        sharing the last active set.
        """
        self.first = True
        self.last = True

    def reset_freq(self):
        self.bucket_stats.reset()

    def avg_bucket_freq(self):
        return self.bucket_stats.avg

    def sum_bucket_freq(self):
        return self.bucket_stats.sum

    def cuda(self, device=None):
        r"""
        moves to specified GPU device. Also sets device used for hashes.
        """
        for t in range(len(self.tables.hashes)):
            self.tables.hashes[t].a = self.tables.hashes[t].a.cuda(device)
            self.tables.hashes[t].bit_mask = self.tables.hashes[t].bit_mask.cuda(device)
        self.device = device
        return self._apply(lambda t: t.cuda(device))

    def cpu(self):
        r"""
        moves to the CPU. Also sets device used for hashes.
        """
        for t in range(len(self.tables.hashes)):
            self.tables.hashes[t].a = self.tables.hashes[t].a.cpu()
            self.tables.hashes[t].bit_mask = self.tables.hashes[t].bit_mask.cpu()
        self.device = torch.device("cpu")
        return self._apply(lambda t: t.cpu())

    def fix(self):
        r"""
        In a sketch conv, the weight Tensor is signed_matrices * sketches
         -  signed_matrices (num_sketches x
                            (in_channels * self.kernel_height * self.kernel_width) x
                            self.sketch_dim)
         -  sketches        (num_sketches x self.sketch_dim x self.out_channels)
        """
        real_weights = torch.bmm(
            self.type1_signed_matrices, self.type1_sketches
        )  # num_sketches x (in_channels * self.kernel_height * self.kernel_width) x self.out_channels
        real_weights = real_weights.permute(0, 2, 1).reshape(
            self.num_sketches,
            self.out_channels,
            self.in_channels,
            self.kernel_height,
            self.kernel_width,
        )
        # TODO: Here we have some optionality for how we want fill the table
        self.fill_table(real_weights.mean(dim=0))

    def forward(self, x):
        r"""
        Forward pass of ALSHSketchConv2d.
         -  x is a 4D tensor, i.e., a batch of images.
         -  LAS (optional) is the indices of the last active set.
            It specifies which kernels this conv
            should use.
        """

        LAS = ALSHSketchConv2d.LAS if not self.first else None

        # Assumes kernel height and width are the same?
        AS, ti = self.get_active_set(
            x, self.kernel_height, self.stride, self.padding, self.dilation, LAS
        )

        active_sketches = None
        active_signed_matrices = None
        if AS.size(0) < 2:
            active_sketches = self.type1_sketches
            active_signed_matrices = self.type1_signed_matrices
        else:
            # TODO: LAS optimization kinda sus for now because
            # I don't understand line 141 in alsh_conv_2d.py
            # if the weight matrix is (out_channels x in_channels x h x w)
            # then choosing the last active set should be over the input channels
            # (axis 1) not axis 2?
            active_sketches = self.type1_sketches[..., AS]
            active_signed_matrices = self.type1_signed_matrices.reshape(
                self.num_sketches,
                self.in_channels,
                self.kernel_height,
                self.kernel_width,
                self.sketch_dim,
            )[:, LAS, ...].flatten(1, 3)

        output = self._forward_sketch_output(
            x.to(self.device)[
                :, LAS, ...
            ],  # input channels should be based on the last LAS subset?
            self.in_channels,
            self.kernel_height,
            self.kernel_width,
            self.dilation,
            self.padding,
            self.stride,
            self.num_sketches,
            self.bias_tensor[AS],
            active_sketches,
            active_signed_matrices,
        )

        h, w = output.size()[2:]

        if self.last:
            out_dims = (x.size(0), self.out_channels, h, w)
            return zero_fill_missing(output, AS, out_dims, device=self.device)

        else:
            ALSHSketchConv2d.LAS = AS
            return output
