import torch
import torch.nn as nn


#TODO: try faiss IndxLSH for Table


class SRPTable:
    def __init__(self, num_hashes, output_dim):
        """
        num_hashes is the number of hashes to concatenate
        2015 paper goes over this in detail, https://arxiv.org/pdf/1410.5410.pdf
        """
        self.bits = num_hashes
        self.dim = output_dim
        self.random_tensor = torch.randn(self.dim, self.bits) # (in_channels * kernel_size * kernel_size + 2, K)
        self.normal = self.random_tensor[:-2]
        self.bit_mask = torch.Tensor([2 ** (i) for i in torch.arange(self.bits)])

    def _preprocess_rows(self, x):
        """
        assuming m = 2 simplifies the problem a lot. And it's not longer
        necessary for it to be any larger than that since Q_obj doesn't
        append 0's, it just uses a splice of self.random_tensor.

        - x should be a matrix where rows are the datum to be inserted.
        """

        norm = x.norm(dim=1)  # norm of each row
        norm /= norm.max() / 0.75
        norm.unsqueeze_(1)
        app1 = 0.5 - (norm**2)
        app2 = 0.5 - (norm**4)
        return torch.cat((x, app1, app2), 1)

    def hash_matr(self, matr):
        """
        Applies SRP hash to the rows a matrix.
        """
        # N x num_bits
        bits = (torch.mm(matr, self.random_tensor.to(matr)) > 0).float()
        return (bits * self.bit_mask.to(matr)).sum(1).view(-1).long()

    def hash_4d_tensor(self, obj, kernel_size, stride, padding, dilation, LAS=None):
        # set normal=a[:-2] to prevent appending / avoid copying
        normal = (
            self.normal.transpose(0, 1)
            .view(self.bits, -1, kernel_size, kernel_size)
            .to(obj)
        )
        
        if LAS is not None:
            normal = normal[:, LAS]

        out = torch.nn.functional.conv2d(
            obj, normal, stride=stride, padding=padding, dilation=dilation
        )

        # convert output of conv into bits
        trs = out.view(out.size(0), self.bits, -1).transpose(1, 2)
        bits = (trs >= 0).float()

        # return bits -> integer hash
        return (bits * self.bit_mask.to(obj)).sum(2)

    def query(self, input, **kwargs):
        """
        applies Q to input and hashes.
        If input object has dim == 4, kwargs should contaion stride,
        padding, and dilation
        """
        assert (
            input.dim() == 4
        ), "MultiHash_SRP.query. Input must be dim 4 but got: " + str(input.dim())
        return self.hash_4d_tensor(input, **kwargs)

    def preprocess(self, input):
        assert input.dim() == 2, "MultiHash_SRP.pre. Input must be dim 2."
        return self.hash_matr(self._preprocess_rows(input))


class AlshConv2d(torch.nn.Conv2d):
    LAS=None

    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 bias,
                 is_first_layer,
                 is_last_layer,
                 num_hashes, # num hash funcs per table, K
                 num_tables, # num of tables, L
                #  final_num_tables, # max num tables at the end of computation
                 max_bits, # max bits per hash, ie table size, we can have more hash funcs than max bits and mod output
                 hash_table=SRPTable, # table with certain type of hash func
                 device="cpu"
                 ): 
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         bias=bias)
        
        # Check device
        assert isinstance(device, str), "ALSHConv, device must be a string"
        assert device in [
            "cpu",
            "cuda",
            "mps",
        ], f"ALSHConv, device must be in {['cpu', 'cuda', 'mps']}."

        # Initialize variables
        self.alsh_dim = in_channels * kernel_size * kernel_size
        self.num_filters = out_channels
        self.tables_dim = self.alsh_dim + 2 #TODO Why?
        self.num_hashes = num_hashes
        self.num_tables = num_tables
        self.max_bits = max_bits
        # self.final_num_tables = final_num_tables

        # Pre-pass: create hash tables
        # create L hash tables
        self.hashes = [hash_table(num_hashes, self.tables_dim) for _ in range(num_tables)]
        self.tables = [[[] for _ in range(max_bits)] for _ in range(num_tables)]  # tables does not contain keys, only values. TODO: is there better way to store this?

        # Other bookkeeping
        self.cpu()
        self.cache = None # cache is used for modifying ALSH tables after an update
        self.first_layer = is_first_layer
        self.last_layer = is_last_layer
        self.fix()


    def forward(self, imgs):

        LAS = AlshConv2d.LAS if not self.first_layer else None

        # Get active set
        AS = self.get_active_set(imgs, self.kernel_size[0], self.stride, self.padding, self.dilation, LAS)

        # If active set is very small then just use weight for active kernels
        if AS.size(0) < 2:
            AK = self.weight
        else:
            if self.first_layer:
                # if its the first ALSHConv2d in the network,
                # then there is no LAS to use!
                AK = self.weight[AS]
            else:
                AK = self.weight[AS][:, AlshConv2d.LAS]

        output = nn.functional.conv2d(
            imgs,
            AK,
            bias=self.bias[AS],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
            
        h, w = output.size()[2:]
        if self.last_layer:
            out_dims = (imgs.size(0), self.out_channels, h, w)
            return self.zero_fill_missing(output, AS, out_dims)
        else:
            AlshConv2d.LAS = AS
            return output

    def fix(self):
        num_filters = self.weight.size(0)
        self.insert_to_tables(self.weight.view(num_filters, -1), torch.arange(0, num_filters).long())

    # Core functions
    def insert_to_tables(self, keys, vals):
        """
        inserts a sequence of values into tables based on keys.
        keys[i] is the key for values[i]
        For this application, only values need to be stored in the hash table.
        They work as references to the keys.
        """
        self.tables = [[[] for _ in range(self.max_bits)] for _ in range(self.num_tables)]
        rows = [
            hash.preprocess(keys) % self.max_bits for hash in self.hashes
        ]

        # can parallelize outer loop for each table?
        for table_i in torch.arange(0, self.num_tables).long():
            for row, val in zip(rows[table_i], vals):
                self.tables[table_i][row].append(int(val)) # in i_th table, put val at preprocessed key

    def get_from_tables(self, key, **kwargs):
        # we can have more hash funcs than max bits and mod output
        return torch.stack([
            hash.query(key, **kwargs) % self.max_bits for hash in self.hashes
        ])

    def get_active_set(self, input, kernel_size, stride, padding, dilation, LAS=None):
        table_i = self.get_from_tables(
            input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            LAS=LAS
        )
        table_i = table_i.view(self.num_tables, -1)

        k = 5
        top_k = torch.zeros(table_i.size(0), k).long() # top k elements for each row
        for row in range(table_i.size(0)):
            top_k[row] = self.most_freq(table_i[row], k=k)

        AS = torch.LongTensor([])
        for i in torch.arange(0, self.num_tables).long():
            for j in top_k[i]:
                ids = torch.LongTensor(self.tables[i][j])
                AS = torch.cat((AS, ids)).unique(sorted=False)
        return AS.sort()[0]

    ## Utils
    def most_freq(self, x, k):
        """
        finds the k most frequently occuring values in x
        """
        bins = self.max_bits
        item_freq = torch.histc(x.cpu(), bins=bins, max=bins)
        # self.bucket_stats.update(item_freq)
        return item_freq.topk(k)[1]

    def zero_fill_missing(self, x, i, dims):
        """
        fills channels that weren't computed with zeros.
        """
        t = torch.empty(dims).to(x).fill_(0)
        t[:, i, :, :] = x[:,]
        return t

    ## Change device for tables
    def cuda(self, device=None):
        """
        moves to specified GPU device. Also sets device used for hashes.
        """
        for t in range(len(self.hashes)):
            self.hashes[t].random_tensor = self.hashes[t].random_tensor.cuda(device)
            self.hashes[t].bit_mask = self.hashes[t].bit_mask.cuda(device)
        self.device = device
        return self._apply(lambda t: t.cuda(device))

    def cpu(self):
        """
        moves to the CPU. Also sets device used for hashes.
        """
        for t in range(len(self.hashes)):
            self.hashes[t].random_tensor = self.hashes[t].random_tensor.cpu()
            self.hashes[t].bit_mask = self.hashes[t].bit_mask.cpu()
        self.device = torch.device("cpu")
        return self._apply(lambda t: t.cpu())


    if __name__=='__main__':
        pass