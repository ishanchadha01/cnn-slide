import torch
import torch.autograd.function as Function


#TODO: try faiss IndxLSH


class SignedRandomProjection(Function): #TODO maybe dont implement as function so it can more easily store state
    def __init__(self):
        pass


class HashTable:
    def __init__(self, hash_func):
        self.hash_func = hash_func
        self.hash_func_arr = []
        self._table = {}
        #TODO implement as just having num_funcs, hash_func, 
        # and element wise mapping of func to each element

    def create_hash_functions(self, num_funcs):
        self.hash_func_arr = num_funcs * [self.hash_func()]

    def apply_hash_function(self, x):
        for func in self.hash_func_arr:
            x = func(x)
        return x

    def __setitem__(self, index, val):
        self._table[index] = val

    def __getitem__(self, index):
        return self._table[index]



class AlshConv2d(torch.nn.Conv2d):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 bias,
                 num_funcs, # max bits, num hash funcs per table, K
                 num_tables, # num of tables, L
                 hash_func, 
                 filters):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         bias=bias)
        
        # Initialize variables
        self.num_filters = out_channels

        # Pre-pass: create hash tables
        # create L hash tables 
        #TODO this is initialized differently
        self.tables = num_tables * [HashTable(hash_func)]
        for table in self.tables:
            table.create_hash_functions(num_funcs)
        
        # transform filters into rows, assuming theyre passed in group of filters.shape[0] filters
        filters_flattened = filters.reshape((filters.shape[0], -1, kernel_size * kernel_size))

        # initialize table
        for table in self.tables:
            for idx, filter in enumerate(filters_flattened):
                entry = self.hash_func(self._preprocess(filter))
                table[entry] = idx

        # Other bookkeeping
        self.cpu()
        self.cache = None # cache is used for modifying ALSH tables after an update
        self.first_layer = False
        self.last_layer = False


    def forward(self, imgs, filters, query_func):
        num_imgs, num_channels, img_height, img_width = imgs.shape
        num_filters = filters.shape[0]
        kernel_size = filter.shape[2]

        imgs_columnized = img2col(imgs) #TODO implement img2col or modify to do without
        filters_flattened = filters.reshape(num_filters, -1, kernel_size * kernel_size)
        filters_to_use = [] # indices of filters to use

        #TODO does not currently work in batches!
        # get filters that are most relevant to whichever hash the img falls into
        for idx, table in enumerate(self.tables):
            hash_count = torch.zeros((len(table.hash_func_arr),))
            for col in imgs_columnized:
                col_hash = table.apply_hash_func(col)
                hash_count[col_hash] += 1
            
            hash = torch.argmax(hash_count) # most frequent hash
            filters_for_hash = table[idx][hash]
            filters_to_use += filters_for_hash

        filters_to_use = torch.Tensor(filters_to_use) # TODO use torch tensor for full computation
        active_set = filters_flattened[filters_to_use] # get filters indexed by filters to use by masking

        # TODO maybe use einsum
        # m * (k*k) x n * (h*w) = m*n*h*w
        output = torch.einsum('mK,nhw->nmhw', active_set, imgs_columnized)
        #TODO reshape into batched output
        return output
    
    def backward(self, grad):
        super(AlshConv2d, self).backward(grad)
        # return grad

    def _preprocess(self, filter):
        # preprocess filter before it can be passed into the hash function
        pass


    if __name__=='__main__':
        pass