import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pdb


"""Implements Temporal Convolutional Network (TCN)

__https://arxiv.org/pdf/1803.01271.pdf
"""

class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size//2:-self.chomp_size//2].contiguous()
        else:
            return x[:, :, :-self.chomp_size].contiguous()
        

class ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type):
        super(ConvBatchChompRelu, self).__init__()

        self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.batchnorm = nn.BatchNorm1d(n_outputs)
        self.chomp = Chomp1d(padding,True)
        if relu_type == 'relu':
            self.non_lin = nn.ReLU()
        elif relu_type == 'prelu':
            self.non_lin = nn.PReLU(num_parameters=n_outputs)

    def forward(self, x):
        out = self.conv( x )
        out = self.batchnorm( out )
        out = self.chomp( out )
        return self.non_lin( out )


        
# --------- MULTI-BRANCH VERSION ---------------
class MultibranchTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding, dropout=0.2, 
                 relu_type = 'relu'):
        super(MultibranchTemporalBlock, self).__init__()
        
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len( kernel_sizes )
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"



        for k_idx,k in enumerate( self.kernel_sizes ):
            cbcr = ConvBatchChompRelu( n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type)
            setattr( self,'cbcr0_{}'.format(k_idx), cbcr )
        self.dropout0 = nn.Dropout(dropout)
        
        for k_idx,k in enumerate( self.kernel_sizes ):
            cbcr = ConvBatchChompRelu( n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type)
            setattr( self,'cbcr1_{}'.format(k_idx), cbcr )
        self.dropout1 = nn.Dropout(dropout)

        # downsample?
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if (n_inputs//self.num_kernels) != n_outputs else None
        
        # final relu
        if relu_type == 'relu':
            self.relu_final = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu_final = nn.PReLU(num_parameters=n_outputs)

    def forward(self, x):

        # first multi-branch set of convolutions
        outputs = []
        for k_idx in range( self.num_kernels ):
            branch_convs = getattr(self,'cbcr0_{}'.format(k_idx))
            outputs.append( branch_convs(x) )
        out0 = torch.cat(outputs, 1)
        out0 = self.dropout0( out0 )

        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range( self.num_kernels ):
            branch_convs = getattr(self,'cbcr1_{}'.format(k_idx))
            outputs.append( branch_convs(out0) )
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1( out1 )
                
        # downsample?
        res = x if self.downsample is None else self.downsample(x)

        return self.relu_final(out1 + res)

class MultibranchTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2, relu_type='relu'):
        super(MultibranchTemporalConvNet, self).__init__()

        self.ksizes = tcn_options['kernel_size']

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]


            padding = [ (s-1)*dilation_size for s in self.ksizes]            
            layers.append( MultibranchTemporalBlock( in_channels, out_channels, self.ksizes, 
                stride=1, dilation=dilation_size, padding = padding, dropout=dropout, relu_type = relu_type) )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)        
# --------------------------------


# --------------- STANDARD VERSION (SINGLE BRANCH) ------------------------
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, 
                 symm_chomp = False, no_padding = False, relu_type = 'relu'):
        super(TemporalBlock, self).__init__()
        
        self.no_padding = no_padding
        if self.no_padding:
            downsample_chomp_size = 2*padding-4
            padding = 1 # hack-ish thing so that we can use 3 layers


        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.batchnorm1 = nn.BatchNorm1d(n_outputs)
        self.chomp1 = Chomp1d(padding,symm_chomp)  if not self.no_padding else None
        if relu_type == 'relu':
            self.relu1 = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.batchnorm2 = nn.BatchNorm1d(n_outputs)
        self.chomp2 = Chomp1d(padding,symm_chomp) if not self.no_padding else None
        if relu_type == 'relu':
            self.relu2 = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu2 = nn.PReLU(num_parameters=n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
  
        if self.no_padding:
            self.net = nn.Sequential(self.conv1, self.batchnorm1, self.relu1, self.dropout1,
                                     self.conv2, self.batchnorm2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.batchnorm1, self.chomp1, self.relu1, self.dropout1,
                                     self.conv2, self.batchnorm2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        if self.no_padding:
            self.downsample_chomp = Chomp1d(downsample_chomp_size,True)
        if relu_type == 'relu':
            self.relu = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu = nn.PReLU(num_parameters=n_outputs)

    def forward(self, x):
        out = self.net(x)
        if self.no_padding:
            x = self.downsample_chomp(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2, relu_type='relu'):
        super(TemporalConvNet, self).__init__()
        self.ksize = tcn_options['kernel_size']
        symm_chomp = tcn_options['symmetric_chomp']
        no_padding = tcn_options['no_padding']

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append( TemporalBlock(in_channels, out_channels, self.ksize, stride=1, dilation=dilation_size,
                                     padding=(self.ksize-1) * dilation_size, dropout=dropout, symm_chomp = symm_chomp,
                                     no_padding = no_padding, relu_type=relu_type) )
            
        #self.tempblock0 = layers[0] # switch on for debugging a block
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
# --------------------------------
        

if __name__ == "__main__":


    import numpy as np
    import pdb

    B = 2
    xx = torch.Tensor( np.random.rand( B, 512, 25 ) ).cpu()


    test_multiscale = False
    if test_multiscale:
        nLayers = 3
        tcn_options = {'kernel_size': [3,5], 'symmetric_chomp': True, 'no_padding': False}
        mTCN = MultiscaleTCN( input_size = 512, num_channels =[256]*nLayers, num_classes = 50, tcn_options = tcn_options, dropout=0.2)
        mTCN = mTCN.cpu()
        yy = mTCN( xx.transpose(1, 2), [22,25], B )
        pdb.set_trace()



        nLayers = 3
        tcn_options = {'kernel_size': 3, 'symmetric_chomp': False, 'no_padding': True}
        tconvnet = TemporalConvNet( num_inputs = 512, num_channels = [256]*nLayers, dropout=0.2, tcn_options = tcn_options )
        tconvnet = tconvnet.cpu()

        pdb.set_trace()
        yy = tconvnet( xx )
        

        # In order to run this, uncomment line that defines tempblock0
        x1 = tconvnet.tempblock0.conv1( xx )       # 2 x 512 x 25 -> 2 x 256 x 27
        x2 = tconvnet.tempblock0.batchnorm1( x1 )  # 2 x 256 x 27
        x3 = tconvnet.tempblock0.chomp1( x2 )      # 2 x 512 x 27 -> 2 x 256 x 25


    test_multibranch = True
    if test_multibranch:
        tcn_options = {'kernel_size': [3,5,7]}
        nLayers = 4
        model = MultibranchTemporalConvNet( num_inputs = 512, num_channels =[3*256]*nLayers, 
                tcn_options = tcn_options, dropout=0.2, relu_type='prelu')

        pdb.set_trace()