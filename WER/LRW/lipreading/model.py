import torch
import torch.nn as nn
from .models.resnet1D import ResNet1D, BasicBlock1D
from .models.tcn import MultibranchTemporalConvNet


def _average_batch( x, lengths, B ):
    #if B == 1:
    #    x = x.unsqueeze(0)
    return torch.stack( [torch.mean( x[index][:,0:i], 1 ) for index, i in enumerate(lengths)],0 )
    
def _middle_pick_batch( x, lengths, B ):
    x = x.unsqueeze(0) if B == 1 else x
    return torch.stack( [ x[index][:,i//2] for index, i in enumerate(lengths) ], 0 )

class MultiscaleMultibranchTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, extract_feature=False):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options['kernel_size']
        self.num_kernels = len( self.kernel_sizes )
        self.middle_pick = tcn_options['middle_pick']

        self.mb_ms_tcn = MultibranchTemporalConvNet(input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _middle_pick_batch if self.middle_pick else _average_batch

        self.has_aux_losses = False

        self.extract_feature = extract_feature

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        xtrans = x.transpose(1, 2)

        out = self.mb_ms_tcn(xtrans)
        out = self.consensus_func( out, lengths, B )
        return out if self.extract_feature else self.tcn_output(out)

class Audio_Model(nn.Module):
    def __init__(self, hidden_dim = 256, backend_out = 512, num_classes = 500, device = None, relu_type = 'relu', gamma_zero = False, avg_pool_downsample = False, tcn_options = {}):
        super(Audio_Model, self).__init__()
        self.trunk = ResNet1D(BasicBlock1D, [2, 2, 2, 2], relu_type = relu_type, gamma_zero = gamma_zero, avg_pool_downsample = avg_pool_downsample)
        self.tcn = MultiscaleMultibranchTCN(input_size = backend_out, num_channels = [hidden_dim]*tcn_options['num_layers'],
	        num_classes = num_classes, tcn_options = tcn_options, dropout = tcn_options['dropout'], relu_type = relu_type)

    def forward(self, x, lengths):
        B, C, T = x.size()
        x = self.trunk(x)
        x = x.transpose(1, 2)
        lengths = [_//640 for _ in lengths]
        return self.tcn(x, lengths, B)
