from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GraphConv, DenseSAGEConv, dense_diff_pool, DenseGCNConv
from torch.nn import Linear, Dropout, PReLU, Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from node_location import convert_dis_m, get_ini_dis_m

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCNmodel(nn.Module):

    def __init__(self, xdim):
        super(GCNmodel, self).__init__()

        self.xdim = xdim

        self.A_dis = torch.FloatTensor(convert_dis_m(get_ini_dis_m(), 9)).to(device)
        self.A_dyn = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]))
        nn.init.xavier_normal_(self.A_dyn)

        self.gconv_d = DenseGCNConv(xdim[2], 32)  # (44*32, 32)
        self.gconv_s= DenseGCNConv(xdim[2], 32)  # (44*32, 32)


    def forward(self, x):

        x = x.reshape(-1, self.xdim[1], self.xdim[2])
        outd = F.relu(self.gconv_d(x, self.A_dyn))
        outs = F.relu(self.gconv_s(x, self.A_dis))
        out = torch.cat((outd, outs), dim=2)

        return out


class PDGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        drop_rate = 0.1
        topk = 10
        self.channels = 62

        self.conv1 = Conv2d(1, 32, (5, 5))
        self.drop1 = Dropout(drop_rate)
        self.pool1 = MaxPool2d((1, 4))
        self.dsgc1 = GCNmodel(xdim=[16, 62, 65 * 32])

        self.conv2 = Conv2d(32, 64, (1, 5))
        self.drop2 = Dropout(drop_rate)
        self.pool2 = MaxPool2d((1, 4))
        self.dsgc2 = GCNmodel(xdim=[16, 62, 15 * 64])

        self.conv3 = Conv2d(64, 128, (1, 5))
        self.drop3 = Dropout(drop_rate)
        self.pool3 = MaxPool2d((1, 4))
        self.dsgc3 = GCNmodel(xdim=[16, 62, 2 * 128])

        self.linend = Linear(62 * 64 * 3, 3)#62 * 64 * 3


    def forward(self, x, edge_index, batch):
        x, mask = to_dense_batch(x, batch)

        x = x.reshape(-1, 1, 5, 265)  # (Batch*channels, 1, Freq_bands, Features)

        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.pool1(x)

        x1 = self.dsgc1(x)

        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.pool2(x)

        x2 = self.dsgc2(x)

        x = F.relu(self.conv3(x))
        x = self.drop3(x)
        x = self.pool3(x)

        x3 = self.dsgc3(x)

        x = torch.cat([x1, x2, x3], dim=1)

        x = x.reshape(-1, 62 * 64 * 3)
        x = self.linend(x)
        pred = F.softmax(x, 1)

        return x, pred
