import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

from torch_scatter import scatter_mean
from models.kmeans import *
from models.gcnlayers import *
from torchsummary import summary

import functools
from einops import rearrange

import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d, MLPLayer

import cv2


import os
import matplotlib.pyplot as plt
hotmappath = '/autodl-fs/data/CD_Dataset/test/hotmap'

if not os.path.exists(hotmappath):
    os.makedirs(hotmappath)

def h(net, name):
    batch = net.shape[0]
    for b in range(0, batch, 1):
        print(b)
        net_copy = net[b]
        # hot = []
        # net_copy.swapaxes(0, 2)
        net_copy = rearrange(net_copy, 'c w h ->  w h c')
        image_avg = np.zeros((net_copy.shape[0], net_copy.shape[1]), dtype=np.uint8)
        # image_1 = np.zeros((net_copy.shape[0], net_copy.shape[1]), dtype=np.uint8)
        hot = np.zeros((net_copy.shape[0], net_copy.shape[1]), dtype=np.uint8)

        # print(net_copy.shape[0])
        for k in range(net_copy.shape[2]):
            image = np.zeros((net_copy.shape[0], net_copy.shape[1]), dtype=np.float64)
            for j in range(net_copy.shape[1]):
                for i in range(net_copy.shape[0]):
                    image[i, j] = net_copy[i, j, k]
            image_avg = image + image_avg
        image_avg = image_avg / 32

        image_1 = image_avg
        cv2.normalize(image_1, image_avg, 0, 255, cv2.NORM_MINMAX)
        image_2 = image_avg.astype(np.uint8)
        image = cv2.applyColorMap(image_2, cv2.COLORMAP_JET)
        cv2.imwrite("/autodl-fs/data/CD_Dataset/test/hotmap/" + name + "_"  + str(b) + ".jpg", image)
        # hot.append(image)
    return 0




###############################################################################
# Helper Functions
###############################################################################
def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0- epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    elif args.lr_policy == 'reduce':

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]): # G: generator
    if args.net_G == 'base_resnet18':
        net = ResNet(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'base_transformer_pos_s4':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned')

    elif args.net_G == 'base_transformer_pos_s4_dd8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8)

    elif args.net_G == 'base_transformer_pos_s4_dd8_dedim8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)

    elif args.net_G == 'ReViT':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos='learned', with_decoder_pos='learned', enc_depth=4, dec_depth=8, decoder_dim_head=8)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnext18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()

        expand = 1
        if backbone == 'resnet18':
            # self.resnet = models.resnet18(pretrained=True,
            #                               replace_stride_with_dilation=[False,True,True])
            self.resnet = models.resnet18()
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        elif backbone == 'resnext18':
            self.resnet = models.resnext18(pretrained=False, replace_stride_with_dilation=[False,True,True])

        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc) # output2

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError

        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

        self.output_nc = output_nc
        self.last_conv = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0),
        )
        self.last_conv2 = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0),
        )
        self.last_conv3 = nn.Sequential(
            nn.Conv2d(128, 8, kernel_size=1, stride=1, padding=0),
        )
        self.last_conv4 = nn.Sequential(
            nn.Conv2d(256, 8, kernel_size=1, stride=1, padding=0),
        )



    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        # print('original image size', x.size())
        x = self.resnet.conv1(x) #
        x = self.resnet.bn1(x)   # 8 64 128 128
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) # 8 64 64 64
        low = x
        # print('conv1 x', x.size())

        # x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64 # 8 64 64 64
        x2 = self.resnet.layer1(x)
        # print('layer1 x2', x2.size())

        #x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128 # 8 128 32 32
        x3 = self.resnet.layer2(x2)
        # print('layer2 x3', x3.size())

        if self.resnet_stages_num > 3:
            # x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256 # 8 256 32 32
            x4 = self.resnet.layer3(x3)
            # print('layer3 x4', x4.size())
        if self.resnet_stages_num == 5:
            # x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
            x = self.resnet.layer4(x4)
            # print('layer4 x', x.size())
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        # if self.if_upsample_2x:
        #     x = self.upsamplex2(x_8)
        # else:
        #     x = x_8
        # print(x3)
        # output layers
        # print('before', 'x', x.size(), 'x3', x3.size(), 'x4', x4.size(), 'x2', x2.size())
        x = F.interpolate(x, size=low.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=low.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=low.size()[2:], mode='bilinear', align_corners=True)
        # print('after interpolate', 'x', x.size(), 'x3', x3.size(), 'x4', x4.size(), 'x2', x2.size())
        # print(x3)

        x = self.last_conv(x)
        x2, x3, x4 = self.last_conv2(x2), self.last_conv3(x3), self.last_conv4(x4)
        # print('x2', x2.size(), 'x3', x3.size(), 'x4', x4.size(), 'x', x.size())

        # print('final', torch.cat([x2,x3,x4,x], dim=1).size())


        # feature_map_data = torch.cat((x2,x3,x4,x), dim=1)[0, 0].cpu().detach().numpy()
        # plt.imshow(feature_map_data, cmap="viridis")
        # plt.savefig(hotmappath, dpi=100)
        # plt.close()

        #x = self.conv_pred(x) # self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1) # 8 32 64 64

        return torch.cat((x2,x3,x4,x), dim=1)

class BASE_Transformer(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnext18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True,
                 k_nums=1000, clusters=10):
        super(BASE_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.k = k_nums
        self.cluster_nums = clusters
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenizer, then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim
        self.cluster_nums = 10
        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.cluster_nums*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)
        self.mlp_layer = MLPLayer(dim=dim, mlp_dim=mlp_dim, dropout=0)
        self.gc1 = GraphConvolution(in_features=32, out_features=1)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        # print('sa before view', spatial_attention.size())
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        # print('sa after view', spatial_attention.size())
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        # print('sa after softmax', spatial_attention.size())
        # print('x before view', x.size())
        x = x.view([b, c, -1]).contiguous()
        # print('x after view', x.size())
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        # print('token', tokens.size())
        return tokens

    def _forward_tokens(self, x, index):
        b, c, h, w = x.shape

        x = x.reshape(b, c, -1)  # x1/2:  b  c   hw

        select_k_x = torch.gather(x, 2, index.repeat(1, c, 1))  # torch.Size([8, c, k])
        # tokens = select_k_x.mean(2, keepdim=True).transpose(1, 2)  # torch.Size([8, 1, 32])
        tokens = select_k_x.transpose(1, 2)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x, x_res= self.transformer(x)
        # x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        # x = self.transformer_decoder(x, m)
        x, x_res = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def kmeansToken(self, x, num_clusters):
        cluster_ids_x, _ = kmeans(X=x.detach(), num_clusters=num_clusters, distance='euclidean')
        c = scatter_mean(x, cluster_ids_x.squeeze(), dim=1, dim_size=num_clusters)
        return c



    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        # draw_features(64,64,x2.cpu().detach().numpy(), "{}/resnet.png".format(hotmappath))
        # h(x1, 'x1')
        # h(x2, 'x2')

        # forward tokenzier
        token1 = self._forward_semantic_tokens(x1)
        token2 = self._forward_semantic_tokens(x2)

        # print('a', token1.size())
        token1 = self.kmeansToken(token1, self.cluster_nums)
        # print('b', token1.size())
        token2 = self.kmeansToken(token2, self.cluster_nums)

        # forward transformer encoder
        self.tokens_ = torch.cat([token1, token2], dim=1)
        self.tokens_ = self.mlp_layer(self.tokens_)
        self.tokens = self._forward_transformer(self.tokens_)
        token1, token2 = self.tokens.chunk(2, dim=1)

        # forward transformer decoder
        x1 = self._forward_transformer_decoder(x1, token1)
        x2 = self._forward_transformer_decoder(x2, token2)
        # h(x1, 'x11')
        # h(x2, 'x22')

        # feature differencing
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # heatmap = cv2.applyColorMap(x.detach().cpu().numpy()[0].astype(np.uint8), cv2.COLORMAP_JET)
        # cv2.imwrite(hotmappath, heatmap)
        # h(x, 'final')
        # print(x.size()) # 8 32 256 256
        # forward small cnn
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

# net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
#                                with_pos='learned', with_decoder_pos = 'learned', enc_depth=4, dec_depth=8, decoder_dim_head=8)
# # print(net)
# device = torch.device("cuda")
# model = net.to(device)
# summary(
#     model,
#     input_size = [(3, 256, 256), (3, 256, 256)],
#     batch_size = 8
# )