# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import math
import torch
from torch import nn
from deform_offset import batch_map_offsets


class OffsetConv2d(nn.Conv2d):

    def __init__(self,
                 **kwargs):
        """
        形变卷积
        kernel_size、padding等属性由父类指定
        """
        super(OffsetConv2d, self).__init__(kernel_size=(3, 3),
                                           padding=(1, 1),
                                           padding_mode='zeros',
                                           bias=True,
                                           **kwargs)
        assert not self.in_channels % self.groups
        assert self.out_channels == 2 * self.in_channels

        self.init_params()

    def forward(self, input: torch.Tensor):

        input_shape = input.size()
        offsets = super(OffsetConv2d, self).forward(input)

        offsets = self.convert_to_bc_h_w_2(offsets, input_shape)

        output = batch_map_offsets(input, offsets)

        return output

    def convert_to_bc_h_w_2(self, x, x_shape):

        # bs, c*2, h, w → bs, c, h, w, 2
        x = x.reshape(x_shape[0], -1, 2, *x_shape[2:])
        x = x.permute((0, 1, 3, 4, 2))

        return x

    def init_params(self):

        for named_param in self.named_parameters():

            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] == 'weight':
                    stddev = math.sqrt(2 / sum(param.size()[:2]))
                    torch.nn.init.normal_(param, std=stddev)
                else:
                    torch.nn.init.zeros_(param)
