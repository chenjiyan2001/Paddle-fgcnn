# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
from paddle.nn import functional as F


class FGCNN(nn.Layer):
    def __init__(self, sparse_num_field, sparse_feature_size,
                 sparse_feature_name, sparse_feature_dim, conv_kernel_width,
                 conv_filters, new_maps, pooling_width, stride,
                 dnn_hidden_units, dnn_dropout):
        '''
        Parameters
            vocab_size - 
        '''
        super(FGCNN, self).__init__()
        self.sparse_num_field = sparse_num_field
        self.sparse_feature_size = sparse_feature_size
        self.sparse_feature_name = sparse_feature_name
        self.sparse_feature_dim = sparse_feature_dim
        self.conv_filters = conv_filters
        self.conv_kernel_width = conv_kernel_width
        self.new_maps = new_maps
        self.pooling_width = pooling_width
        self.stride = stride
 
        
        self.fg_embedding = nn.LayerList([
            EmbeddingLayer(
                num_embeddings=self.sparse_feature_size[i],
                embedding_dim=self.sparse_feature_dim,
                feature_name=self.sparse_feature_name[i] + '_fg_emd'
            ) for i in range(self.sparse_num_field)])

        self.embedding = nn.LayerList([
            EmbeddingLayer(
                num_embeddings=self.sparse_feature_size[i],
                embedding_dim=self.sparse_feature_dim,
                feature_name=self.sparse_feature_name[i] + '_emd'
            ) for i in range(self.sparse_num_field)])

        self.fgcnn = FGCNNLayer(self.sparse_num_field, self.sparse_feature_dim,
                                self.conv_filters, self.conv_kernel_width, 
                                self.new_maps, self.pooling_width, self.stride)
        
        self.combined_feture_num = self.fgcnn.new_feture_num + self.sparse_num_field
        self.dnn_input_dim = self.combined_feture_num * (self.combined_feture_num - 1) // 2\
                                + self.combined_feture_num * self.sparse_feature_dim

        self.dnn = DNNLayer(self.dnn_input_dim, dnn_hidden_units, dnn_dropout)

        self.fc_linear = self.add_sublayer(
            name='fc_linear',
            sublayer=nn.Linear(in_features=dnn_hidden_units[-1], out_features=1)
            )

    def forward(self, inputs):
        inputs = paddle.to_tensor(inputs).reshape((-1, self.sparse_num_field))
        # sparse
        fg_input_list = []
        origin_input_list = []
        for i in range(self.sparse_num_field):
            fg_input_list.append(self.fg_embedding[i](inputs[:, i].astype('int64')).reshape((-1, 1, self.sparse_feature_dim)))
            origin_input_list.append(self.embedding[i](inputs[:, i].astype('int64')).reshape((-1, 1, self.sparse_feature_dim)))
        fg_input = paddle.concat(fg_input_list, axis=1)
        origin_input = paddle.concat(origin_input_list, axis=1)
        # dense
        # fg_input = paddle.concat([fg_input, inputs[..., self.sparse_num_field:]], axis=0)
        # origin_input = paddle.concat([origin_input, inputs[..., self.sparse_num_field:]], axis=0)
        new_features = self.fgcnn(fg_input)
        combined_input = paddle.concat([origin_input, new_features], axis=1)

        # inner product
        embed_list = paddle.split(
            x=combined_input, 
            num_or_sections=combined_input.shape[1], 
            axis=1)
        row = []
        col = []
        num_inputs = len(embed_list)
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = paddle.concat([embed_list[idx] for idx in row], axis=1)  # batch num_pairs k
        q = paddle.concat([embed_list[idx] for idx in col], axis=1)

        inner_product = paddle.sum(p * q, axis=2, keepdim=True)
        inner_product = paddle.flatten(inner_product, start_axis=1)
        linear_signal = paddle.flatten(combined_input, start_axis=1)
        dnn_input = paddle.concat([linear_signal, inner_product], axis=1)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.fc_linear(dnn_output)
        y_pred = F.sigmoid(dnn_logit)
        return y_pred

class EmbeddingLayer(nn.Layer):
    def __init__(self, num_embeddings, embedding_dim, feature_name):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            name=feature_name,
            sparse=True
        )

    def forward(self, inputs):
        return self.embedding(inputs)
    
class FGCNNLayer(nn.Layer):
    def __init__(self, sparse_num_field, embedding_size, filters, kernel_width, new_maps, pooling_width, stride):
        super(FGCNNLayer, self).__init__()
        self.sparse_num_field = sparse_num_field
        self.embedding_size = embedding_size
        self.filters = filters
        self.kernel_width = kernel_width
        self.new_maps = new_maps
        self.pooling_width = pooling_width
        self.stride = stride
        self.init()

        self.conv_pooling = nn.LayerList([nn.Sequential(
                nn.Conv2D(
                    in_channels=self.in_channels_size[i], 
                    out_channels=self.filters[i], 
                    kernel_size=(self.kernel_width[i], 1), 
                    padding=(self.padding_size[i], 0),
                    stride=self.stride),
                # nn.BatchNorm2D(self.filters[i]),
                nn.Tanh(),
                nn.MaxPool2D(
                    kernel_size=(self.pooling_width[i], 1), 
                    stride=(self.pooling_width[i], 1)),
            ) for i in range(len(self.filters))])

        self.recombination = nn.LayerList([nn.Sequential(
                nn.Linear(
                    in_features=self.filters[i] * self.pooling_shape[i] * self.embedding_size,
                    out_features=self.pooling_shape[i] * self.embedding_size * self.new_maps[i],
                    name='fgcnn_linear_%d' % i),
                nn.Tanh()
            ) for i in range(len(self.filters))])

    def forward(self, inputs):
        feature = inputs.unsqueeze(1)
        new_feature_list = []

        for i in range(0, len(self.filters)):
            feature = self.conv_pooling[i](feature)
            result = self.recombination[i](paddle.flatten(feature, start_axis=1))
            new_feature_list.append(
                paddle.reshape(x=result, shape=(-1, self.pooling_shape[i] * self.new_maps[i] , self.embedding_size)))
        new_features = paddle.concat(new_feature_list, axis=1)
        return new_features

    def init(self):
        # compute pooling shape
        self.pooling_shape = []
        self.pooling_shape.append(self.sparse_num_field // self.pooling_width[0])
        for i in range(1, len(self.filters)):
            self.pooling_shape.append(self.pooling_shape[i-1] // self.pooling_width[i])
        
        # compute padding size
        self.padding_size = []
        self.padding_size.append(((self.sparse_num_field - 1) * self.stride[0] + self.kernel_width[0] - self.sparse_num_field) // 2)
        for i in range(1, len(self.filters)):
            self.padding_size.append(
                ((self.pooling_shape[i-1] - 1) * self.stride[0] + self.kernel_width[i] - self.pooling_shape[i-1]) // 2)
        
        self.in_channels_size = [1,] + list(self.filters)
        self.new_feture_num = sum([self.pooling_shape[i] * self.new_maps[i] for i in range(len(self.filters))])

class DNNLayer(nn.Layer):
    def __init__(self, inputs_dim, hidden_units, dropout_rate):
        super(DNNLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

        hidden_units = [inputs_dim] + list(hidden_units)
        self.linears = nn.LayerList([nn.Sequential(
            nn.Linear(
                in_features=hidden_units[i], 
                out_features=hidden_units[i + 1],
                weight_attr=nn.initializer.Normal(mean=0, std=1e-4),
                name='dnn_%d' % i),
            # nn.BatchNorm(hidden_units[i+1])
            ) for i in range(len(hidden_units) - 1)])
        
        self.activation_layers = nn.LayerList(
            [nn.ReLU(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

    def forward(self, inputs):
        for i in range(len(self.linears)):
            inputs = self.linears[i](inputs)
            inputs = self.activation_layers[i](inputs)
            inputs = self.dropout(inputs)
        return inputs