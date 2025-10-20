import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import random
import clip
import utils
from collections import OrderedDict
import time
Houston_Input_Dimension = 144
IP_Input_Dimension = 200

PU_Input_Dimension = 103
SA_Input_Dimension = 204
PC_Input_Dimension = 102
Source_Input_Dimension = 128
Output_Dimension = 128

FEATURE_DIM = 128
SRC_INPUT_DIMENSION = 128
TAR_INPUT_DIMENSION = 103
N_DIMENSION = 100
CLASS_NUM = 9
SHOT_NUM_PER_CLASS = 1
QUERY_NUM_PER_CLASS = 19
EPISODE = 20000
TEST_EPISODE = 600
LEARNING_RATE = 0.001
GPU = 0
HIDDEN_UNIT = 10
# 从预训练模型获得参数：预训练模型的尺寸信息
pretrained_dict  = torch.load('/data/Student/LHY/DCFSL-2021-main/ViT-B-32.pt', map_location="cpu").state_dict()
# embed_dim=512 -> 投影到160 -> 投影到128
embed_dim = 128 # pretrained_dict ["text_projection"].shape[1]
# context_length=77
context_length = 77

# vocab_size=49408
vocab_size = pretrained_dict ["token_embedding.weight"].shape[0]
# width=512
transformer_width = pretrained_dict ["ln_final.weight"].shape[0]
# transformer_heads=8
transformer_heads = transformer_width // 64
transformer_layers = 3
utils.same_seeds(0)

class Mapping(nn.Module):
    def __init__(self, in_features, out_features):
        super(Mapping, self).__init__()
        self.map = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0)
        self.map_bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.map(x)
        x = self.map_bn(x)
        return x

#######文本model######
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16.
    对 torch 中的 LayerNorm 进行子类化以处理 fp16 数据
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


# 提供一个近似的GELU激活函数
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        # 建立多个神经网络层组成的序列 先是一个线性层，然后是一个激活函数gelu，最后说一个线性层
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    # 声明x的预期类型为torch.tenor 并且将attn_mask也转换为这种类型
    def attention(self, x: torch.Tensor):
        # 先检查mask是否为none
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # 返回自注意力机制计算后的输出张量，不输出注意力权重，获取结果的第一个元素
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    # 默认mask为none
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        # sequential是torch的一个容器，按顺序组合乙烯类的神经网络层或模块，每个层的输出会作为下个层的输入；
        # *[xxxx for x in xxx]创建了一个包含layer（3）个res对象的列表，并将其作为seq的输入
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)



class MultiHeadAttention(nn.Module):
    def __init__(self, dimension, num_heads=8, dropout_rate=0.):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout_rate
        self.dimension = dimension
        self.layer_Q = nn.Linear(dimension, dimension, bias=False)
        self.layer_K = nn.Linear(dimension, dimension, bias=False)
        self.layer_V = nn.Linear(dimension, dimension, bias=False)
        self.restore = nn.Linear(dimension, dimension, bias=False)
        self.ln = torch.nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, queries, keys, values):
        batch_size = queries.shape[0]
        dim_prehead = self.dimension//self.num_heads
        Q = self.layer_Q(queries).view(batch_size*self.num_heads, -1, dim_prehead)
        K = self.layer_K(keys).view(batch_size*self.num_heads, -1, dim_prehead)
        V = self.layer_V(values).view(batch_size*self.num_heads, -1, dim_prehead)

        # Q_, K_, V_ 计算Attention
        outputs = scaled_dot_product_attention(Q, K, V)

        # Restore shape
        outputs = outputs.view(batch_size, -1, dim_prehead*self.num_heads)
        outputs = self.restore(outputs)

        outputs = self.dropout(outputs)
        # Residual connection
        outputs += queries
        # Normalize (N, T_q, d_model)
        outputs = self.ln(outputs)  # 这个ln还有待争议
        return outputs


def scaled_dot_product_attention(Q, K, V):
    # 计算Q,K相似度。
    # Q: (h*N, T_q, d_model/h)  V: (h*N, T_k, d_model/h),其中 T_q == T_k
    # tf.transpose,高维度矩阵转置，输出维度:(h*N, d_model/h,T_k)
    # tf.matmul，最后两维度做矩阵乘法，所以最后维度为：
    # (h*N, T_q, T_k)
    d_k = Q.shape[2]*8
    outputs = torch.matmul(Q, torch.transpose(K, 2, 1))
    # scale，同样，对值scale有点不清楚为啥
    outputs /= d_k ** 0.5
    # softmax，数值转化为概率
    outputs = F.softmax(outputs, dim=-1)
    # # dropout
    # outputs = F.dropout(outputs, p=dropout_rate)

    # weighted sum (context vectors)
    outputs = torch.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.fc(x)
        output = self.ln(residual + output)
        return output


class TransformBlock(nn.Module):
    def __init__(self, dimension, num_heads=8, dropout_rate=0.):
        super(TransformBlock, self).__init__()
        self.MultiHeadAttention = MultiHeadAttention(dimension=dimension, num_heads=num_heads,
                                                     dropout_rate=dropout_rate)
        self.FFN = FeedForward(dimension, dimension * 4)

    def forward(self, queries, keys, values):
        output = self.MultiHeadAttention(queries, keys, values)
        output = self.FFN(output)
        return output


class Embedding(nn.Module):
    def __init__(self, dimension):
        super(Embedding, self).__init__()
        # self.source_cls_token = nn.Parameter(torch.randn(1, 1, dimension))
        # self.source_position = nn.Parameter(torch.randn(81, dimension))
        # self.target_cls_token = nn.Parameter(torch.randn(1, 1, dimension))
        self.target_position = nn.Parameter(torch.randn(81, dimension))

    def forward(self, x):
        batch = x.shape[0]
        x += self.target_position
        return x




# class Spatial_adaptation(nn.Module):
#     def __init__(self):
#         super(Spatial_adaptation, self).__init__()
#         # 初始化可学习权重系数
#         self.w = nn.Parameter(torch.ones(4))
#
#     def forward(self, x):
#         [nColumn, nBand, nRow] = x.shape
#         x1_start = (nRow - 3 ** 2) // 2
#         b1 = x[:, :,x1_start:x1_start + 3 ** 2]
#         x2_start = (nRow - 5 ** 2) // 2
#         b2 = x[:, :,x2_start:x2_start + 5 ** 2]
#         x3_start = (nRow - 7 ** 2) // 2
#         b3 = x[:, :,x3_start:x3_start + 7 ** 2]
#         b4 = x
#
#         # 归一化权重
#         w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
#         w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
#         w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
#         w4 = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))
#
#         # 多特征融合
#         feature = b1 * w1 + b2 * w2 + b3 * w3 + b4 * w4
#         return feature

# 特征提取
class TransformNet(nn.Module):
    def __init__(self, output_unit, num_head=8, dropout_rate=0.1):
        super(TransformNet, self).__init__()
        self.source_mapping = Mapping(Source_Input_Dimension, Output_Dimension)
        # self.target_mapping = Mapping(IP_Input_Dimension, Output_Dimension)
        self.target_mapping = Mapping(Houston_Input_Dimension, Output_Dimension)
        # self.spatial_adaptation = SpatialAdaptation()
        self.transform1 = TransformBlock(dimension=Output_Dimension, num_heads=num_head, dropout_rate=dropout_rate)

        self.classifier_target = nn.Sequential(
            nn.Linear(25*Output_Dimension, output_unit),
        )
        self.classifier_target_text = nn.Sequential(
            nn.Linear(25 * Output_Dimension, output_unit),
        )
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.final_feat_dim = FEATURE_DIM
        # 将torch的embeding实例化 第一个参数：多少个字，一个词用多少个数字代表
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.linear_layer = nn.Linear(77,25)
        self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM)
        self.classifier_text = nn.Linear(in_features=FEATURE_DIM, out_features=CLASS_NUM, bias=False)
        self.initialize_parameters()

    @property
    def dtype(self):
        return self.transform1.MultiHeadAttention.layer_V.weight.dtype

    def initialize_parameters(self):
        # 正态分布初始化embeding，和positional，std为所设标准差
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def classifier(self, text):
        return self.classifier_text(text)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # 归一化
        x = self.ln_final(x).type(self.dtype)
        # 对第二维度的文本特征投影到128
        x = x @ self.text_projection
        # 将第一维度的词密度77映射至25
        x = self.linear_layer(x.transpose(1,2)).transpose(1,2)
        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, x, text_data = None, text_queue = None, domain='source'):
        # 计算特征矢量
        if domain == "source":
            x = self.source_mapping(x)
        elif domain == "target":
            x = self.target_mapping(x)
        # x = self.spatial_adaptation(x)
        x = torch.transpose(x, -1, -2)
        # (x_query, x_key) = (x[:, 25:49, :], x[:, 49:, :])
        # x_out = self.transform1(x_query, x_key, x_key)
        # (x_query, x_key) = (x[:, 9:25, :], x_out)
        # x_out = self.transform2(x_query, x_key, x_key)
        # (x_query, x_key) = (x[:, 1:9, :], x_out)
        # x_out = self.transform3(x_query, x_key, x_key)
        # (x_query, x_key) = (x[:, 0:1, :], x_out)
        # x = self.transform4(x_query, x_key, x_key)

        (x_query, x_key) = (x[:, 0:25, :], x[:, 25:, :])
        x = self.transform1(x_query, x_key, x_key)
        # x = self.transform2(x, x, x)
        # x = self.transform2(x, x, x)
        # x = self.transform3(x, x, x)
        # x = self.transform4(x, x, x)

        feature = x.view(x.shape[0], -1)
        # 分类
        # if domain == "source":
        #     output = self.classifier_source(feature)
        # if domain == "target":
        #     output = self.classifier_target(feature)
        output = self.classifier_target(feature)


        if text_data == None and text_queue == None :
            return feature, output
        elif text_data != None and text_queue == None :
            feature_text = self.encode_text(text_data)
            feature_text = feature_text.reshape(feature_text.shape[0], -1)
            cls = self.classifier_target_text(feature_text)
            return feature, output, feature_text, cls
        elif text_data == None and text_queue != None:
            feature_queue = self.encode_text(text_queue)
            cls = self.classifier(feature_queue)
            return feature, output, feature_queue ,cls
        else:
            feature_text = self.encode_text(text_data)
            feature_queue = self.encode_text(text_queue)
            feature_text = feature_text.reshape(feature_text.shape[0], -1)
            feature_queue = feature_queue.reshape(feature_queue.shape[0], -1)
            cls = self.classifier_target_text(feature_text)
            return feature, output, feature_text, cls, feature_queue


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits

def fuzzy_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)

    # 下面是使用高斯隶属度函数进行计算
    sig = 4  # 这个值越大，不同类别之间的隶属度就越接近，建议在10 ~ 20之间
    Dis = torch.pow(a - b, 2).sum(2)  #这个就是方差的头部，再除以数目
    fuzzy_degree = torch.exp(-Dis / (2 * sig ** 2))

    # logits = -((a - b)**2).sum(dim=2)
    return fuzzy_degree

def cube_to_list(cube):
    # spatial_adaptation = SpatialAdaptation()
    # cube = spatial_adaptation(cube)
    size = cube.shape[2]
    rope_all = np.zeros([cube.shape[0], cube.shape[1], cube.shape[2] * cube.shape[3]], dtype=np.float32)
    rope_all[:, :, 0] = cube[:, :, size // 2, size // 2]
    for i in range(1, size // 2 + 1):
        a = size // 2
        max = i * 2 + 1
        center = cube[:, :, a - i:size + i - a, a - i:size + i - a]
        rope = np.zeros([center.shape[0], center.shape[1], center.shape[2] ** 2 - (center.shape[2] - 2) ** 2], dtype=np.float32)
        rope[:, :, 0:max] = center[:, :, 0, :]
        rope[:, :, max - 1:max * 2 - 1] = center[:, :, :, max - 1]
        rope[:, :, max * 2 - 2:max * 3 - 2] = center[:, :, max - 1, ::-1]
        rope[:, :, max * 3 - 3:max * 4 - 3] = center[:, :, 1:, 0][:, :, ::-1]
        rope_all[:, :, (max - 2) ** 2:max ** 2] = rope[:, :, :]
    return rope_all

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1

class DomainClassifier(nn.Module):
    def __init__(self):  # torch.Size([1, 64, 7, 3, 3])
        super(DomainClassifier, self).__init__()  #
        self.layer = nn.Sequential(
            nn.Linear(1024, 1024),  # nn.Linear(320, 512), nn.Linear(FEATURE_DIM*CLASS_NUM, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

        )
        self.domain = nn.Linear(1024, 1)  # 512

    def forward(self, x, iter_num):
        coeff = calc_coeff(iter_num, 1.0, 0.0, 10, 10000.0)
        x.register_hook(grl_hook(coeff))

        x = self.layer(x)
        domain_y = self.domain(x)
        return domain_y
