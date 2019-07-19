import torch.nn as nn
import torch
from torch.autograd import Variable
import math, copy
import torch.nn.functional as F
import numpy as np





class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, ft_size, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = 22
        self.time_len = time_len

        self.domain = domain



        if domain == "temporal" or domain == "mask_t":
            #temporal positial embedding
            pos_list = list(range(self.joint_num * self.time_len))


        elif domain == "spatial" or domain == "mask_s":
            # spatial positial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)


        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.time_len * self.joint_num, ft_size)
        #position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, ft_size, 2).float() *
                             -(math.log(10000.0) / ft_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).cuda()
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, ft_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(ft_dim))
        self.b_2 = nn.Parameter(torch.zeros(ft_dim))
        self.eps = eps

    def forward(self, x):
        #[batch, time, ft_dim)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class MultiHeadedAttention(nn.Module):
    def __init__(self, h_num, h_dim, input_dim, dp_rate,domain):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        #assert d_model % h == 0
        # We assume d_v always equals d_k
        self.h_dim = h_dim # head dimension
        self.h_num = h_num #head num
        self.attn = None #calculate_att weight
        #self.att_ft_dropout = nn.Dropout(p=dp_rate)
        self.domain = domain  # spatial of  tempoal

        self.register_buffer('t_mask', self.get_domain_mask()[0])
        self.register_buffer('s_mask', self.get_domain_mask()[1])



        self.key_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            nn.Dropout(dp_rate),
                            )


        self.query_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            nn.Dropout(dp_rate),
                            )


        self.value_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            nn.ReLU(),
                            nn.Dropout(dp_rate),
                                     )

    def get_domain_mask(self):
        # Sec 3.4
        time_len = 8
        joint_num = 22
        t_mask = torch.ones(time_len * joint_num, time_len * joint_num)
        filted_area = torch.zeros(joint_num, joint_num)

        for i in range(time_len):
            row_begin = i * joint_num
            column_begin = row_begin
            row_num = joint_num
            column_num = row_num

            t_mask[row_begin: row_begin + row_num, column_begin: column_begin + column_num] *= filted_area #Sec 3.4


        I = torch.eye(time_len * joint_num)
        s_mask = Variable((1 - t_mask)).cuda()
        t_mask = Variable(t_mask + I).cuda()
        return t_mask, s_mask



    def attention(self,query, key, value):
        "Compute 'Scaled Dot Product Attention'"
        # [batch, time, ft_dim)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if self.domain is not None:
            #section 3.4 spatial temporal mask operation
            if self.domain == "temporal":
                scores *= self.t_mask  # set weight to 0 to block gradient
                scores += (1 - self.t_mask) * (-9e15)  # set weight to -inf to remove effect in Softmax
            elif self.domain == "spatial":
                scores *= self.s_mask  # set weight to 0 to block gradient
                scores += (1 - self.s_mask) * (-9e15)  # set weight to -inf to remove effect in Softmax

        # apply weight_mask to bolck information passage between ineer-joint

        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, x):
        "Implements Figure 2"
        nbatches = x.size(0) # [batch, t, dim]
        # 1) Do all the linear projections in batch from d_model => h x d_k

        query = self.query_map(x).view(nbatches, -1, self.h_num, self.h_dim).transpose(1, 2)
        key = self.key_map(x).view(nbatches, -1, self.h_num, self.h_dim).transpose(1, 2)
        value = self.value_map(x).view(nbatches, -1, self.h_num, self.h_dim).transpose(1, 2)


        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value) #[batch, h_num, T, h_dim ]

            # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h_dim * self.h_num)#[batch, T, h_dim * h_num ]


        return x





class ST_ATT_Layer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, input_size, output_size, h_num, h_dim, dp_rate, time_len, domain):
        #input_size : the dim of input
        #output_size: the dim of output
        #h_num: att head num
        #h_dim: dim of each att head
        #time_len: input frame number
        #domain: do att on spatial domain or temporal domain

        super(ST_ATT_Layer, self).__init__()

        self.pe = PositionalEncoding(input_size, time_len, domain)
        #h_num, h_dim, input_dim, dp_rate,domain
        self.attn = MultiHeadedAttention(h_num, h_dim, input_size, dp_rate, domain) #do att on input dim

        self.ft_map = nn.Sequential(
                        nn.Linear(h_num * h_dim, output_size),
                        nn.ReLU(),
                        LayerNorm(output_size),
                        nn.Dropout(dp_rate),

                        )

        self.init_parameters()








    def forward(self, x):

        x = self.pe(x) #add PE
        x = self.attn(x) #pass attention model
        x = self.ft_map(x)
        return x

    def init_parameters(self):
        model_list = [ self.attn, self.ft_map]
        for model in model_list:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform(p)