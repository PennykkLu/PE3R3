import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.nn.init as init


class RMCL(nn.Module):
    def __init__(self, num_users, num_items, num_factors, re_model, mode, ue, ie):
        super(RMCL, self).__init__()
        self.re_model = re_model
        self.mode = mode
        self.user_id_vec = nn.Embedding(num_users, num_factors)
        self.item_id_vec = nn.Embedding(num_items, num_factors)

        self.dropout = 0.3
        self.k = 10

        self.h_vectors = nn.Parameter(torch.randn(self.k, num_factors))

        user_rep = pickle.load(open("../dataset/Amazon_sum/Instant_Video_user_rep.pkl", 'rb'))
        item_rep = pickle.load(open("../dataset/Amazon_sum/Instant_Video_item_rep.pkl", 'rb'))
        self.ue = ue
        self.ie = ie

        uid = list(user_rep.keys())
        uvec = list(user_rep.values())
        uvec_s = []
        encoded_id = self.ue.transform(uid)
        for i in range(0, len(encoded_id)):
            uvec_s.append(uvec[np.where(encoded_id == i)[0][0]])
        # [user num, 768]
        user_vec = torch.cat(uvec_s, dim=0)

        iid = list(item_rep.keys())
        ivec = list(item_rep.values())
        ivec_s = []
        encoded_id = self.ie.transform(iid)
        for i in range(0, len(encoded_id)):
            ivec_s.append(ivec[np.where(encoded_id == i)[0][0]])
        # [user num, 768]
        item_vec = torch.cat(ivec_s, dim=0)

        self.user_c = nn.Embedding.from_pretrained(user_vec, freeze=False)
        self.item_c = nn.Embedding.from_pretrained(item_vec, freeze=False)

        # MLP网络--没说user和item是不是共享的mlp encoder
        self.mlp1 = nn.Sequential(
            nn.Linear(768, 400),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(400, 128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(128, self.k),
            nn.Softmax(dim=-1)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(768, 400),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(400, 128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )
        self.mlp4 = nn.Sequential(
            nn.Linear(128, self.k),
            nn.Softmax(dim=-1)
        )

        self.temperature = 0.1

        self.model = nn.Sequential(
            nn.Linear(4*128, 3*128),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(3*128, 2*128),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(2*128, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(128, 1)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data, mean=0.0, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)

    def forward(self, user, item):
        user_c = self.user_c(user)
        item_c = self.item_c(item)

        user_c = self.mlp1(user_c)
        user_coe = self.mlp2(user_c)
        weighted_hu = torch.sum(user_coe.unsqueeze(-1) * self.h_vectors, dim=1)
        lsim_u = torch.mean(F.cosine_similarity(user_c, weighted_hu, dim=1))

        item_c = self.mlp3(item_c)
        item_coe = self.mlp4(item_c)
        weighted_hi = torch.sum(item_coe.unsqueeze(-1) * self.h_vectors, dim=1)
        lsim_i = torch.mean(F.cosine_similarity(item_c, weighted_hi, dim=1))

        inner_product = torch.mm(self.h_vectors, self.h_vectors.t())
        inner_product.fill_diagonal_(0)
        norms = torch.norm(self.h_vectors, dim=1, keepdim=True) ** 2
        norm_product = torch.mm(norms, norms.t())
        lind = (inner_product ** 2) / (norm_product + 1e-8)
        lind = torch.mean(lind)

        lcl_u = -torch.sum(user_coe * torch.log(
            F.softmax(torch.matmul(user_c, self.h_vectors.T) / self.temperature, dim=1) + 1e-10), dim=1).mean()
        lcl_i = -torch.sum(item_coe * torch.log(
            F.softmax(torch.matmul(item_c, self.h_vectors.T) / self.temperature, dim=1) + 1e-10), dim=1).mean()

        loss = lcl_i + lcl_u + 100 * lind - (lsim_u + lsim_i)

        u_vec = self.user_id_vec(user)
        i_vec = self.item_id_vec(item)
        if self.mode is not "naive":
            # 执行查询
            u_indices = self.re_model.res_ensemble(u_vec)
            u_vec_re = u_vec + (self.re_model.get_emb(u_indices) - u_vec).detach()
            i_indices = self.re_model.res_ensemble(i_vec)
            i_vec_re = i_vec + (self.re_model.get_emb(i_indices) - i_vec).detach()
            recon_loss = ((u_vec - (u_vec + (u_vec_re - u_vec)).detach()) ** 2).mean() + \
                         ((i_vec - (i_vec + (i_vec_re - i_vec)).detach()) ** 2).mean()
            u_vec = u_vec_re
            i_vec = i_vec_re
        else:
            recon_loss = 0.0

        predict = self.model(torch.cat((weighted_hu * weighted_hi,
                                        weighted_hu - weighted_hi,
                                        u_vec,
                                        i_vec), dim=1))

        return predict, loss + recon_loss
        # return predict, loss
