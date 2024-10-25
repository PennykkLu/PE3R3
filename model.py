import torch
import torch.nn as nn


class FmLayer(nn.Module):
    def __init__(self, p, k):
        super(FmLayer, self).__init__()
        self.p, self.k = p, k
        self.linear = nn.Linear(self.p, 1, bias=True)
        self.v = nn.Parameter(torch.Tensor(self.p, self.k), requires_grad=True)
        self.v.data.uniform_(-0.01, 0.01)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        linear_part = self.linear(x)
        inter_part1 = torch.pow(torch.mm(x, self.v), 2)
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        pair_interactions = torch.sum(torch.sub(inter_part1, inter_part2), dim=1)
        self.drop(pair_interactions)
        output = linear_part.transpose(1, 0) + 0.5 * pair_interactions
        return output.view(-1, 1)


class FM(nn.Module):
    def __init__(self, num_users, num_items, num_factors, re_model, mode):
        super(FM, self).__init__()
        self.re_model = re_model
        self.mode = mode
        self.user_id_vec = nn.Embedding(num_users, num_factors)
        self.item_id_vec = nn.Embedding(num_items, num_factors)

        self.fm = FmLayer(num_factors * 2, 10)

    def forward(self, u_id, i_id):
        u_vec = self.user_id_vec(u_id)
        i_vec = self.item_id_vec(i_id)
        # add residual ensemble for embedding learning
        if self.mode is not "naive":
            # 执行查询
            u_indices = self.re_model.res_ensemble(u_vec)
            u_vec_re = u_vec + (self.re_model.get_emb(u_indices) - u_vec).detach()
            i_indices = self.re_model.res_ensemble(i_vec)
            i_vec_re = i_vec + (self.re_model.get_emb(i_indices) - i_vec).detach()
            recon_loss = ((u_vec - (u_vec + (u_vec_re - u_vec)).detach()) ** 2).mean() + \
                         ((i_vec - (i_vec + (i_vec_re - i_vec)).detach()) ** 2).mean()
            x = torch.cat((u_vec_re, i_vec_re), dim=1)
        else:
            x = torch.cat((u_vec, i_vec), dim=1)
            recon_loss = 0.0
        rate = self.fm(x)
        return rate, recon_loss


class Deepfm(nn.Module):
    def __init__(self, num_users, num_items, num_factors, re_model, mode):
        super(Deepfm, self).__init__()
        self.user_id_vec = nn.Embedding(num_users, num_factors)
        self.item_id_vec = nn.Embedding(num_items, num_factors)
        self.re_model = re_model
        self.mode = mode

        self.fm = FmLayer(num_factors * 2, 10)

        self.dnn_hidden_units = [128, 64, 32]

        # DNN
        self.dropout = nn.Dropout(0.2)
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.hidden_units = [num_factors * 2] + self.dnn_hidden_units
        self.Linears = nn.ModuleList(
            [nn.Linear(self.hidden_units[i], self.hidden_units[i + 1]) for i in range(len(self.hidden_units) - 1)])
        self.relus = nn.ModuleList([nn.ReLU() for i in range(len(self.hidden_units) - 1)])
        for name, tensor in self.Linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.01)
        self.dnn_outlayer = nn.Linear(self.dnn_hidden_units[-1], 1, bias=False)

    def forward(self, u_id, i_id):
        u_vec = self.user_id_vec(u_id)
        i_vec = self.item_id_vec(i_id)

        # add residual ensemble for embedding learning
        if self.mode is not "naive":
            # 执行查询
            u_indices = self.re_model.res_ensemble(u_vec)
            u_vec_re = u_vec + (self.re_model.get_emb(u_indices) - u_vec).detach()
            i_indices = self.re_model.res_ensemble(i_vec)
            i_vec_re = i_vec + (self.re_model.get_emb(i_indices) - i_vec).detach()
            recon_loss = ((u_vec - (u_vec + (u_vec_re - u_vec)).detach()) ** 2).mean() + \
                         ((i_vec - (i_vec + (i_vec_re - i_vec)).detach()) ** 2).mean()
            input_x = torch.cat((u_vec_re, i_vec_re), dim=1)
        else:
            input_x = torch.cat((u_vec, i_vec), dim=1)
            recon_loss = 0.0

        # FM
        fm_logit = self.fm(input_x)
        # Deep
        for i in range(len(self.Linears)):
            fc = self.Linears[i](input_x)
            fc = self.relus[i](fc)
            fc = self.dropout(fc)
            input_x = fc
        dnn_logit = self.dnn_outlayer(input_x)

        y_pre = fm_logit + dnn_logit + self.bias
        return y_pre, recon_loss
