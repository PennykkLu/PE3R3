import torch
from torch.utils.data import DataLoader
from residual_ensemble import ResidualEnsemble
import torch.nn.functional as F

threshold = 4.0
gpu_id = 0
device = f"cuda:{gpu_id}"

if __name__ == '__main__':
    user_emb = torch.load("user_emb.pt").to(device)
    item_emb = torch.load("item_emb.pt").to(device)

    dataloader = torch.load("dataloader.pt")
    pre_rate = torch.load("pre_rate.pt")

    high_user = []
    low_user = []
    high_item = []
    low_item = []

    for idx, batch in enumerate(dataloader):
        user = batch[0]
        item = batch[1]
        rate = pre_rate[idx].cpu().tolist()

        for i in range(0, len(rate)):
            if rate[i][0] >= threshold:
                high_user.append(user[i])
                high_item.append(item[i])
            else:
                low_user.append(user[i])
                low_item.append(item[i])

    high_user = torch.tensor(high_user).to(device)
    high_item = torch.tensor(high_item).to(device)
    low_user = torch.tensor(low_user).to(device)
    low_item = torch.tensor(low_item).to(device)

    high_user = user_emb(high_user)
    high_item = item_emb(high_item)
    low_user = user_emb(low_user)
    low_item = item_emb(low_item)

    re_model = ResidualEnsemble(d=3,
                                codename="Instant",
                                cdb_path="./codebook/centroids_embedding_",
                                sim_mode="dot",
                                gpu_id=gpu_id)
    hu_indices = re_model.res_ensemble(high_user)
    lu_indices = re_model.res_ensemble(low_user)
    hi_indices = re_model.res_ensemble(high_item)
    li_indices = re_model.res_ensemble(low_item)


    mmat_h1 = 0.0
    mmat_h2 = 0.0
    mmat_h3 = 0.0
    mmat_l1 = 0.0
    mmat_l2 = 0.0
    mmat_l3 = 0.0

    for i in range(0, high_user.size(0)):
        hu = [hu_indices[0][i], hu_indices[1][i], hu_indices[2][i]]
        hi = [hi_indices[0][i], hi_indices[1][i], hi_indices[2][i]]
        mmat_h1 += F.cosine_similarity(re_model.codebooks[0][hu[0]],
                                       re_model.codebooks[0][hi[0]], dim=0)
        mmat_h2 += F.cosine_similarity(re_model.codebooks[0][hu[0]] + re_model.codebooks[1][hu[1]],
                                       re_model.codebooks[0][hi[0]] + re_model.codebooks[1][hi[1]], dim=0)
        mmat_h3 += F.cosine_similarity(re_model.codebooks[0][hu[0]] + re_model.codebooks[1][hu[1]] + re_model.codebooks[2][hu[2]],
                                       re_model.codebooks[0][hi[0]] + re_model.codebooks[1][hi[1]] + re_model.codebooks[2][hi[2]], dim=0)

    print(f"mmat_h1: {mmat_h1 / high_user.size(0)}, mmat_h2: {mmat_h2 / high_user.size(0)}, mmat_h3: {mmat_h3 / high_user.size(0)}")


    for i in range(0, low_item.size(0)):
        lu = [lu_indices[0][i], lu_indices[1][i], lu_indices[2][i]]
        li = [li_indices[0][i], li_indices[1][i], li_indices[2][i]]
        mmat_l1 += F.cosine_similarity(re_model.codebooks[0][lu[0]],
                                       re_model.codebooks[0][li[0]], dim=0)
        mmat_l2 += F.cosine_similarity(re_model.codebooks[0][lu[0]] + re_model.codebooks[1][lu[1]],
                                       re_model.codebooks[0][li[0]] + re_model.codebooks[1][li[1]], dim=0)
        mmat_l3 += F.cosine_similarity(re_model.codebooks[0][lu[0]] + re_model.codebooks[1][lu[1]] + re_model.codebooks[2][lu[2]],
                                       re_model.codebooks[0][li[0]] + re_model.codebooks[1][li[1]] + re_model.codebooks[2][li[2]], dim=0)
    print(f"mmat_l1: {mmat_l1 / low_user.size(0)}, mmat_l2: {mmat_l2 / low_user.size(0)}, mmat_l3: {mmat_l3 / low_user.size(0)}")

    """
    mmat_h1: 0.06692607700824738, mmat_h2: 0.04794375225901604, mmat_h3: 0.0463026762008667
    mmat_l1: 0.0714874193072319, mmat_l2: 0.01465596817433834, mmat_l3: 0.01068129949271679
    """