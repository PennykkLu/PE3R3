import torch
import torch.nn.functional as F


class ResidualEnsemble(torch.nn.Module):
    def __init__(self, d, codename, cdb_path, sim_mode, gpu_id):
        super(ResidualEnsemble, self).__init__()
        self.d = d
        self.gpu_id = gpu_id
        self.sim_mode = sim_mode
        self.codebooks = self.load_codebooks(codename, cdb_path)
        self.to_device()

    def load_codebooks(self, codename, cdb_path):
        codebooks = []
        for i in range(self.d):
            codebook = torch.tensor(torch.load(cdb_path + codename + str(i + 1) + ".pt"))
            codebooks.append(codebook)
        return codebooks

    def to_device(self):
        self.to(torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"))
        for i in range(self.d):
            self.codebooks[i] = self.codebooks[i].to(
                torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"))

    def res_ensemble(self, query, current_codebook_index=0):
        if self.sim_mode == "cos":
            similarities = F.cosine_similarity(query.unsqueeze(0), self.codebooks[current_codebook_index].unsqueeze(0),
                                               dim=1)
        else:
            similarities = torch.mm(query, self.codebooks[current_codebook_index].t())

        max_index = torch.argmax(similarities, dim=1)

        if current_codebook_index < self.d - 1:
            residual_vector = query - self.codebooks[current_codebook_index][max_index].detach()
            next_index = current_codebook_index + 1
            residual_result = self.res_ensemble(residual_vector, next_index)
            return [max_index] + residual_result
        else:
            return [max_index]

    def get_emb(self, indices):
        emb = torch.zeros_like(self.codebooks[0][0]).unsqueeze(0).repeat(indices[0].size(0), 1)
        for i, index in enumerate(indices):
            emb += self.codebooks[i][index]
        return emb