from torch.utils.data import Dataset
import pandas as pd
import random


class MyDataset(Dataset):
    def __init__(self, path, split, seed, user_label_encoder, item_label_encoder):
        self.split = split
        self.user_label_encoder = user_label_encoder
        self.item_label_encoder = item_label_encoder
        random.seed(seed)
        self.path = path
        self._load_dataset()
        self._build(seed)

    # 从数据集中获取一个训练样本
    def __getitem__(self, index):
        instance = self.instances.iloc[index].tolist()
        return instance

    # 数据集长度
    def __len__(self):
        return len(self.instances)

    def _load_dataset(self):
        self.dataset = pd.read_csv(self.path)
        self.dataset['user'] = self.user_label_encoder.fit_transform(self.dataset['user'])
        self.dataset['item'] = self.item_label_encoder.fit_transform(self.dataset['item'])


    def _build(self,seed):
        data = self.dataset.sample(frac=1, random_state=seed)
        data.reset_index(drop=True, inplace=True)

        if self.split == 'train':
            self.instances = data.iloc[:int(len(data) * 0.8)]
        elif self.split == "valid":
            self.instances = data.iloc[int(len(data) * 0.8):int(len(data) * 0.9)]
        elif self.split == "test":
            self.instances = data.iloc[int(len(data) * 0.9):]
        else:
            raise Exception("wrong split")

