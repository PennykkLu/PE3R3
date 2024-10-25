import argparse
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import set_seed, AdamW, get_linear_schedule_with_warmup
from dataset_constructor import MyDataset
from arguments import predefined_args
from model import FM, Deepfm
from trainer import Trainer
from RMCL import RMCL
from residual_ensemble import ResidualEnsemble
from sklearn.preprocessing import LabelEncoder

parser = predefined_args(argparse.ArgumentParser())
args = parser.parse_args()
set_seed(args.seed)
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

# dataset
user_label_encoder = LabelEncoder()
item_label_encoder = LabelEncoder()
train_data = MyDataset(args.dataset, "train", args.seed,
                       user_label_encoder=user_label_encoder,
                       item_label_encoder=item_label_encoder)
valid_data = MyDataset(args.dataset, "valid", args.seed,
                       user_label_encoder=user_label_encoder,
                       item_label_encoder=item_label_encoder)
test_data = MyDataset(args.dataset, "test", args.seed,
                      user_label_encoder=user_label_encoder,
                      item_label_encoder=item_label_encoder)

train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=args.inf_batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=args.inf_batch_size, shuffle=False)

re_model = ResidualEnsemble(d=args.d,
                            codename=args.codename,
                            cdb_path=args.cdb_path,
                            sim_mode=args.sim_mode,
                            gpu_id=args.gpu_id)
re_model.to_device()

# Instant: 5130, 1685
# Tools: 16638, 10217
# Pet: 19856,8510
# Toys: 19412, 11924
model = FM(num_users=5130,
           num_items=1685,
           num_factors=args.emb_dim,
           re_model=re_model,
           mode='re').to(device)  # "naive" or not

# model = Deepfm(num_users=1429,
#                num_items=900,
#                num_factors=args.emb_dim,
#                re_model=re_model,
#                mode='naive').to(device)  # "naive" or not


# office: 4905, 2420
# music: 1429, 900
# model = RMCL(num_users=5130,
#              num_items=1685,
#              num_factors=args.emb_dim,
#              re_model=re_model,
#              mode='not',
#              ue=user_label_encoder,
#              ie=item_label_encoder).to(device)  # "naive" or not


optimizer = AdamW(model.parameters(), lr=args.lr)

total_steps = len(train_dataloader) * args.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
criterion = nn.MSELoss()

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=valid_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=args.epochs,
    criterion=criterion,
    device=device,
    patience=args.patience
)
trainer.train()

# if args.save:
#     name = args.save_model_path
#     trainer.save(f'{name}-{args.dataset}-Q-{args.learning_rate}-{int(time.time())}')
