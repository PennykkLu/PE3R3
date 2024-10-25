import torch.nn.functional as F


class Trainer:
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            optimizer,
            scheduler,
            num_epochs,
            device,
            criterion,
            patience
    ) -> None:

        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.criterion = criterion
        self.patience = patience

    def train(self):
        best_val_loss = float('inf')
        test_loss = float('inf')
        current_patience = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            try:
                for idx, batch in enumerate(self.train_dataloader):
                    user = batch[0].to(self.device)
                    item = batch[1].to(self.device)
                    true_rating = batch[3].float().unsqueeze(-1).to(self.device)
                    predicted_rating, recon_loss = self.model(user, item)
                    loss = self.criterion(predicted_rating, true_rating) + recon_loss * 2
                    # loss = self.criterion(predicted_rating, true_rating)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    if idx % 100 == 0:
                        print(f"epoch {epoch} step {idx}, loss = {loss.item()}")
            except Exception as e:
                print(e)

            pre_rate, val_loss = self.inference(epoch, self.eval_dataloader)
            print(f"valid loss: {val_loss}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                current_patience = 0
                _, test_loss = self.inference(epoch, self.test_dataloader, mode='test')
                import torch
                torch.save(self.model.user_id_vec, "user_emb.pt")
                torch.save(self.model.item_id_vec, "item_emb.pt")
                torch.save(self.eval_dataloader, "dataloader.pt")
                torch.save(pre_rate, "pre_rate.pt")
            else:
                current_patience += 1
                if current_patience == self.patience:
                    print("Early stopping triggered!")
                    print(f"Best validation mse: {best_val_loss}, test mse: {test_loss}")
                    break

    def inference(self, epoch, dataloader, mode='valid'):
        self.model.eval()
        total_batches = 0
        total_mse = 0.0
        pre_rate = []
        for idx, batch in enumerate(dataloader):
            user = batch[0].to(self.device)
            item = batch[1].to(self.device)
            true_rating = batch[3].float().unsqueeze(-1).to(self.device)
            predicted_rating, _ = self.model(user, item)
            pre_rate.append(predicted_rating)
            batch_mse = F.mse_loss(predicted_rating, true_rating)
            total_mse += batch_mse
            total_batches += 1
        # if mode == 'valid':
        #     print(f"********** Validation: epoch {epoch}, mse = {total_mse / total_batches} **********")
        # else:
        #     print(f"********** Inference: epoch {epoch}, mse = {total_mse / total_batches} **********")
        return pre_rate, total_mse / total_batches
