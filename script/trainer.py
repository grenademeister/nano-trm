import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

from data_utils import batch_to_tokens


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        num_epochs,
        max_supervision_steps,
        max_val_batches,
        log_every_progress_pct,
        use_ds,
        use_ema,
        ema_model,
        optim_cfg,
        scheduler_cfg,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.max_supervision_steps = max_supervision_steps
        self.max_val_batches = max_val_batches
        self.log_every_progress_pct = log_every_progress_pct
        self.use_ds = use_ds
        self.use_ema = use_ema
        self.ema_model = ema_model
        self.loss_fn = nn.CrossEntropyLoss()

        estimated_train_batches = max(1, len(train_loader))
        total_optimizer_steps = max(
            1, estimated_train_batches * num_epochs * max_supervision_steps
        )
        warmup_step = max(1, int(total_optimizer_steps * scheduler_cfg["warmup_ratio"]))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optim_cfg["lr"],
            betas=(optim_cfg["beta1"], optim_cfg["beta2"]),
            weight_decay=optim_cfg["weight_decay"],
        )
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_step,
            num_training_steps=total_optimizer_steps,
        )

    def run_deep_supervision_steps(self, question_tokens, answer_tokens):
        for step in range(self.max_supervision_steps):
            self.optimizer.zero_grad(set_to_none=True)
            x = self.model.embed_input(question_tokens)
            if self.use_ds and step != 0:
                (y, z), y_hat = self.model(x, y, z)
            else:
                y = x.clone()
                z = x.clone()
                (_, _), y_hat = self.model(x, y, z)
            loss = self.loss_fn(y_hat.reshape(-1, 10), answer_tokens.reshape(-1))
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            if self.use_ema:
                self.ema_model.update(self.model)
        return y_hat, loss

    def run_validation(self, model):
        was_training = model.training
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                question_tokens, answer_tokens = batch_to_tokens(batch, self.device)
                x = model.embed_input(question_tokens)
                y = x.clone()
                z = x.clone()
                (_, _), y_hat = model(x, y, z)
                loss = self.loss_fn(y_hat.reshape(-1, 10), answer_tokens.reshape(-1))
                val_loss += loss.item()
                preds = y_hat.argmax(dim=-1)
                val_correct += (preds == answer_tokens).sum().item()
                val_total += answer_tokens.numel()
                val_batches += 1
                if val_batches >= self.max_val_batches:
                    break
        if was_training:
            model.train()
        avg_val_loss = val_loss / max(1, val_batches)
        val_acc = val_correct / max(1, val_total)
        return avg_val_loss, val_acc

    def train(self):
        self.model.train()
        train_correct = 0
        train_total = 0
        estimated_train_batches = max(1, len(self.train_loader))

        for epoch in range(self.num_epochs):
            next_log_progress_pct = self.log_every_progress_pct
            for batch_in_epoch, batch in enumerate(self.train_loader, start=1):
                question_tokens, answer_tokens = batch_to_tokens(batch, self.device)
                y_hat, loss = self.run_deep_supervision_steps(
                    question_tokens, answer_tokens
                )

                preds = y_hat.argmax(dim=-1)
                batch_correct = (preds == answer_tokens).sum().item()
                batch_total = answer_tokens.numel()
                train_correct += batch_correct
                train_total += batch_total
                batch_train_acc = batch_correct / max(1, batch_total)
                running_train_acc = train_correct / max(1, train_total)
                epoch_progress_pct = min(
                    100.0, 100.0 * batch_in_epoch / max(1, estimated_train_batches)
                )

                if (
                    epoch_progress_pct >= next_log_progress_pct
                    or batch_in_epoch == estimated_train_batches
                ):
                    val_loss, val_acc = self.run_validation(self.model)
                    if self.use_ema:
                        ema_val_loss, ema_val_acc = self.run_validation(self.ema_model.module)
                        print(
                            f"epoch={epoch+1}/{self.num_epochs} progress={epoch_progress_pct:.1f}% "
                            f"loss={loss.item():.4f} "
                            f"train_acc={batch_train_acc:.4f} running_train_acc={running_train_acc:.4f} "
                            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                            f"ema_val_loss={ema_val_loss:.4f} ema_val_acc={ema_val_acc:.4f}"
                        )
                    else:
                        print(
                            f"epoch={epoch+1}/{self.num_epochs} progress={epoch_progress_pct:.1f}% "
                            f"loss={loss.item():.4f} "
                            f"train_acc={batch_train_acc:.4f} running_train_acc={running_train_acc:.4f} "
                            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                        )
                    while next_log_progress_pct <= epoch_progress_pct:
                        next_log_progress_pct += self.log_every_progress_pct

        avg_val_loss, val_acc = self.run_validation(self.model)
        print(f"final_val_loss={avg_val_loss:.4f} final_val_acc={val_acc:.4f}")
        if self.use_ema:
            ema_avg_val_loss, ema_val_acc = self.run_validation(self.ema_model.module)
            print(
                f"final_ema_val_loss={ema_avg_val_loss:.4f} final_ema_val_acc={ema_val_acc:.4f}"
            )
