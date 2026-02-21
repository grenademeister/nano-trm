import torch
import torch.nn as nn
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
import argparse
from timm.utils import ModelEmaV2
from transformers import get_cosine_schedule_with_warmup
features = Features(
    {
        "source": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "rating": Value("int64"),
    }
)


def collate_fn(batch):
    return {
        "source": [item["source"] for item in batch],
        "question": [item["question"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "rating": [item["rating"] for item in batch],
    }


def encode_row(row):
    values = []
    for ch in str(row):
        if ch == ".":
            values.append(0)
        elif ch.isdigit():
            values.append(int(ch))
    if len(values) != 81:
        raise ValueError(f"Expected 81 cells, got {len(values)} from row={row!r}")
    return values


def print_model_parameter_count(model: nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


class TinyNetworkBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8,att='self_att'):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.att= att
        if(att == 'self_att'):
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        elif(att == 'mlp'):
            self.attn =nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        else:
            raise ValueError(f"arg parse att should be att or mlp , not {att}")
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, query, context):
        # maybe add pre-norm later?
        if(self.att == "self_att"):
            attn_output, _ = self.attn(query, context, context)
        else:
            mixed = torch.cat([query,context],dim=-1)
            attn_output = self.attn(mixed)
        query = self.norm1(query + attn_output)
        ff_output = self.mlp(query)
        query = self.norm2(query + ff_output)
        return query


class TRM(nn.Module):
    def __init__(self, embed_dim=512, n_reasoning_steps=6, n_recursion_steps=3,att='self_att'):
        super().__init__()
        self.net = TinyNetworkBlock(embed_dim,att=att)
        self.net2 = TinyNetworkBlock(embed_dim,att=att)
        self.n = n_reasoning_steps
        self.T = n_recursion_steps
        self.token_embedding = nn.Embedding(10, embed_dim)
        self.positional_embedding = nn.Embedding(81, embed_dim)
        self.output_head = nn.Linear(embed_dim, 10)  # For Sudoku (0-9 tokens)

    def embed_input(self, x):
        batch_size, seq_len = x.size()
        pos_ids = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        return self.token_embedding(x) + self.positional_embedding(pos_ids)

    def latent_recursion(self, x, y, z):
        for _ in range(self.n):  # latent reasoning
            combined_context = torch.cat([x, y], dim=1)
            z = self.net(query=z, context=combined_context)
        y = self.net2(query=y, context=z)  # answer refinement
        return y, z

    def forward(self, x, y, z):
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self.latent_recursion(x, y, z)
        y, z = self.latent_recursion(x, y, z)
        return (y.detach(), z.detach()), self.output_head(y)


def main():
    parser = argparse.ArgumentParser(description="d")
    parser.add_argument('--ds',action='store_true',help='whether to use deep supervision')
    parser.add_argument('--att',type=str,default='self_att',help='self_att or mlp ( for sudoku)')
    parser.add_argument('--ema',action='store_true',help='using ema')
    args = parser.parse_args()
    embed_dim = 512
    batch_size = 2
    max_supervision_steps = 16  # N_sup from the paper
    max_val_batches = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trm_model = TRM(embed_dim, n_reasoning_steps=6,att = args.att).to(device)
    ema_model = ModelEmaV2(trm_model,decay=0.999)
    ema_model.module.eval()
    print_model_parameter_count(trm_model)
    trm_model.train()
    optimizer = torch.optim.AdamW(trm_model.parameters(),
    lr=1e-4,
    betas=(0.9,0.95),
    weight_decay=0.1)
    warmup_step=200
    training_step=1000
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_step,
        num_training_steps=training_step

    )
    
    loss_fn = nn.CrossEntropyLoss()

    full_train = load_dataset("sapientinc/sudoku-extreme", features=features)["train"]
    split = full_train.train_test_split(test_size=0.01, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    for batch_idx, batch in enumerate(train_loader):
        question_tokens = torch.tensor(
            [encode_row(q) for q in batch["question"]], dtype=torch.long, device=device
        )
        answer_tokens = torch.tensor(
            [encode_row(a) for a in batch["answer"]], dtype=torch.long, device=device
        )

        for _ in range(max_supervision_steps):
            optimizer.zero_grad(set_to_none=True)
            x = trm_model.embed_input(question_tokens)
            if(args.ds and (_!=0)):
                (y,z),y_hat = trm_model(x,y,z)
            else:
                y = x.clone()
                z = x.clone()
                (a, b), y_hat = trm_model(x, y, z)
            loss = loss_fn(y_hat.reshape(-1, 10), answer_tokens.reshape(-1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if(args.ema):
                ema_model.update(trm_model)
        if batch_idx % 5 == 0:
            if(args.ema):
                with torch.no_grad():
                    x=ema_model.module.embed_input(question_tokens)
                    y = x.clone()
                    z = x.clone()
                    (a, b), y_hat = ema_model.module(x, y, z)
                    ema_loss = loss_fn(y_hat.reshape(-1, 10), answer_tokens.reshape(-1))
                print(f"batch={batch_idx} loss={loss.item():.4f} ema loss={ema_loss.item():.4f}")
            else:
                print(f"batch={batch_idx} loss={loss.item():.4f} ")
        if batch_idx >= training_step:  # Train on a subset for demo purposes
            break

    trm_model.eval()
    val_loss = 0.0
    val_correct = 0
    ema_val_loss=0.0
    ema_val_correct=0
    val_total = 0
    val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            question_tokens = torch.tensor(
                [encode_row(q) for q in batch["question"]],
                dtype=torch.long,
                device=device,
            )
            answer_tokens = torch.tensor(
                [encode_row(a) for a in batch["answer"]],
                dtype=torch.long,
                device=device,
            )
            x = trm_model.embed_input(question_tokens)
            y = x.clone()
            z = x.clone()
            (_, _), y_hat = trm_model(x, y, z)
            loss = loss_fn(y_hat.reshape(-1, 10), answer_tokens.reshape(-1))
            val_loss += loss.item()
            preds = y_hat.argmax(dim=-1)
            val_correct += (preds == answer_tokens).sum().item()
            val_total += answer_tokens.numel()
            val_batches += 1
            if(args.ema):
                x = ema_model.module.embed_input(question_tokens)
                y = x.clone()
                z = x.clone()
                (_, _), y_hat = ema_model.module(x, y, z)
                loss = loss_fn(y_hat.reshape(-1, 10), answer_tokens.reshape(-1))
                ema_val_loss += loss.item()
                preds = y_hat.argmax(dim=-1)
                ema_val_correct += (preds == answer_tokens).sum().item()               
            if val_batches >= max_val_batches:
                break

    avg_val_loss = val_loss / max(1, val_batches)
    val_acc = val_correct / max(1, val_total)
    print(f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}")
    if(args.ema):
        ema_avg_val_loss = ema_val_loss / max(1, val_batches)
        ema_val_acc = ema_val_correct / max(1, val_total)
        print(f"ema_val_loss={ema_avg_val_loss:.4f} ema_val_acc={ema_val_acc:.4f}")


if __name__ == "__main__":
    main()
