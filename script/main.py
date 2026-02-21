import torch
from timm.utils import ModelEmaV2

from data_utils import build_dataloaders, get_or_create_cached_splits
from helper import load_config, print_model_parameter_count
from model.trm import TRM
from parser import parse_args
from trainer import Trainer


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    optim_cfg = cfg["optimizer"]
    scheduler_cfg = cfg["scheduler"]
    val_cfg = cfg["validation"]
    log_cfg = cfg["logging"]

    att = args.att if args.att is not None else model_cfg["att"]
    use_ds = args.ds if args.ds is not None else train_cfg["ds"]
    use_ema = args.ema if args.ema is not None else train_cfg["ema"]

    embed_dim = model_cfg["embed_dim"]
    batch_size = train_cfg["batch_size"]
    num_epochs = train_cfg["num_epochs"]
    max_supervision_steps = train_cfg["max_supervision_steps"]
    log_every_progress_pct = log_cfg["every_progress_pct"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trm_model = TRM(
        embed_dim=embed_dim,
        n_reasoning_steps=model_cfg["n_reasoning_steps"],
        n_recursion_steps=model_cfg["n_recursion_steps"],
        att=att,
    ).to(device)
    ema_model = None
    if use_ema:
        ema_model = ModelEmaV2(trm_model, decay=train_cfg["ema_decay"])
        ema_model.module.eval()
    print_model_parameter_count(trm_model)

    train_dataset, val_dataset = get_or_create_cached_splits(data_cfg)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    train_loader, val_loader = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=batch_size,
        val_batch_size=val_cfg["batch_size"],
        device=device,
    )

    trainer = Trainer(
        model=trm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        max_supervision_steps=max_supervision_steps,
        max_val_batches=val_cfg["max_batches"],
        log_every_progress_pct=log_every_progress_pct,
        use_ds=use_ds,
        use_ema=use_ema,
        ema_model=ema_model,
        optim_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()
