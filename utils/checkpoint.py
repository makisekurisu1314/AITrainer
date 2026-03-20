import torch
import os


def save_checkpoint(state, is_best, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, "latest.pth")
    torch.save(state, latest_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save(state, best_path)


def load_checkpoint(model, optimizer, checkpoint_path="checkpoints/latest.pth"):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}, best_loss={best_loss}")
        return model, optimizer, start_epoch, best_loss
    else:
        print("No checkpoint found, starting fresh")
        return model, optimizer, 0, float('inf')
