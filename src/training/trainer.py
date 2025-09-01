import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

def train_model(model, train_set, val_set, loss_fn, config, device, save_dir):
    writer = SummaryWriter(os.path.join(save_dir, "logs"))
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=config['lr_gamma'])

    best_val_loss = float('inf')
    patience = config.get('patience', 10)
    epochs_no_improve = 0

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        writer.add_scalar('Loss/train', train_loss, epoch)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        writer.add_scalar('Loss/val', val_loss, epoch)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        scheduler.step()

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "models", "best_model.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break
        if (epoch+1) % config.get('save_every', 5) == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, "models", f"checkpoint_epoch{epoch+1}.pth"))
    writer.close()