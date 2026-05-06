# Exercise 1.1 — MNIST Denoising with a Fully Connected Network
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from mnist_dataloader import create_dataloaders, Noisy_MNIST
import os
import argparse


class FullyConnectedNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape
        x = x.flatten(start_dim=1)
        x = self.network(x)
        x = x.view(shape)
        return x


def train_model(model, train_split_loader, val_split_loader, criterion,
                optimizer, num_epochs, save_every_n, checkpoint_dir, name):
    """Train the model and return train/val losses per epoch."""
    train_losses, val_losses = [], []
    iteration = 0
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for clean, noisy, _ in train_split_loader:
            x      = noisy[:, 0, :, :]
            target = clean[:, 0, :, :]
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            iteration += 1
            if iteration % save_every_n == 0:
                ckpt = os.path.join(checkpoint_dir, f"{name}_iter{iteration}.pth")
                torch.save(model.state_dict(), ckpt)
        train_losses.append(epoch_train_loss / len(train_split_loader))

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for clean, noisy, _ in val_split_loader:
                x      = noisy[:, 0, :, :]
                target = clean[:, 0, :, :]
                epoch_val_loss += criterion(model(x), target).item()
        val_losses.append(epoch_val_loss / len(val_split_loader))

        print(f"  [{name}] Epoch {epoch+1}/{num_epochs}  "
              f"Train: {train_losses[-1]:.4f}  Val: {val_losses[-1]:.4f}")

    return train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true',
                        help='Load saved results and plot figures without retraining.')
    args = parser.parse_args()

    data_loc = './5LSL0-Datasets'
    batch_size = 64
    num_epochs = 30
    save_every_n = 100
    results_dir   = os.path.join('results', 'exercise_1')
    figures_dir   = os.path.join('figures', 'exercise_1')
    checkpoint_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(results_dir,   exist_ok=True)
    os.makedirs(figures_dir,   exist_ok=True)

    experiments = [
        {'name': 'small',  'layer_sizes': [32**2, 128,    32**2]},
        {'name': 'medium', 'layer_sizes': [32**2, 24**2, 16**2, 32**2]},
        {'name': 'large',  'layer_sizes': [32**2, 32**2, 24**2, 16**2, 24**2, 32**2]},
    ]

    # Load data once
    train_loader, test_loader = create_dataloaders(data_loc, batch_size)
    full_train = Noisy_MNIST("train", data_loc)
    n_val = int(0.2 * len(full_train))
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(full_train, [n_train, n_val])
    train_split_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_split_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, drop_last=False)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_loader.dataset)}")

    criterion = nn.MSELoss()

    # plot a sample of the dataset
    clean_sample, noisy_sample, label_sample = next(iter(train_loader))
    fig, axes = plt.subplots(2, 10, figsize=(14, 3))
    for i in range(10):
        axes[0, i].imshow(clean_sample[i, 0].numpy(), cmap='gray')
        axes[0, i].set_title(f"{label_sample[i].item()}", fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(noisy_sample[i, 0].numpy(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Clean', fontsize=8)
    axes[1, 0].set_ylabel('Noisy', fontsize=8)
    plt.suptitle('MNIST dataset sample', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'dataset_sample.png'), dpi=150)
    plt.show()

    results = {}  # name -> dict of losses and predictions

    if args.load:
        # Restore noisy_in and labels from the first tar file (same for all experiments)
        first_ckpt = torch.load(os.path.join(results_dir, f"results_{experiments[0]['name']}.tar"))
        noisy_in   = first_ckpt['noisy_in']
        labels     = first_ckpt['labels']
        num_epochs = first_ckpt['num_epochs']  # restore so the loss plot x-axis is correct
        for exp in experiments:
            name = exp['name']
            fname = os.path.join(results_dir, f"results_{name}.tar")
            ckpt = torch.load(fname)
            results[name] = {
                'train_losses':   ckpt['train_losses'],
                'val_losses':     ckpt['val_losses'],
                'untrained_pred': ckpt['untrained_pred'],
                'trained_pred':   ckpt['trained_pred'],
            }
            print(f"Loaded {fname}")
    else:
        # Get the fixed test batch used for all predictions
        clean_batch, noisy_batch, labels = next(iter(test_loader))
        noisy_in = noisy_batch[:, 0, :, :]

        for exp in experiments:
            name       = exp['name']
            layer_sizes = exp['layer_sizes']
            print(f"\n=== Training: {name} {layer_sizes} ===")

            model     = FullyConnectedNetwork(layer_sizes)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

            # Untrained prediction
            model.eval()
            with torch.no_grad():
                untrained_pred = model(noisy_in)

            train_losses, val_losses = train_model(
                model, train_split_loader, val_split_loader,
                criterion, optimizer, num_epochs, save_every_n,
                checkpoint_dir, name
            )

            # Trained prediction
            model.eval()
            with torch.no_grad():
                trained_pred = model(noisy_in)

            results[name] = {
                'train_losses':   train_losses,
                'val_losses':     val_losses,
                'untrained_pred': untrained_pred,
                'trained_pred':   trained_pred,
            }

            tar_path = os.path.join(results_dir, f"results_{name}.tar")
            torch.save({
                'layer_sizes':    layer_sizes,
                'num_epochs':     num_epochs,
                'train_losses':   train_losses,
                'val_losses':     val_losses,
                'noisy_in':       noisy_in,
                'labels':         labels,
                'untrained_pred': untrained_pred,
                'trained_pred':   trained_pred,
                'model_state':    model.state_dict(),
            }, tar_path)
            print(f"  Saved {tar_path}")

    plt.figure(figsize=(8, 5))
    epochs = range(1, num_epochs + 1)
    for exp in experiments:
        name = exp['name']
        plt.plot(epochs, results[name]['train_losses'], linestyle='--', label=f'{name} train')
        plt.plot(epochs, results[name]['val_losses'],   linestyle='-',  label=f'{name} val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Across Network Sizes')
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'loss_comparison.png'), dpi=150)
    plt.show()

    n_show = 5
    n_rows = 1 + 2 * len(experiments)  # noisy input + (untrained + trained) per experiment
    fig, axes = plt.subplots(n_rows, n_show, figsize=(n_show * 2, n_rows * 2))

    def show_row(row_idx, imgs, ylabel):
        for i in range(n_show):
            axes[row_idx, i].imshow(imgs[i].numpy(), cmap='gray')
            axes[row_idx, i].axis('off')
            if row_idx == 0:
                axes[row_idx, i].set_title(f"{labels[i].item()}", fontsize=8)
        axes[row_idx, 0].set_ylabel(ylabel, fontsize=7)

    show_row(0, noisy_in, 'Noisy input')
    for k, exp in enumerate(experiments):
        name = exp['name']
        show_row(1 + 2*k,     results[name]['untrained_pred'], f'{name}\n(untrained)')
        show_row(1 + 2*k + 1, results[name]['trained_pred'],   f'{name}\n(trained)')

    plt.suptitle('Test set predictions per network size', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'prediction_comparison.png'), dpi=150)
    plt.show()
