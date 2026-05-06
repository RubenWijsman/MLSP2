# Exercise 1.2 — Noise Detection with a Fully Connected Network
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader, Dataset
from mnist_dataloader import Noisy_MNIST
import os
import argparse


class NoiseDetectionDataset(Dataset):
    def __init__(self, noisy_mnist: Noisy_MNIST):
        self.clean = noisy_mnist.Clean_Images  # (N, 1, 32, 32)
        self.noisy = noisy_mnist.Noisy_Images  # (N, 1, 32, 32)

    def __len__(self):
        return 2 * self.clean.shape[0]

    def __getitem__(self, idx):
        n = self.clean.shape[0]
        if idx < n:
            return self.clean[idx, 0], torch.tensor(1.0)
        else:
            return self.noisy[idx - n, 0], torch.tensor(0.0)


class NoiseDetector(nn.Module):
    def __init__(self, layer_sizes, batch_norm=True):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.network(x).squeeze(1)  # (B,)  raw logits


def _eval_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            predicted = (model(images) >= 0.0).float()  # threshold logits at 0
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)
    return correct / total


def train_noise_detector(model, train_loader, val_loader, criterion,
                          optimizer, num_epochs, results_dir, name):
    train_losses, val_losses = [], []
    train_accs = []
    os.makedirs(results_dir, exist_ok=True)

    pre_acc = _eval_accuracy(model, val_loader)
    val_accs = [pre_acc]
    print(f"  Before training  Val Acc: {pre_acc*100:.1f}%")

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted   = (preds >= 0.0).float()  # threshold logits at 0
            correct    += (predicted == labels).sum().item()
            total      += labels.size(0)

        train_losses.append(epoch_loss / len(train_loader))
        train_accs.append(correct / total)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                preds       = model(images)
                val_loss   += criterion(preds, labels).item()
                predicted   = (preds >= 0.0).float()  # threshold logits at 0
                correct    += (predicted == labels).sum().item()
                total      += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(correct / total)

        print(f"  Epoch {epoch+1}/{num_epochs}  "
              f"Train Loss: {train_losses[-1]:.4f}  Train Acc: {train_accs[-1]*100:.1f}%  "
              f"Val Loss: {val_losses[-1]:.4f}  Val Acc: {val_accs[-1]*100:.1f}%")

    return train_losses, val_losses, train_accs, val_accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true',
                        help='Load saved results and plot figures without retraining.')
    args = parser.parse_args()

    data_loc    = './5LSL0-Datasets'
    batch_size  = 64
    num_epochs  = 20
    results_dir = os.path.join('results', 'exercise_2')
    figures_dir = os.path.join('figures', 'exercise_2')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Three architecture variations to compare
    experiments = [
        {'name': 'shallow',  'layer_sizes': [32 * 32, 128, 1],                   'batch_norm': True},
        {'name': 'medium',   'layer_sizes': [32 * 32, 512, 128, 1],               'batch_norm': True},
        {'name': 'deep',     'layer_sizes': [32 * 32, 512, 256, 128, 64, 1],      'batch_norm': True},
    ]

    # -----------------------------------------------------------------------
    # Build datasets (shared across all experiments)
    # -----------------------------------------------------------------------
    full_train_mnist = Noisy_MNIST("train", data_loc)
    full_test_mnist  = Noisy_MNIST("test",  data_loc)

    full_det_train = NoiseDetectionDataset(full_train_mnist)
    full_det_test  = NoiseDetectionDataset(full_test_mnist)

    n_val   = int(0.2 * len(full_det_train))
    n_train = len(full_det_train) - n_val
    train_set, val_set = random_split(full_det_train, [n_train, n_val])

    train_loader = DataLoader(train_set,     batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_set,       batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(full_det_test, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(full_det_test)}")

    criterion = nn.BCEWithLogitsLoss()

    results = {}

    for exp in experiments:
        name        = exp['name']
        layer_sizes = exp['layer_sizes']
        results_file = os.path.join(results_dir, f"results_{name}.tar")

        batch_norm = exp['batch_norm']

        if args.load:
            ckpt = torch.load(results_file)
            batch_norm  = ckpt.get('batch_norm', False)  # backwards compat
            layer_sizes = ckpt.get('layer_sizes', layer_sizes)
            results[name] = {
                'train_losses': ckpt['train_losses'],
                'val_losses':   ckpt['val_losses'],
                'train_accs':   ckpt['train_accs'],
                'val_accs':     ckpt['val_accs'],
                'model_state':  ckpt['model_state'],
                'layer_sizes':  layer_sizes,
                'batch_norm':   batch_norm,
            }
            num_epochs = ckpt['num_epochs']
            print(f"Loaded {results_file}")
        else:
            model     = NoiseDetector(layer_sizes, batch_norm=batch_norm)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            print(f"\nTraining '{name}'  {layer_sizes}  batch_norm={batch_norm}")
            train_losses, val_losses, train_accs, val_accs = train_noise_detector(
                model, train_loader, val_loader, criterion, optimizer, num_epochs,
                results_dir, name
            )
            torch.save({
                'train_losses': train_losses,
                'val_losses':   val_losses,
                'train_accs':   train_accs,
                'val_accs':     val_accs,
                'num_epochs':   num_epochs,
                'batch_norm':   batch_norm,
                'layer_sizes':  layer_sizes,
                'model_state':  model.state_dict(),
            }, results_file)
            print(f"Saved results to {results_file}")

            results[name] = {
                'train_losses': train_losses,
                'val_losses':   val_losses,
                'train_accs':   train_accs,
                'val_accs':     val_accs,
                'model_state':  model.state_dict(),
                'layer_sizes':  layer_sizes,
                'batch_norm':   batch_norm,
            }

    print()
    for exp in experiments:
        name        = exp['name']
        layer_sizes = exp['layer_sizes']
        model = NoiseDetector(layer_sizes, batch_norm=results[name]['batch_norm'])
        model.load_state_dict(results[name]['model_state'])
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                preds      = model(images)
                test_loss += criterion(preds, labels).item()
                predicted  = (preds >= 0.0).float()
                correct   += (predicted == labels).sum().item()
                total     += labels.size(0)
        print(f"[{name}]  Test Loss: {test_loss/len(test_loader):.4f}  "
              f"Test Accuracy: {correct/total*100:.2f}%")

    epochs     = list(range(1, num_epochs + 1))
    val_epochs = list(range(0, num_epochs + 1))

    # --- Combined: train loss + val loss + val accuracy ---
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 4))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, exp in enumerate(experiments):
        name  = exp['name']
        color = colors[i % len(colors)]
        ax_loss.plot(epochs, results[name]['train_losses'],
                     color=color, linestyle='-',  label=f'{name} train')
        ax_loss.plot(epochs, results[name]['val_losses'],
                     color=color, linestyle='--', label=f'{name} val')
        ax_acc.plot(val_epochs, [a * 100 for a in results[name]['val_accs']],
                    color=color, label=name)

    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('BCE Loss')
    ax_loss.set_title('Train (—) vs Validation (- -) Loss')
    ax_loss.legend(fontsize=7)

    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.set_title('Validation Accuracy  (epoch 0 = untrained)')
    ax_acc.legend()

    plt.suptitle('Noise Detector — Training Curves', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'training_curves.png'), dpi=150)
    plt.show()

    # --- Qualitative predictions (best model by final val accuracy) ---
    best_name  = max(experiments, key=lambda e: results[e['name']]['val_accs'][-1])['name']
    best_sizes = results[best_name]['layer_sizes']
    best_bn    = results[best_name]['batch_norm']
    best_model = NoiseDetector(best_sizes, batch_norm=best_bn)
    best_model.load_state_dict(results[best_name]['model_state'])
    best_model.eval()
    print(f"\nBest model (highest val accuracy): '{best_name}'")

    test_mnist_loader = DataLoader(full_test_mnist, batch_size=20, shuffle=False)
    clean_imgs, noisy_imgs, _ = next(iter(test_mnist_loader))

    with torch.no_grad():
        p_clean_clean = torch.sigmoid(best_model(clean_imgs[:, 0])).numpy()
        p_clean_noisy = torch.sigmoid(best_model(noisy_imgs[:, 0])).numpy()

    fig, axes = plt.subplots(2, 10, figsize=(14, 3))
    for i in range(10):
        axes[0, i].imshow(clean_imgs[i, 0].numpy(), cmap='gray')
        axes[0, i].set_title(f"p={p_clean_clean[i]:.3f}", fontsize=7)
        axes[0, i].axis('off')
        axes[1, i].imshow(noisy_imgs[i, 0].numpy(), cmap='gray')
        axes[1, i].set_title(f"p={p_clean_noisy[i]:.3f}", fontsize=7)
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Clean', fontsize=8)
    axes[1, 0].set_ylabel('Noisy', fontsize=8)
    plt.suptitle(f"Noise Detector predictions — {best_name}  (p = p_clean)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'predictions.png'), dpi=150)
    plt.show()
