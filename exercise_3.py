import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from mnist_dataloader import Noisy_MNIST
import os
import argparse

from exercise_1 import FullyConnectedNetwork
from exercise_2 import NoiseDetector


def load_models(denoiser_path, detector_path):
    den_ckpt = torch.load(denoiser_path, weights_only=False)
    denoiser = FullyConnectedNetwork(den_ckpt['layer_sizes'])
    denoiser.load_state_dict(den_ckpt['model_state'])

    det_ckpt = torch.load(detector_path, weights_only=False)
    detector = NoiseDetector(det_ckpt['layer_sizes'], batch_norm=det_ckpt['batch_norm'])
    detector.load_state_dict(det_ckpt['model_state'])

    return denoiser, detector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true',
                        help='Load saved results and plot figures without retraining.')
    parser.add_argument('--denoiser', type=str,
                        default=os.path.join('results', 'exercise_1', 'results_large.tar'),
                        help='Path to the denoiser weights (.tar file).')
    parser.add_argument('--detector', type=str,
                        default=os.path.join('results', 'exercise_2', 'results_shallow.tar'),
                        help='Path to the detector weights (.tar file).')
    args = parser.parse_args()

    data_loc    = './5LSL0-Datasets'
    results_dir = os.path.join('results', 'exercise_3')
    figures_dir = os.path.join('figures', 'exercise_3')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    denoiser, detector = load_models(args.denoiser, args.detector)
    print(f"Loaded denoiser from  {args.denoiser}")
    print(f"Loaded detector from  {args.detector}")

    results_file = os.path.join(results_dir, 'likelihoods.tar')

    if args.load:
        ckpt = torch.load(results_file, weights_only=False)
        p_clean    = ckpt['p_clean']
        p_noisy    = ckpt['p_noisy']
        p_denoised = ckpt['p_denoised']
        all_clean   = ckpt['all_clean']
        all_noisy   = ckpt['all_noisy']
        all_denoised = ckpt['all_denoised']
    else:
        test_mnist = Noisy_MNIST("test", data_loc)
        test_loader = DataLoader(test_mnist, batch_size=256, shuffle=False)

        p_clean, p_noisy, p_denoised = [], [], []
        all_clean, all_noisy, all_denoised = [], [], []
        denoiser.eval()
        detector.eval()
        with torch.no_grad():
            for clean, noisy, _ in test_loader:
                denoised = denoiser(noisy[:, 0])

                p_clean.append(torch.sigmoid(detector(clean[:, 0])))
                p_noisy.append(torch.sigmoid(detector(noisy[:, 0])))
                p_denoised.append(torch.sigmoid(detector(denoised)))

                all_clean.append(clean[:, 0])
                all_noisy.append(noisy[:, 0])
                all_denoised.append(denoised)

        p_clean    = torch.cat(p_clean).numpy()
        p_noisy    = torch.cat(p_noisy).numpy()
        p_denoised = torch.cat(p_denoised).numpy()
        all_clean   = torch.cat(all_clean).numpy()
        all_noisy   = torch.cat(all_noisy).numpy()
        all_denoised = torch.cat(all_denoised).numpy()

        torch.save({
            'p_clean': p_clean, 'p_noisy': p_noisy, 'p_denoised': p_denoised,
            'all_clean': all_clean, 'all_noisy': all_noisy, 'all_denoised': all_denoised,
        }, results_file)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([1 - p_clean, 1 - p_noisy, 1 - p_denoised], labels=['Clean', 'Noisy', 'Denoised'])
    ax.set_ylabel('p(noisy)')
    ax.set_title('Detector likelihood per image set')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'boxplot_likelihoods.png'), dpi=150)
    plt.show()

    change = p_denoised - p_noisy

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot(change, labels=['Denoised - Noisy'])
    ax.set_ylabel('Δ p(clean)')
    ax.set_title('Change in detector likelihood after denoising')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'boxplot_change.png'), dpi=150)
    plt.show()

    import numpy as np
    idx_most   = int(np.argmax(change))
    idx_least  = int(np.argmin(change))
    idx_median = int(np.argsort(change)[len(change) // 2])

    selected = [('Most fooled',  idx_most),
                ('Median',       idx_median),
                ('Least fooled', idx_least)]

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    col_labels = ['Clean', 'Noisy', 'Denoised']

    for row, (title, idx) in enumerate(selected):
        imgs   = [all_clean[idx], all_noisy[idx], all_denoised[idx]]
        p_vals = [p_clean[idx],   p_noisy[idx],   p_denoised[idx]]
        for col, (img, p) in enumerate(zip(imgs, p_vals)):
            ax = axes[row, col]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{col_labels[col]}\np(clean)={p:.3f}', fontsize=8)
            ax.axis('off')
        axes[row, 0].set_ylabel(title, fontsize=9)

    plt.suptitle('Most fooled / Median / Least fooled', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'grid_fooled.png'), dpi=150)
    plt.show()

    detector2_file = os.path.join(results_dir, 'detector2.tar')
    det_layer_sizes = [1024, 128, 1]
    num_epochs      = 20
    batch_size      = 64

    if args.load and os.path.exists(detector2_file):
        ckpt2 = torch.load(detector2_file, weights_only=False)
        detector2_state  = ckpt2['model_state']
        train_losses2    = ckpt2['train_losses']
        val_losses2      = ckpt2['val_losses']
        train_accs2      = ckpt2['train_accs']
        val_accs2        = ckpt2['val_accs']
    else:
        from torch.utils.data import random_split
        train_mnist = Noisy_MNIST("train", data_loc)
        n_val   = int(0.2 * len(train_mnist))
        n_train = len(train_mnist) - n_val
        train_set, val_set = random_split(train_mnist, [n_train, n_val])
        train_loader2 = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader2   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

        detector2  = NoiseDetector(det_layer_sizes, batch_norm=True)
        optimizer2 = torch.optim.Adam(detector2.parameters(), lr=1e-3)
        criterion2 = torch.nn.BCEWithLogitsLoss()

        for p in denoiser.parameters():
            p.requires_grad = False
        denoiser.eval()

        train_losses2, val_losses2, train_accs2 = [], [], []
        val_accs2 = []

        for epoch in range(num_epochs):
            detector2.train()
            epoch_loss, correct, total = 0.0, 0, 0
            for clean, noisy, _ in train_loader2:
                mask = torch.rand(noisy.size(0)) < 0.5
                with torch.no_grad():
                    noisy_input = noisy[:, 0].clone()
                    noisy_input[mask] = denoiser(noisy[:, 0])[mask]

                imgs   = torch.cat([clean[:, 0], noisy_input])
                labels = torch.cat([torch.ones(clean.size(0)),
                                    torch.zeros(noisy.size(0))])

                optimizer2.zero_grad()
                preds = detector2(imgs)
                loss  = criterion2(preds, labels)
                loss.backward()
                optimizer2.step()

                epoch_loss += loss.item()
                correct    += ((preds >= 0).float() == labels).sum().item()
                total      += labels.size(0)

            train_losses2.append(epoch_loss / len(train_loader2))
            train_accs2.append(correct / total)

            detector2.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for clean, noisy, _ in val_loader2:
                    imgs   = torch.cat([clean[:, 0], noisy[:, 0]])
                    labels = torch.cat([torch.ones(clean.size(0)),
                                        torch.zeros(noisy.size(0))])
                    preds  = detector2(imgs)
                    val_loss += criterion2(preds, labels).item()
                    correct  += ((preds >= 0).float() == labels).sum().item()
                    total    += labels.size(0)

            val_losses2.append(val_loss / len(val_loader2))
            val_accs2.append(correct / total)
            print(f"  [detector2] Epoch {epoch+1}/{num_epochs}  "
                  f"Train Loss: {train_losses2[-1]:.4f}  Train Acc: {train_accs2[-1]*100:.1f}%  "
                  f"Val Loss: {val_losses2[-1]:.4f}  Val Acc: {val_accs2[-1]*100:.1f}%")

        detector2_state = detector2.state_dict()
        torch.save({
            'model_state':   detector2_state,
            'layer_sizes':   det_layer_sizes,
            'batch_norm':    True,
            'train_losses':  train_losses2,
            'val_losses':    val_losses2,
            'train_accs':    train_accs2,
            'val_accs':      val_accs2,
        }, detector2_file)
        print(f"Saved {detector2_file}")

    epochs = range(1, num_epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_losses2, label='train')
    ax1.plot(epochs, val_losses2,   label='val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('BCE Loss')
    ax1.set_title('Detector v2 — Loss'); ax1.legend()
    ax2.plot(epochs, [a * 100 for a in train_accs2], label='train')
    ax2.plot(epochs, [a * 100 for a in val_accs2],   label='val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Detector v2 — Accuracy'); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'detector2_training.png'), dpi=150)
    plt.show()

    results2_file = os.path.join(results_dir, 'likelihoods2.tar')
    detector2 = NoiseDetector(det_layer_sizes, batch_norm=True)
    detector2.load_state_dict(detector2_state)
    detector2.eval()

    if args.load and os.path.exists(results2_file):
        ckpt = torch.load(results2_file, weights_only=False)
        p2_clean    = ckpt['p_clean']
        p2_noisy    = ckpt['p_noisy']
        p2_denoised = ckpt['p_denoised']
    else:
        all_noisy_t    = torch.from_numpy(all_noisy)
        all_clean_t    = torch.from_numpy(all_clean)
        all_denoised_t = torch.from_numpy(all_denoised)
        p2_clean, p2_noisy, p2_denoised = [], [], []
        with torch.no_grad():
            for i in range(0, len(all_noisy_t), 256):
                p2_clean.append(torch.sigmoid(detector2(all_clean_t[i:i+256])))
                p2_noisy.append(torch.sigmoid(detector2(all_noisy_t[i:i+256])))
                p2_denoised.append(torch.sigmoid(detector2(all_denoised_t[i:i+256])))
        p2_clean    = torch.cat(p2_clean).numpy()
        p2_noisy    = torch.cat(p2_noisy).numpy()
        p2_denoised = torch.cat(p2_denoised).numpy()
        torch.save({'p_clean': p2_clean, 'p_noisy': p2_noisy, 'p_denoised': p2_denoised}, results2_file)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([1 - p2_clean, 1 - p2_noisy, 1 - p2_denoised], labels=['Clean', 'Noisy', 'Denoised'])
    ax.set_ylabel('p(noisy)')
    ax.set_title('Detector v2 likelihood per image set')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'g_boxplot_likelihoods.png'), dpi=150)
    plt.show()

    change2 = p2_denoised - p2_noisy
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot([change, change2], labels=['Detector v1', 'Detector v2'])
    ax.set_ylabel('Δ p(clean)')
    ax.set_title('Change in likelihood after denoising')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'g_boxplot_change.png'), dpi=150)
    plt.show()

    idx2_most   = int(np.argmax(change2))
    idx2_least  = int(np.argmin(change2))
    idx2_median = int(np.argsort(change2)[len(change2) // 2])

    selected2 = [('Most fooled',  idx2_most),
                 ('Median',       idx2_median),
                 ('Least fooled', idx2_least)]

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for row, (title, idx) in enumerate(selected2):
        imgs   = [all_clean[idx], all_noisy[idx], all_denoised[idx]]
        p_vals = [p2_clean[idx],  p2_noisy[idx],  p2_denoised[idx]]
        for col, (img, p) in enumerate(zip(imgs, p_vals)):
            ax = axes[row, col]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{col_labels[col]}\np(clean)={p:.3f}', fontsize=8)
            ax.axis('off')
        axes[row, 0].set_ylabel(title, fontsize=9)
    plt.suptitle('Detector v2 — Most fooled / Median / Least fooled', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'g_grid_fooled.png'), dpi=150)
    plt.show()

    denoiser2_file = os.path.join(results_dir, 'denoiser2.tar')
    alpha      = 0.001
    num_epochs = 20
    batch_size = 64

    for p in detector2.parameters():
        p.requires_grad = False
    detector2.eval()

    if args.load and os.path.exists(denoiser2_file):
        ckpt = torch.load(denoiser2_file, weights_only=False)
        denoiser2_state   = ckpt['model_state']
        den_layer_sizes2  = ckpt['layer_sizes']
        mse_losses        = ckpt['mse_losses']
        det_losses        = ckpt['det_losses']
    else:
        from torch.utils.data import random_split
        den_ckpt = torch.load(args.denoiser, weights_only=False)
        den_layer_sizes2 = den_ckpt['layer_sizes']
        denoiser2  = FullyConnectedNetwork(den_layer_sizes2)
        denoiser2.load_state_dict(den_ckpt['model_state'])
        optimizer3 = torch.optim.Adam(denoiser2.parameters(), lr=1e-4)
        mse_criterion = torch.nn.MSELoss()
        det_criterion = torch.nn.BCEWithLogitsLoss()

        train_mnist3 = Noisy_MNIST("train", data_loc)
        n_val3   = int(0.2 * len(train_mnist3))
        n_train3 = len(train_mnist3) - n_val3
        train_set3, _ = random_split(train_mnist3, [n_train3, n_val3])
        train_loader3 = DataLoader(train_set3, batch_size=batch_size, shuffle=True)

        mse_losses, det_losses = [], []

        for epoch in range(num_epochs):
            denoiser2.train()
            epoch_mse, epoch_det = 0.0, 0.0
            for clean, noisy, _ in train_loader3:
                optimizer3.zero_grad()
                denoised = denoiser2(noisy[:, 0])

                mse_loss = mse_criterion(denoised, clean[:, 0])
                det_logits = detector2(denoised)
                det_loss   = det_criterion(det_logits, torch.zeros(denoised.size(0)))
                total_loss = mse_loss - alpha * det_loss

                total_loss.backward()
                optimizer3.step()

                epoch_mse += mse_loss.item()
                epoch_det += det_loss.item()

            mse_losses.append(epoch_mse / len(train_loader3))
            det_losses.append(epoch_det / len(train_loader3))
            print(f"  [denoiser2] Epoch {epoch+1}/{num_epochs}  "
                  f"MSE: {mse_losses[-1]:.4f}  Det: {det_losses[-1]:.4f}")
        denoiser2_state = denoiser2.state_dict()
        torch.save({
            'model_state': denoiser2_state,
            'layer_sizes': den_layer_sizes2,
            'mse_losses':  mse_losses,
            'det_losses':  det_losses,
        }, denoiser2_file)
        print(f"Saved {denoiser2_file}")

    epochs = range(1, num_epochs + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, mse_losses, label='MSE loss')
    ax.plot(epochs, det_losses, label='Detection loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Denoiser v2 — adversarial training loss components')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'denoiser2_training.png'), dpi=150)
    plt.show()

    results3_file = os.path.join(results_dir, 'likelihoods3.tar')
    denoiser2 = FullyConnectedNetwork(den_layer_sizes2)
    denoiser2.load_state_dict(denoiser2_state)
    denoiser2.eval()

    if args.load and os.path.exists(results3_file):
        ckpt = torch.load(results3_file, weights_only=False)
        p3_clean    = ckpt['p_clean']
        p3_noisy    = ckpt['p_noisy']
        p3_denoised = ckpt['p_denoised']
        all3_clean    = ckpt['all_clean']
        all3_noisy    = ckpt['all_noisy']
        all3_denoised = ckpt['all_denoised']
    else:
        test_mnist3  = Noisy_MNIST("test", data_loc)
        test_loader3 = DataLoader(test_mnist3, batch_size=256, shuffle=False)
        p3_clean, p3_noisy, p3_denoised = [], [], []
        all3_clean, all3_noisy, all3_denoised = [], [], []
        with torch.no_grad():
            for clean, noisy, _ in test_loader3:
                denoised3 = denoiser2(noisy[:, 0])
                p3_clean.append(torch.sigmoid(detector2(clean[:, 0])))
                p3_noisy.append(torch.sigmoid(detector2(noisy[:, 0])))
                p3_denoised.append(torch.sigmoid(detector2(denoised3)))
                all3_clean.append(clean[:, 0])
                all3_noisy.append(noisy[:, 0])
                all3_denoised.append(denoised3)
        p3_clean    = torch.cat(p3_clean).numpy()
        p3_noisy    = torch.cat(p3_noisy).numpy()
        p3_denoised = torch.cat(p3_denoised).numpy()
        all3_clean    = torch.cat(all3_clean).numpy()
        all3_noisy    = torch.cat(all3_noisy).numpy()
        all3_denoised = torch.cat(all3_denoised).numpy()
        torch.save({
            'p_clean': p3_clean, 'p_noisy': p3_noisy, 'p_denoised': p3_denoised,
            'all_clean': all3_clean, 'all_noisy': all3_noisy, 'all_denoised': all3_denoised,
        }, results3_file)
    #boxplot likelihoods
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([1 - p3_clean, 1 - p3_noisy, 1 - p3_denoised], labels=['Clean', 'Noisy', 'Denoised'])
    ax.set_ylabel('p(noisy)')
    ax.set_title('Detector v2 + Denoiser v2 — likelihood per image set')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'j_boxplot_likelihoods.png'), dpi=150)
    plt.show()

    change3 = p3_denoised - p3_noisy
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([change, change2, change3],
               labels=['Det v1\nDen v1', 'Det v2\nDen v1', 'Det v2\nDen v2'])
    ax.set_ylabel('Δ p(clean)')
    ax.set_title('Change in likelihood after denoising — all versions')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'j_boxplot_change.png'), dpi=150)
    plt.show()

    # (j) 3x3 grid
    idx3_most   = int(np.argmax(change3))
    idx3_least  = int(np.argmin(change3))
    idx3_median = int(np.argsort(change3)[len(change3) // 2])

    selected3 = [('Most fooled',  idx3_most),
                 ('Median',       idx3_median),
                 ('Least fooled', idx3_least)]

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for row, (title, idx) in enumerate(selected3):
        imgs   = [all3_clean[idx], all3_noisy[idx], all3_denoised[idx]]
        p_vals = [p3_clean[idx],   p3_noisy[idx],   p3_denoised[idx]]
        for col, (img, p) in enumerate(zip(imgs, p_vals)):
            ax = axes[row, col]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{col_labels[col]}\np(clean)={p:.3f}', fontsize=8)
            ax.axis('off')
        axes[row, 0].set_ylabel(title, fontsize=9)
    plt.suptitle('Det v2 + Den v2 — Most fooled / Median / Least fooled', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'j_grid_fooled.png'), dpi=150)
    plt.show()
