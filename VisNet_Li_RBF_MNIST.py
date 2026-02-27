"""
VisNet-LI Hebbian classifier for MNIST.

This script shows how to build a lightweight VisNet-LI-inspired pipeline:
1. Create a Gabor filter bank as the initial feature extractor.
2. Pass Gabor responses through several Hebbian competitive layers that enforce
   spatial sparsity via local inhibition.
3. Train a shallow linear classifier on top of the Hebbian stack.

The cleaned-up version is structured for easy sharing on GitHub.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def gabor_kernel(size: int, sigma: float, theta: float, lambd: float, gamma: float, psi: float) -> np.ndarray:
    """Return a single Gabor kernel."""
    half = (size - 1) / 2
    y, x = np.meshgrid(np.linspace(-half, half, size), np.linspace(-half, half, size))
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-0.5 * ((x_theta ** 2 + (gamma ** 2) * y_theta ** 2) / (sigma ** 2)))
    carrier = np.cos(2 * np.pi / lambd * x_theta + psi)
    return (envelope * carrier).astype(np.float32)


def build_gabor_bank(
    kernel_size: int = 15,
    frequencies: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    orientations: Sequence[float] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    phases: Sequence[float] = (0, np.pi / 2),
    gammas: Sequence[float] = (0.5, 1.0),
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Stack a bank of Gabor filters into a conv2d-ready tensor."""
    sigma = kernel_size * 0.35
    filters = []
    for freq in frequencies:
        lambd = 1.0 / freq
        for theta in orientations:
            for psi in phases:
                for gamma in gammas:
                    kernel = gabor_kernel(kernel_size, sigma, theta, lambd, gamma, psi)
                    filters.append(torch.from_numpy(kernel))
    return torch.stack(filters).unsqueeze(1).to(device)


def preprocess_images(images: torch.Tensor, gabor_filters: torch.Tensor) -> torch.Tensor:
    """Apply the Gabor filters and return the feature maps."""
    responses = F.conv2d(images, gabor_filters, padding=gabor_filters.shape[-1] // 2)
    return responses


def apply_local_inhibition_and_sparsity(activations: torch.Tensor, keep_ratio: float = 0.02) -> torch.Tensor:
    """Global sparsity inhibition: keep strongest fraction per sample."""
    if not (0 < keep_ratio < 1):
        return activations
    keep = max(1, round(activations.shape[1] * keep_ratio))
    kth = activations.shape[1] - keep + 1
    threshold = torch.kthvalue(activations, kth, dim=1, keepdim=True)[0]
    return torch.where(activations >= threshold, activations, torch.zeros_like(activations))


def l2_normalize(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize tensor rows to unit length."""
    norms = tensor.norm(dim=1, keepdim=True).clamp(min=eps)
    return tensor / norms


def normalize_rows(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize each row independently (used for centers)."""
    norms = tensor.norm(dim=1, keepdim=True).clamp(min=eps)
    return tensor / norms


class HebbianLayer2d(nn.Module):
    """2D Hebbian layer where each neuron observes a local patch (kernel_size×kernel_size)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int | None = None,
        keep_ratio: float = 0.02,
        gamma: float = 0.6,
        upsample_factor: float = 1.0,
        hebbian_decay: float = 0.01,
        post_trace_alpha: float = 0.8,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2 if padding is None else padding
        self.keep_ratio = keep_ratio
        self.gamma = gamma
        self.upsample_factor = upsample_factor
        self.hebbian_decay = hebbian_decay
        self.post_trace_alpha = post_trace_alpha
        self.in_features = in_channels * kernel_size * kernel_size
        self.weights = nn.Parameter(torch.randn(out_channels, self.in_features), requires_grad=False)
        self.register_buffer("_prev_post", torch.empty(0))

    def _extract_patches(self, inputs: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        unfolded_input = inputs
        if not torch.isclose(torch.tensor(self.upsample_factor), torch.tensor(1.0)):
            unfolded_input = F.interpolate(
                inputs,
                scale_factor=self.upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
        patches = F.unfold(
            unfolded_input,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )
        B = unfolded_input.size(0)
        H_out = (unfolded_input.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (unfolded_input.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        patches = patches.transpose(1, 2).reshape(-1, self.in_features)
        return patches, H_out, W_out

    def _compute_activations(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        patches, H_out, W_out = self._extract_patches(inputs)
        distances = torch.cdist(patches, self.weights)
        activations = torch.exp(-self.gamma * distances)
        activations = apply_local_inhibition_and_sparsity(activations, keep_ratio=self.keep_ratio)
        activations = activations.view(-1, self.out_channels)
        return activations, patches, H_out, W_out

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activations, _, H_out, W_out = self._compute_activations(inputs)
        B = inputs.size(0)
        activations = activations.view(B, H_out * W_out, self.out_channels)
        activations = activations.transpose(1, 2).reshape(B, self.out_channels, H_out, W_out)
        return activations
  
    def activate(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        activations, patches, H_out, W_out = self._compute_activations(inputs)
        B = inputs.size(0)
        activations = activations.view(B, H_out * W_out, self.out_channels)
        activations = activations.transpose(1, 2).reshape(B, self.out_channels, H_out, W_out)
        return activations, patches

    def hebbian_update(self, patches: torch.Tensor, activations: torch.Tensor, lr: float) -> None:
        if lr <= 0:
            return None
        flat_activations = activations.reshape(-1, self.out_channels)
        if self.post_trace_alpha > 0:
            if self._prev_post.numel() == 0 or self._prev_post.shape != flat_activations.shape:
                post = (1.0 - self.post_trace_alpha) * flat_activations
            else:
                post = (1.0 - self.post_trace_alpha) * flat_activations + self.post_trace_alpha * self._prev_post
            self._prev_post = flat_activations.detach()
        else:
            post = flat_activations
        with torch.no_grad():
            # Original-style masked pre/post interaction across successive layers.
            updates = post.T @ patches
            updates = updates / max(1, patches.size(0))
            self.weights += lr * updates - self.hebbian_decay * self.weights
            self.weights.copy_(normalize_rows(self.weights))


def build_data_loaders(batch_size: int, input_size: int) -> tuple[DataLoader, DataLoader]:
    """Create standard MNIST train/test loaders."""
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the fraction of correct predictions."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def evaluate(
    layers: Iterable[HebbianLayer2d],
    classifier: nn.Linear,
    dataloader: DataLoader,
    gabor_filters: torch.Tensor,
    device: torch.device,
    ) -> float:
    """Measure accuracy on the provided dataloader."""
    classifier.eval()
    correct = total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        features = preprocess_images(images, gabor_filters)
        for layer in layers:
            features = layer(features)
        logits = classifier(features.reshape(features.size(0), -1))
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def train_and_evaluate(
    layers: Sequence[HebbianLayer2d],
    classifier: nn.Linear,
    train_loader: DataLoader,
    test_loader: DataLoader,
    gabor_filters: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    ) -> None:
    """Run the cleaned training loop."""
    optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            features = preprocess_images(images, gabor_filters)
            current_inputs = features
            for layer in layers:
                activations, patches = layer.activate(current_inputs)
                layer.hebbian_update(patches, activations, args.hebbian_lr)
                current_inputs = activations
            logits = classifier(current_inputs.reshape(current_inputs.size(0), -1))
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += calculate_accuracy(logits, labels)
        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_acc / len(train_loader)
        val_acc = evaluate(layers, classifier, test_loader, gabor_filters, device)
        best_acc = max(best_acc, val_acc)
        print(
            f"Epoch {epoch:02d} | Loss {avg_loss:.4f} | Train {avg_acc*100:.2f}% | "
            f"Val {val_acc*100:.2f}% | Best {best_acc*100:.2f}%"
        )
    print(f"Training complete. Best validation accuracy: {best_acc*100:.2f}%")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training script."""
    parser = argparse.ArgumentParser(description="VisNet-LI Hebbian RBF training on MNIST.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hebbian-lr", type=float, default=1e-3, help="Hebbian learning rate.")
    parser.add_argument("--classifier-lr", type=float, default=1e-3, help="Classifier learning rate.")
    parser.add_argument(
        "--layer-dims",
        type=int,
        nargs="+",
        default=[400, 400, 400, 400],
        help="Output size for each Hebbian layer.",
    )
    parser.add_argument("--keep-ratio", type=float, default=0.001, help="Fraction of neurons kept per layer.")
    parser.add_argument("--gamma", type=float, default=0.6, help="RBF sharpness parameter.")
    parser.add_argument("--hebbian-decay", type=float, default=0.01, help="Weight decay term in Hebbian update.")
    parser.add_argument(
        "--post-trace-alpha",
        type=float,
        default=0.8,
        help="EMA factor for postsynaptic trace in Hebbian update (0 disables trace).",
    )
    parser.add_argument("--kernel-size", type=int, default=15, help="Height/width of the Gabor kernels.")
    parser.add_argument("--patch-size", type=int, default=7, help="Height/width of Hebbian receptive fields.")
    parser.add_argument("--stride", type=int, default=2, help="Stride when extracting patches.")
    parser.add_argument("--gammas", type=float, nargs="+", default=(0.5, 1.0), help="Gamma values for the Gabor bank.")
    parser.add_argument("--upsample-factor", type=float, default=1.0, help="Upsample factor before unfolding (use 2 for 0.5 stride).")
    parser.add_argument("--input-size", type=int, default=32, help="Size of the (square) input images.")
    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_data_loaders(args.batch_size, args.input_size)
    gabor_filters = build_gabor_bank(kernel_size=args.kernel_size, gammas=args.gammas, device=device)
    layers: list[HebbianLayer2d] = []
    in_channels = gabor_filters.shape[0]
    for idx, out_dim in enumerate(args.layer_dims):
        layer = HebbianLayer2d(
            in_channels,
            out_dim,
            kernel_size=args.patch_size,
            stride=args.stride,
            keep_ratio=args.keep_ratio,
            gamma=args.gamma,
            upsample_factor=args.upsample_factor if idx == 0 else 1.0,
            hebbian_decay=args.hebbian_decay,
            post_trace_alpha=args.post_trace_alpha,
        ).to(device)
        layers.append(layer)
        in_channels = out_dim
    dummy = torch.randn(1, gabor_filters.shape[0], args.input_size, args.input_size).to(device)
    current = dummy
    for layer in layers:
        current = layer(current)
    classifier_input = current.reshape(1, -1).size(1)
    classifier = nn.Linear(classifier_input, 10).to(device)
    train_and_evaluate(layers, classifier, train_loader, test_loader, gabor_filters, args, device)


if __name__ == "__main__":
    main()

