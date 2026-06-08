"""Four-way STL-10 LIME/MARS comparison.

This script intentionally keeps the experiment small and fixed: one image, one
segmentation, one sample count, one Gaussian-noise level, and one target class.
It compares exactly four explanation settings:

1. Standard LIME with Ridge surrogate.
2. Standard LIME-style perturbations with a MARS (py-earth Earth) surrogate.
3. Gaussian-noise LIME-style perturbations with a MARS surrogate.
4. Gaussian-noise LIME-style perturbations with a Ridge surrogate.

The MARS usage follows the user's provided LEMON/HF-LIME code pattern: use
``pyearth.Earth`` as the local surrogate, then convert basis-function outputs
back into per-feature / per-superpixel contributions.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lime import lime_image
from matplotlib.patches import Patch
from pyearth import Earth
from skimage.segmentation import slic
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


class ImprovedSTL10CNN(nn.Module):
    """Same STL-10 CNN architecture used by the existing notebooks."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: Path, device: torch.device) -> ImprovedSTL10CNN:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Copy stl10_best_model.pt "
            "into this folder before running the script."
        )
    model = ImprovedSTL10CNN(num_classes=len(CLASS_NAMES)).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_test_dataset(data_root: str):
    eval_transform = transforms.Compose([transforms.ToTensor()])
    return datasets.STL10(
        root=data_root,
        split="test",
        download=True,
        transform=eval_transform,
    )


def make_predict_fn(model: nn.Module, device: torch.device):
    def predict_fn(images):
        model.eval()
        images = np.asarray(images).astype(np.float32)
        if images.max() > 1.0:
            images = images / 255.0
        images_tensor = torch.tensor(images).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            probs = F.softmax(model(images_tensor), dim=1)
        return probs.cpu().numpy()

    return predict_fn


def find_correct_example(dataset, predict_fn, desired_label: int | None = 8):
    for idx in range(len(dataset)):
        image_tensor, label = dataset[idx]
        if desired_label is not None and label != desired_label:
            continue
        image_uint8 = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        probs = predict_fn(np.array([image_uint8]))[0]
        pred = int(np.argmax(probs))
        if pred == label:
            return idx, image_uint8, label, probs

    raise RuntimeError("No correctly classified test image found for the requested class.")


def segmentation_fn(image, n_segments: int = 70, compactness: int = 12, sigma: int = 1):
    return slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
    )


def default_kernel_width(num_segments: int) -> float:
    return 0.75 * np.sqrt(num_segments)


def lime_kernel(distance: np.ndarray, kernel_width: float) -> np.ndarray:
    return np.sqrt(np.exp(-(distance**2) / (kernel_width**2)))


def perturb_hide_color(image: np.ndarray, segments: np.ndarray, mask: np.ndarray, hide_color=0):
    perturbed = image.copy()
    for i, seg_val in enumerate(np.unique(segments)):
        if mask[i] == 0:
            perturbed[segments == seg_val] = hide_color
    return perturbed.astype(np.uint8)


def perturb_with_gaussian_noise(
    image: np.ndarray,
    segments: np.ndarray,
    mask: np.ndarray,
    sigma_noise: float,
):
    perturbed = image.astype(np.float32).copy()
    noise = np.random.normal(loc=0, scale=sigma_noise, size=image.shape)
    for i, seg_val in enumerate(np.unique(segments)):
        if mask[i] == 0:
            region = segments == seg_val
            perturbed[region] += noise[region]
    return np.clip(perturbed, 0, 255).astype(np.uint8)


def make_transfer_dataset(
    image: np.ndarray,
    segments: np.ndarray,
    predict_fn,
    target_class: int,
    num_samples: int,
    kernel_width: float,
    perturbation: str,
    sigma_noise: float,
):
    num_segments = len(np.unique(segments))
    data = np.ones((num_samples, num_segments), dtype=np.float32)
    data[1:] = np.random.randint(0, 2, size=(num_samples - 1, num_segments))

    perturbed_images = []
    for mask in data:
        if perturbation == "standard":
            perturbed = perturb_hide_color(image, segments, mask, hide_color=0)
        elif perturbation == "gaussian":
            perturbed = perturb_with_gaussian_noise(image, segments, mask, sigma_noise=sigma_noise)
        else:
            raise ValueError(f"Unknown perturbation: {perturbation}")
        perturbed_images.append(perturbed)

    y = predict_fn(np.asarray(perturbed_images))[:, target_class]
    distances = np.sqrt(np.sum((1 - data) ** 2, axis=1))
    weights = lime_kernel(distances, kernel_width)
    return data, y, weights


def ridge_segment_weights(data: np.ndarray, y: np.ndarray, weights: np.ndarray):
    surrogate = Ridge(alpha=1.0, fit_intercept=True, random_state=42)
    surrogate.fit(data, y, sample_weight=weights)
    return np.asarray(surrogate.coef_).ravel(), surrogate.score(data, y, sample_weight=weights)


def _basis_variables(bfun):
    for attr in ("variables", "variable"):
        if hasattr(bfun, attr):
            vars_idx = getattr(bfun, attr)
            if callable(vars_idx):
                vars_idx = vars_idx()
            break
    else:
        if hasattr(bfun, "get_variable"):
            vars_idx = bfun.get_variable()
        else:
            return None

    if vars_idx is None:
        return None
    if isinstance(vars_idx, (int, np.integer)):
        return (int(vars_idx),)
    try:
        return tuple(int(v) for v in vars_idx)
    except TypeError:
        return None


def mars_feature_contributions(earth_model: Earth, instance: np.ndarray, n_features: int):
    """Aggregate py-earth basis-function contributions back to original features."""
    x = np.asarray(instance, dtype=float).reshape(1, -1)
    basis_values = np.asarray(earth_model.transform(x))
    coef = np.asarray(earth_model.coef_).squeeze()
    if coef.ndim != 1:
        coef = coef.ravel()

    if len(coef) == basis_values.shape[1] + 1:
        term_coefs = coef[1:]
        term_values = basis_values[0]
    elif len(coef) == basis_values.shape[1]:
        term_coefs = coef
        term_values = basis_values[0]
    else:
        raise ValueError(
            f"Unexpected MARS shape: len(coef)={len(coef)}, terms={basis_values.shape[1]}"
        )

    phi_basis = term_coefs * term_values
    feature_contrib = np.zeros(n_features, dtype=float)
    basis = list(earth_model.basis_)
    if len(basis) != len(phi_basis):
        basis = basis[-len(phi_basis) :]

    for phi_j, bfun in zip(phi_basis, basis):
        if hasattr(bfun, "is_pruned") and bfun.is_pruned():
            continue
        vars_idx = _basis_variables(bfun)
        if vars_idx is None:
            continue
        share = float(phi_j) / len(vars_idx)
        for var in vars_idx:
            if 0 <= var < n_features:
                feature_contrib[var] += share

    return feature_contrib


def mars_segment_weights(data: np.ndarray, y: np.ndarray):
    surrogate = Earth(max_degree=1, feature_importance_type="gcv")
    surrogate.fit(data, y)
    original_mask = np.ones(data.shape[1], dtype=float)
    segment_weights = mars_feature_contributions(surrogate, original_mask, data.shape[1])
    return segment_weights, surrogate.score(data, y)


def run_custom_explanation(
    image: np.ndarray,
    segments: np.ndarray,
    predict_fn,
    target_class: int,
    num_samples: int,
    kernel_width: float,
    perturbation: str,
    surrogate_type: str,
    sigma_noise: float,
):
    data, y, weights = make_transfer_dataset(
        image=image,
        segments=segments,
        predict_fn=predict_fn,
        target_class=target_class,
        num_samples=num_samples,
        kernel_width=kernel_width,
        perturbation=perturbation,
        sigma_noise=sigma_noise,
    )

    if surrogate_type == "ridge":
        segment_weights, score = ridge_segment_weights(data, y, weights)
    elif surrogate_type == "mars":
        segment_weights, score = mars_segment_weights(data, y)
    else:
        raise ValueError(f"Unknown surrogate_type: {surrogate_type}")

    return {
        "segments": segments,
        "segment_weights": segment_weights,
        "score": score,
        "target_class": target_class,
        "perturbation": perturbation,
        "surrogate_type": surrogate_type,
    }


def standard_lime_ridge(
    image: np.ndarray,
    predict_fn,
    target_class: int,
    num_samples: int,
):
    explainer = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        image,
        predict_fn,
        labels=(target_class,),
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=segmentation_fn,
    )
    segments = explanation.segments
    local_exp = dict(explanation.local_exp[target_class])
    segment_weights = np.zeros(len(np.unique(segments)), dtype=float)
    for segment_id, weight in local_exp.items():
        if 0 <= segment_id < len(segment_weights):
            segment_weights[segment_id] = weight
    return {
        "segments": segments,
        "segment_weights": segment_weights,
        "score": explanation.score,
        "target_class": target_class,
        "perturbation": "standard",
        "surrogate_type": "lime_default_ridge",
    }


def build_signed_maps(segments: np.ndarray, segment_weights: np.ndarray):
    positive_map = np.zeros(segments.shape, dtype=np.float32)
    negative_map = np.zeros(segments.shape, dtype=np.float32)
    for i, seg_val in enumerate(np.unique(segments)):
        weight = segment_weights[i]
        if weight > 0:
            positive_map[segments == seg_val] = weight
        elif weight < 0:
            negative_map[segments == seg_val] = abs(weight)
    if positive_map.max() > 0:
        positive_map /= positive_map.max()
    if negative_map.max() > 0:
        negative_map /= negative_map.max()
    return positive_map, negative_map


def plot_four_way(image: np.ndarray, results: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()
    for ax, result in zip(axes, results):
        pos_map, neg_map = build_signed_maps(result["segments"], result["segment_weights"])
        ax.imshow(image)
        ax.imshow(pos_map, cmap="Greens", alpha=0.55)
        ax.imshow(neg_map, cmap="Reds", alpha=0.55)
        ax.set_title(f"{result['name']}\nscore={result['score']:.3f}")
        ax.axis("off")

    legend_handles = [
        Patch(facecolor=(0, 1, 0, 0.55), edgecolor="none", label="Positive / supports class"),
        Patch(facecolor=(1, 0, 0, 0.55), edgecolor="none", label="Negative / opposes class"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--checkpoint", default="stl10_best_model.pt")
    parser.add_argument("--output", default="outputs/lime_mars_four_way.png")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--sigma-noise", type=float, default=35.0)
    parser.add_argument("--desired-label", type=int, default=8, help="Default: ship")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)
    predict_fn = make_predict_fn(model, device)
    test_dataset = load_test_dataset(args.data_root)
    DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    example_idx, image, true_label, probs = find_correct_example(
        test_dataset,
        predict_fn,
        desired_label=args.desired_label,
    )
    target_class = int(np.argmax(probs))
    segments = segmentation_fn(image)
    kernel_width = default_kernel_width(len(np.unique(segments)))

    results = []
    result = standard_lime_ridge(image, predict_fn, target_class, args.num_samples)
    result["name"] = "1. Standard LIME (Ridge)"
    results.append(result)

    result = run_custom_explanation(
        image,
        segments,
        predict_fn,
        target_class,
        args.num_samples,
        kernel_width,
        perturbation="standard",
        surrogate_type="mars",
        sigma_noise=args.sigma_noise,
    )
    result["name"] = "2. Standard LIME + MARS"
    results.append(result)

    result = run_custom_explanation(
        image,
        segments,
        predict_fn,
        target_class,
        args.num_samples,
        kernel_width,
        perturbation="gaussian",
        surrogate_type="mars",
        sigma_noise=args.sigma_noise,
    )
    result["name"] = "3. Gaussian-noise LIME + MARS"
    results.append(result)

    result = run_custom_explanation(
        image,
        segments,
        predict_fn,
        target_class,
        args.num_samples,
        kernel_width,
        perturbation="gaussian",
        surrogate_type="ridge",
        sigma_noise=args.sigma_noise,
    )
    result["name"] = "4. Gaussian-noise LIME (Ridge)"
    results.append(result)

    output_path = Path(args.output)
    plot_four_way(image, results, output_path)

    print(f"Example index: {example_idx}")
    print(f"True class: {CLASS_NAMES[true_label]}")
    print(f"Target/predicted class: {CLASS_NAMES[target_class]} ({probs[target_class]:.4f})")
    print(f"Segments: {len(np.unique(segments))}")
    print(f"Samples per method: {args.num_samples}")
    print(f"Kernel width: {kernel_width:.3f}")
    print(f"Gaussian sigma: {args.sigma_noise}")
    for result in results:
        print(f"{result['name']}: local score={result['score']:.4f}")
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
