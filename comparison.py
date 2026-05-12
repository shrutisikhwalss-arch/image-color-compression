import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from compression import compress_with_kmeans, get_image_size_kb


def compress_jpeg(image_array, quality=80):
    """Compress image using JPEG at given quality."""
    buffer = io.BytesIO()
    img = Image.fromarray(image_array)
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer).convert("RGB"))


def compress_webp(image_array, quality=80):
    """Compress image using WebP at given quality."""
    buffer = io.BytesIO()
    img = Image.fromarray(image_array)
    img.save(buffer, format="WEBP", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer).convert("RGB"))


def compress_pil_quantize(image_array, n_colors=16):
    """Compress image using PIL's built-in quantization (Median Cut)."""
    img = Image.fromarray(image_array)
    quantized = img.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    return np.array(quantized.convert("RGB"))


def get_size_kb(image_array, format="PNG", quality=80):
    """Get in-memory size of image in KB."""
    buffer = io.BytesIO()
    img = Image.fromarray(image_array.astype(np.uint8))
    if format in ("JPEG", "WEBP"):
        img.save(buffer, format=format, quality=quality)
    else:
        img.save(buffer, format=format)
    return buffer.tell() / 1024


def run_comparison(image_array):
    """
    Run all compression methods and return a results dictionary.
    """
    original_kb = get_size_kb(image_array, format="PNG")

    methods = {
        "Original (PNG)":       {"array": image_array,                          "kb": original_kb},
        "KMeans 16 colors":     {"array": compress_with_kmeans(image_array, 16), "kb": None},
        "JPEG (q=80)":          {"array": compress_jpeg(image_array, 80),        "kb": None},
        "JPEG (q=50)":          {"array": compress_jpeg(image_array, 50),        "kb": None},
        "WebP (q=80)":          {"array": compress_webp(image_array, 80),        "kb": None},
        "PIL Quantize 16":      {"array": compress_pil_quantize(image_array, 16),"kb": None},
    }

    fmt_map = {
        "Original (PNG)":   "PNG",
        "KMeans 16 colors": "PNG",
        "JPEG (q=80)":      "JPEG",
        "JPEG (q=50)":      "JPEG",
        "WebP (q=80)":      "WEBP",
        "PIL Quantize 16":  "PNG",
    }

    for name, info in methods.items():
        if info["kb"] is None:
            info["kb"] = get_size_kb(info["array"], format=fmt_map[name])
        info["reduction"] = (1 - info["kb"] / original_kb) * 100

    return methods


def plot_comparison_chart(methods):
    """
    Plot a bar chart comparing file sizes and reductions.
    Returns a matplotlib figure.
    """
    names = list(methods.keys())
    sizes = [m["kb"] for m in methods.values()]
    reductions = [m["reduction"] for m in methods.values()]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Compression Method Comparison", fontsize=16, fontweight="bold")

    # File size bar chart
    bars = axes[0].barh(names, sizes, color=colors)
    axes[0].set_xlabel("File Size (KB)")
    axes[0].set_title("File Size by Method")
    for bar, size in zip(bars, sizes):
        axes[0].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{size:.1f} KB", va="center", fontsize=9)

    # Reduction bar chart
    bars2 = axes[1].barh(names, reductions, color=colors)
    axes[1].set_xlabel("Size Reduction (%)")
    axes[1].set_title("Size Reduction vs Original")
    for bar, red in zip(bars2, reductions):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{red:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_visual_comparison(methods):
    """
    Plot all compressed images side by side.
    Returns a matplotlib figure.
    """
    names = list(methods.keys())
    n = len(names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, (name, info) in enumerate(methods.items()):
        axes[i].imshow(info["array"])
        axes[i].set_title(f"{name}\n{info['kb']:.1f} KB  ({info['reduction']:.1f}% smaller)",
                          fontsize=10)
        axes[i].axis("off")

    plt.suptitle("Visual Quality Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig