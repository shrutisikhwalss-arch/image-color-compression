import io
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from img_compression import load_image, compress_with_kmeans, get_size_reduction
from img_comparison import run_comparison, plot_comparison_chart, plot_visual_comparison

import sys
import os
sys.path.append(os.path.dirname(__file__))  

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Color Compressor",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Image Color Compressor")
st.markdown(
    "Compress images using **KMeans color quantization**. "
    )
st.divider()

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "webp"]
    )

    use_sample = st.checkbox("Use sample image instead", value=not bool(uploaded_file))

    n_colors = st.slider(
        "Number of colors", min_value=2, max_value=64, value=16, step=2
    )

    show_comparison = st.checkbox("Show full method comparison", value=False)


# ── Load image ─────────────────────────────────────────────────────────────────
image_array = None

if uploaded_file:
    image_array = load_image(uploaded_file)
elif use_sample:
    try:
        from sklearn.datasets import load_sample_image
        image_array = load_sample_image("flower.jpg")
    except Exception:
        st.warning("Sample image not found. Please upload an image.")

# ── Main content ───────────────────────────────────────────────────────────────
if image_array is not None:

    # ── Compress ───────────────────────────────────────────────────────────────
    with st.spinner("Compressing image..."):
        compressed = compress_with_kmeans(image_array, n_colors)
        orig_kb, comp_kb, reduction = get_size_reduction(image_array, compressed)

    # ── Metrics ────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original Size",    f"{orig_kb:.1f} KB")
    col2.metric("Compressed Size",  f"{comp_kb:.1f} KB")
    col3.metric("Size Reduction",   f"{reduction:.1f}%")
    col4.metric("Colors Used",      f"{n_colors}")

    st.divider()

    # ── Before / After ─────────────────────────────────────────────────────────
    st.subheader("Before vs After")
    left, right = st.columns(2)

    left.image(image_array,  caption="Original Image",             use_container_width=True)
    right.image(compressed,  caption=f"Compressed ({n_colors} colors)", use_container_width=True)

    # ── Download button ────────────────────────────────────────────────────────
    st.divider()
    buf = io.BytesIO()
    Image.fromarray(compressed).save(buf, format="PNG")
    st.download_button(
        label="Download Compressed Image",
        data=buf.getvalue(),
        file_name=f"compressed_{n_colors}colors.png",
        mime="image/png",
    )

    # ── Color scatter plot ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Color Space Visualization")

    data_flat = (image_array / 255.0).reshape(-1, 3)
    comp_flat = (compressed / 255.0).reshape(-1, 3)

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(data_flat))[:5000]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Color Distribution: Original vs Compressed", fontsize=14)

    axes[0].scatter(data_flat[idx, 0], data_flat[idx, 1],
                    color=data_flat[idx], marker=".", s=5)
    axes[0].set(xlabel="Red", ylabel="Green", title="Original", xlim=(0,1), ylim=(0,1))

    axes[1].scatter(comp_flat[idx, 0], comp_flat[idx, 1],
                    color=comp_flat[idx], marker=".", s=5)
    axes[1].set(xlabel="Red", ylabel="Green", title=f"Compressed ({n_colors} colors)",
                xlim=(0,1), ylim=(0,1))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Full comparison study ──────────────────────────────────────────────────
    if show_comparison:
        st.divider()
        st.subheader("Full Compression Method Comparison")
        st.info("Comparing KMeans, JPEG, WebP, and PIL Quantize on your image.")

        with st.spinner("Running all compression methods..."):
            results = run_comparison(image_array)

        # Chart
        chart_fig = plot_comparison_chart(results)
        st.pyplot(chart_fig)
        plt.close()

        # Results table
        st.subheader("Results Table")
        rows = {
            name: {
                "Size (KB)": f"{info['kb']:.1f}",
                "Reduction (%)": f"{info['reduction']:.1f}%",
            }
            for name, info in results.items()
        }
        st.table(rows)

        # Visual comparison
        st.subheader("Visual Quality Comparison")
        visual_fig = plot_visual_comparison(results)
        st.pyplot(visual_fig)
        plt.close()

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.info("Upload an image from the sidebar or enable the sample image to get started.")
    st.markdown(
        """
        ### What you can do:
        - Upload any JPG, PNG, or WebP image
        - Adjust the number of colors using the slider
        - Compare compression methods side by side
        - ⬇Download the compressed result
        """
    )