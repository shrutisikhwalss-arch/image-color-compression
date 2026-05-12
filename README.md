# 🎨 Image Color Compressor

An interactive web app that compresses images using **KMeans color quantization** — reducing millions of possible colors down to just a handful, while preserving visual structure.

Built with Python, Scikit-Learn, and Streamlit.

---

## 🖼️ Demo

| Original | 16 Colors | 8 Colors |
|----------|-----------|----------|
| ~890 KB  | ~78 KB    | ~45 KB   |

> Upload any image and watch it compress in real time.

---

## ✨ Features

- 📤 Upload your own image (JPG, PNG, WebP)
- 🎛️ Adjust number of colors with a live slider (2–64)
- 📊 Before/after comparison with file size metrics
- 🎯 Color space scatter plot visualization
- 📉 Full compression method comparison (KMeans vs JPEG vs WebP vs PIL)
- ⬇️ Download the compressed result

---

## 🧠 How It Works

1. Each pixel in the image is a point in 3D color space (R, G, B)
2. **MiniBatchKMeans** clusters all pixels into N groups
3. Every pixel is replaced by its cluster's center color
4. Result: the image now uses only N unique colors
5. Fewer unique colors = smaller file size when saved

---

## 📊 Compression Method Comparison

| Method          | Avg Size | Reduction | Quality |
|-----------------|----------|-----------|---------|
| Original PNG    | 890 KB   | 0%        | Perfect |
| KMeans 16 colors| 78 KB    | 91%       | Good    |
| JPEG (q=80)     | 45 KB    | 95%       | Great   |
| WebP (q=80)     | 32 KB    | 96%       | Great   |
| PIL Quantize 16 | 21 KB    | 98%       | Good    |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/image-color-compression.git
cd image-color-compression
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
image-color-compression/
│
├── app.py              ← Streamlit web app (main entry point)
├── compression.py      ← KMeans compression logic
├── comparison.py       ← Multi-method comparison study
├── requirements.txt    ← Python dependencies
└── samples/
    └── flower.jpg      ← Sample image for testing
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Scikit-Learn** — MiniBatchKMeans clustering
- **Streamlit** — Interactive web interface
- **Pillow** — Image I/O and format conversion
- **Matplotlib** — Visualization and scatter plots
- **NumPy** — Array operations

---

## 📖 What I Learned

- How KMeans can be applied to non-traditional problems like image processing
- The difference between lossy (JPEG) and lossless (PNG) compression
- Why chroma subsampling works — human eyes are more sensitive to brightness than color
- How different compression algorithms trade off quality vs file size
- Building and deploying interactive ML apps with Streamlit

---

## 🔮 Future Improvements

- [ ] Add JPEG 2000 (wavelet-based) compression
- [ ] Side-by-side SSIM (quality score) comparison
- [ ] Batch processing for multiple images
- [ ] Deploy to Streamlit Cloud

---

## 📄 License

MIT License — free to use and modify.