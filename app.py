import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from skimage import filters, morphology, segmentation, measure, color, util
from scipy import ndimage as ndi
import io

st.set_page_config(page_title="Cell Segmentation Lab", page_icon="🔬", layout="wide")

# -------------------- Styling --------------------
st.markdown("""
<style>
/* Global light palette: explicit colors prevent washed-out Streamlit text. */
:root {
    color-scheme: light;
    --text-color: #182230;
    --primary-color: #d94b58;
    --background-color: #f4f7fb;
    --secondary-background-color: #ffffff;
}
.stApp {
    background: #f4f7fb !important;
    color: #182230 !important;
}
[data-testid="stMain"] {
    background: #f4f7fb !important;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 1.45rem !important;
    padding-bottom: 2.4rem !important;
    max-width: 1540px !important;
}

/* Main-area typography */
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {
    color: #172b45 !important;
    opacity: 1 !important;
}
[data-testid="stMain"] p,
[data-testid="stMain"] li,
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] {
    color: #344054;
    opacity: 1;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
    color: #667085 !important;
    opacity: 1 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #101b2b !important;
    border-right: 1px solid #23344d;
}
[data-testid="stSidebar"] * {
    color: #edf3fb !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #dce7f5 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    color: #182230 !important;
}

/* Header */
.hero {
    background: linear-gradient(115deg, #14243d 0%, #24547b 100%);
    padding: 1.45rem 1.65rem;
    border-radius: 18px;
    color: #ffffff !important;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 22px rgba(18, 42, 70, .12);
}
.hero h1 {
    color: #ffffff !important;
    margin: 0;
    font-size: 1.9rem;
    line-height: 1.25;
}
.hero p {
    color: #e4edf7 !important;
    margin: .5rem 0 0;
    font-size: .98rem;
}

/* Metric cards: label and number are forced to high-contrast dark colors. */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #dce4ef !important;
    border-radius: 14px !important;
    padding: .95rem 1rem !important;
    box-shadow: 0 2px 7px rgba(16, 37, 63, .035);
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
    color: #53657c !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
    color: #172b45 !important;
    opacity: 1 !important;
    font-weight: 750 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] * {
    color: #53657c !important;
    opacity: 1 !important;
}

/* Tabs: inactive tabs stay readable, active tab gets a clear accent. */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #d8e0eb;
}
.stTabs [data-baseweb="tab"] {
    color: #40536b !important;
    background: #e9eef5 !important;
    border-radius: 10px 10px 0 0 !important;
    padding: .65rem 1rem !important;
    font-weight: 650 !important;
    opacity: 1 !important;
}
.stTabs [data-baseweb="tab"] * {
    color: inherit !important;
    opacity: 1 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
    background: #1d4e79 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background: #d94b58 !important;
}

/* Buttons and download buttons */
[data-testid="stDownloadButton"] button,
[data-testid="stButton"] button,
.stDownloadButton button,
.stButton button {
    background: #1f527d !important;
    color: #ffffff !important;
    border: 1px solid #1f527d !important;
    border-radius: 9px !important;
    min-height: 2.7rem;
    font-weight: 650 !important;
    opacity: 1 !important;
    box-shadow: 0 2px 5px rgba(24, 61, 95, .12);
}
[data-testid="stDownloadButton"] button *,
[data-testid="stButton"] button *,
.stDownloadButton button *,
.stButton button * {
    color: #ffffff !important;
    fill: #ffffff !important;
    opacity: 1 !important;
}
[data-testid="stDownloadButton"] button:hover,
[data-testid="stButton"] button:hover,
.stDownloadButton button:hover,
.stButton button:hover {
    background: #173e60 !important;
    border-color: #173e60 !important;
}

/* Inputs and select boxes in main content */
[data-testid="stMain"] input,
[data-testid="stMain"] textarea {
    color: #182230 !important;
}
[data-testid="stMain"] [data-baseweb="select"] * {
    color: #182230 !important;
}

/* Export cards */
.download-card {
    background: #ffffff;
    border: 1px solid #dce4ef;
    border-radius: 13px;
    padding: 1rem 1.1rem;
    min-height: 112px;
    margin-bottom: .75rem;
}
.download-card h4 {
    color: #172b45 !important;
    margin: 0 0 .35rem 0;
    font-size: 1rem;
}
.download-card p {
    color: #667085 !important;
    margin: 0;
    font-size: .88rem;
}

/* Dataframes and alerts */
[data-testid="stDataFrame"] {
    border: 1px solid #dce4ef;
    border-radius: 10px;
    overflow: hidden;
}
[data-testid="stAlert"] p {
    color: #26384d !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------- Helpers --------------------
@st.cache_data(show_spinner=False)
def decode_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(image)


def get_channel(rgb, channel):
    if channel == "Grayscale":
        return color.rgb2gray(rgb)
    if channel == "Red":
        return rgb[:, :, 0] / 255.0
    if channel == "Green":
        return rgb[:, :, 1] / 255.0
    if channel == "Blue":
        return rgb[:, :, 2] / 255.0
    # HSV saturation can be helpful when cells have color contrast
    return color.rgb2hsv(rgb)[:, :, 1]


def make_mask(gray, threshold_mode, threshold_value, polarity, morph_size):
    if threshold_mode == "Otsu otomatis":
        threshold = filters.threshold_otsu(gray)
    else:
        threshold = threshold_value / 255.0
    mask = gray > threshold
    if polarity == "Objek gelap":
        mask = ~mask
    elif polarity == "Auto (area foreground)":
        # Keep the less dominant side as the candidate foreground
        if mask.mean() > 0.5:
            mask = ~mask
    if morph_size > 0:
        footprint = morphology.disk(max(1, morph_size))
        mask = morphology.opening(mask, footprint)
        mask = morphology.closing(mask, footprint)
    return mask, float(threshold)


def distance_map(mask, metric):
    if metric == "L2 (Euclidean)":
        return ndi.distance_transform_edt(mask)
    # Manhattan distance computed efficiently with SciPy's taxicab transform
    return ndi.distance_transform_cdt(mask, metric="taxicab").astype(np.float32)


def watershed_run(rgb, mask, metric, h_value, min_area):
    dist = distance_map(mask, metric)
    if h_value <= 0:
        peaks = morphology.local_maxima(dist) & mask
    else:
        peaks = morphology.h_maxima(dist, h=h_value) & mask
    markers, marker_count = ndi.label(peaks)
    if marker_count == 0:
        markers, marker_count = ndi.label(morphology.local_maxima(dist) & mask)
    labels = segmentation.watershed(-dist, markers, mask=mask)
    if min_area > 1 and labels.max() > 0:
        counts = np.bincount(labels.ravel())
        keep = counts >= min_area
        keep[0] = True
        labels[~keep[labels]] = 0
        labels = measure.label(labels > 0)
    boundaries = segmentation.find_boundaries(labels, mode="inner")
    overlay = rgb.copy()
    overlay[boundaries] = [255, 60, 60]
    colored = color.label2rgb(labels, image=rgb, bg_label=0, alpha=0.5)
    return dist, markers, labels, boundaries, overlay, colored, int(marker_count)


def figure_grid(images, titles, cmaps=None, cols=3, figsize=None):
    n = len(images)
    rows = int(np.ceil(n / cols))
    if figsize is None:
        figsize = (cols * 4.2, rows * 3.3)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    cmaps = cmaps or [None] * n
    for ax, im, title, cmap_name in zip(axes, images, titles, cmaps):
        ax.imshow(im, cmap=cmap_name)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def png_bytes(arr, is_rgb=False):
    if is_rgb:
        img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
    else:
        arr = np.asarray(arr)
        if arr.dtype == bool:
            arr = arr.astype(np.uint8) * 255
        elif arr.dtype != np.uint8:
            arr = util.img_as_ubyte(arr / (arr.max() if arr.max() else 1))
        img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# -------------------- Sidebar controls --------------------
st.sidebar.markdown("## 🔬 Segmentation Controls")
st.sidebar.caption("Upload citra dan atur parameter pemrosesan")
uploaded = st.sidebar.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
channel = st.sidebar.selectbox("Kanal / warna", ["Grayscale", "Blue", "Green", "Red", "HSV Saturation"], index=0)
th_mode = st.sidebar.radio("Threshold", ["Otsu otomatis", "Manual"], horizontal=True)
th_val = st.sidebar.slider("Threshold manual", 0, 255, 110, disabled=th_mode != "Manual")
polarity = st.sidebar.selectbox("Polaritas objek", ["Auto (area foreground)", "Objek terang", "Objek gelap"], index=0)
morph_size = st.sidebar.slider("Kernel morfologi (radius)", 0, 7, 1)
metric = st.sidebar.selectbox("Distance metric", ["L2 (Euclidean)", "L1 (Manhattan)"])
h_value = st.sidebar.slider("h untuk marker (h-maxima)", 0.0, 12.0, 2.0, 0.5)
min_area = st.sidebar.slider("Minimum area region (px)", 0, 1500, 30, 10)

st.markdown("<div class='hero'><h1>Distance Transform & Watershed</h1><p>Cell / object segmentation • preprocessing • marker generation • parameter experiments</p></div>", unsafe_allow_html=True)

if uploaded is None:
    st.info("Upload citra melalui panel kiri untuk memulai proses segmentasi.")
    st.markdown("#### Pipeline analisis")
    st.markdown("**Image input → Preprocessing → Binary mask → Distance transform → Marker → Watershed → Boundary overlay**")
    st.stop()


try:
    rgb = decode_image(uploaded.getvalue())
except Exception as e:
    st.error(f"File gambar tidak dapat dibaca: {e}")
    st.stop()

channel_img = get_channel(rgb, channel)
mask, threshold_used = make_mask(channel_img, th_mode, th_val, polarity, morph_size)
dist, markers, labels, boundaries, overlay, colored, marker_count = watershed_run(rgb, mask, metric, h_value, min_area)

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Marker detected", marker_count)
c2.metric("Final regions", int(labels.max()))
c3.metric("Foreground pixels", f"{int(mask.sum()):,}")
c4.metric("Threshold used", f"{threshold_used:.3f}")

# Main tabs
result_tab, experiment_tab, download_tab = st.tabs(["📊 Results", "🧪 Experiments", "⬇️ Export"])
with result_tab:
    st.markdown("### Processing overview")
    images = [rgb, channel_img, mask, dist, markers, overlay]
    titles = ["Original image", f"Selected channel: {channel}", "Binary mask", "Distance transform", "Markers", "Boundary overlay"]
    cmaps = [None, "gray", "gray", "magma", "nipy_spectral", None]
    fig = figure_grid(images, titles, cmaps, cols=3, figsize=(15, 8))
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Watershed label map")
        st.image(colored, use_container_width=True)
    with col_b:
        st.markdown("### Final boundary on original image")
        st.image(overlay, use_container_width=True)
    st.caption("Marker berasal dari h-maxima pada distance map. Nilai h yang lebih tinggi umumnya mengurangi marker kecil, tetapi dapat membuat objek berdekatan tidak terpisah.")

with experiment_tab:
    st.markdown("### Parameter comparison")
    st.write("Bandingkan kombinasi h dan distance metric. Gunakan visual boundary untuk menentukan hasil terbaik, bukan hanya jumlah region.")
    h_values = st.multiselect("Pilih nilai h", [0, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10], default=[1, 2, 4])
    metrics = st.multiselect("Pilih distance metric", ["L1 (Manhattan)", "L2 (Euclidean)"], default=["L1 (Manhattan)", "L2 (Euclidean)"])
    if st.button("Run experiments", type="primary"):
        if not h_values or not metrics:
            st.warning("Pilih minimal satu nilai h dan satu distance metric.")
        else:
            rows = []
            comparison = []
            for m in metrics:
                for hv in h_values:
                    d, mk, lab, bd, ov, col, mc = watershed_run(rgb, mask, m, float(hv), min_area)
                    rows.append({"Distance metric": m, "h": hv, "Markers": mc, "Final regions": int(lab.max())})
                    comparison.append((ov, f"{m} | h={hv} | regions={int(lab.max())}"))
            st.dataframe(rows, use_container_width=True, hide_index=True)
            for i in range(0, len(comparison), 2):
                cols = st.columns(2)
                for j, col_slot in enumerate(cols):
                    if i+j < len(comparison):
                        im, title = comparison[i+j]
                        col_slot.image(im, caption=title, use_container_width=True)
            st.info("Interpretasi: marker terlalu banyak dapat menyebabkan over-segmentation, sedangkan marker terlalu sedikit dapat menyebabkan beberapa objek bergabung. Pilih parameter berdasarkan boundary yang paling sesuai dengan objek.")

with download_tab:
    st.markdown("### Download hasil segmentasi")
    st.markdown(
        "Simpan setiap tahap pemrosesan untuk dokumentasi, laporan, atau analisis lanjutan. "
        "Tombol unduh di bawah akan menghasilkan file PNG.")

    norm_dist = (dist / dist.max() * 255).astype(np.uint8) if dist.max() > 0 else np.zeros_like(dist, dtype=np.uint8)
    label_preview = labels.astype(np.uint16)

    export_items = [
        ("Binary mask", "Mask hitam-putih hasil thresholding dan morfologi.",
         png_bytes(mask), "binary_mask.png"),
        ("Distance transform", "Peta jarak foreground terhadap background terdekat.",
         png_bytes(norm_dist), "distance_transform.png"),
        ("Boundary overlay", "Garis batas watershed yang ditumpangkan pada citra asli.",
         png_bytes(overlay, is_rgb=True), "watershed_boundary.png"),
        ("Watershed label map", "Setiap region memiliki label segmentasi tersendiri.",
         png_bytes(label_preview), "watershed_labels.png"),
    ]

    export_cols = st.columns(2, gap="large")
    for idx, (title, description, data, filename) in enumerate(export_items):
        with export_cols[idx % 2]:
            st.markdown(
                f'<div class="download-card"><h4>{title}</h4><p>{description}</p></div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                label=f"⬇️ Download {title}",
                data=data,
                file_name=filename,
                mime="image/png",
                use_container_width=True,
                key=f"download_{filename}",
            )
