import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure, segmentation, color
from scipy import ndimage as ndi
from PIL import Image
import io

st.set_page_config(page_title='Distance Transform & Watershed', layout='wide')
st.title('Distance Transform & Watershed Segmentation')
st.caption('Preprocessing • Distance Transform • Marker Generation • Watershed • Parameter Experiments')

@st.cache_data

def load_image_bytes(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess(rgb, channel, threshold_mode, threshold_value, morph_size):
    if channel == 'Grayscale':
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    elif channel == 'Blue channel':
        gray = rgb[:, :, 2]
    elif channel == 'Green channel':
        gray = rgb[:, :, 1]
    else:
        gray = rgb[:, :, 0]

    if threshold_mode == 'Otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Pilih otomatis polaritas objek/background berdasarkan area foreground
    if np.mean(binary > 0) > 0.5:
        binary = cv2.bitwise_not(binary)
    mask = binary > 0

    if morph_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = binary > 0
    return gray, mask



def segment(rgb, mask, distance_metric, h_value, min_area):
    # Distance Transform dihitung pada foreground mask
    dist_type = cv2.DIST_L2 if distance_metric == 'L2 (Euclidean)' else cv2.DIST_L1
    distance = cv2.distanceTransform(mask.astype(np.uint8), dist_type, 3 if dist_type == cv2.DIST_L2 else 1)


    # Marker: h-maxima pada distance map (ekuivalen mencari puncak lokal yang cukup menonjol)
    if h_value <= 0:
        peaks = morphology.local_maxima(distance)
    else:
        peaks = morphology.h_maxima(distance, h=h_value)
    peaks &= mask
    markers, n_markers = ndi.label(peaks)


    # Jika marker terlalu sedikit, fallback ke local maxima
    if n_markers == 0:
        peaks = morphology.local_maxima(distance) & mask
        markers, n_markers = ndi.label(peaks)
    labels = segmentation.watershed(-distance, markers, mask=mask)


    # Hilangkan region kecil dan relabel
    if min_area > 1:
        sizes = np.bincount(labels.ravel())
        keep = np.zeros_like(sizes, dtype=bool)
        keep[0] = True
        keep[sizes >= min_area] = True
        labels[~keep[labels]] = 0
        labels = measure.label(labels > 0)

    boundaries = segmentation.find_boundaries(labels, mode='inner')
    overlay = rgb.copy()
    overlay[boundaries] = [255, 0, 0]
    return distance, markers, labels, boundaries, overlay, n_markers



def plot_images(images, titles, cmap_list=None, figsize=(16, 5)):
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    for ax, im, title, cmap in zip(axes, images, titles, cmap_list or [None]*len(images)):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    return fig



with st.sidebar:
    st.header('Input & Parameter')
    uploaded = st.file_uploader('Upload citra (JPG/PNG)', type=['jpg','jpeg','png'])
    channel = st.selectbox('Grayscale / kanal warna', ['Grayscale', 'Blue channel', 'Green channel', 'Red channel'], index=1)
    threshold_mode = st.radio('Thresholding', ['Otsu', 'Manual'], horizontal=True)
    threshold_value = st.slider('Threshold manual', 0, 255, 80, disabled=threshold_mode == 'Otsu')
    morph_size = st.slider('Ukuran kernel morfologi', 0, 15, 3, step=2)
    distance_metric = st.selectbox('Distance metric', ['L2 (Euclidean)', 'L1 (Manhattan)'])
    h_value = st.slider('h untuk marker (h-maxima)', 0.0, 20.0, 2.0, 0.5)
    min_area = st.slider('Minimum luas region (pixel)', 1, 1000, 80, 10)

if uploaded is None:
    st.info('Upload salah satu citra dari dataset FISHnDISH untuk mulai segmentasi.')
    st.stop()


rgb = load_image_bytes(uploaded.getvalue())
gray, mask = preprocess(rgb, channel, threshold_mode, threshold_value, morph_size)
distance, markers, labels, boundaries, overlay, marker_count = segment(rgb, mask, distance_metric, h_value, min_area)


st.subheader('Hasil Proses')
fig = plot_images([rgb, gray, mask, distance, markers, overlay],
                 ['Original Image', 'Selected Channel', 'Binary Mask', 'Distance Transform', 'Markers', 'Watershed Boundary'],
                 [None, 'gray', 'gray', 'magma', 'nipy_spectral', None], figsize=(18, 5))
st.pyplot(fig)
plt.close(fig)
st.metric('Marker awal terdeteksi', marker_count)
st.metric('Objek / region akhir', int(labels.max()))


st.subheader('Visualisasi Label Watershed')
colored_labels = color.label2rgb(labels, image=rgb, bg_label=0, alpha=0.45)
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.imshow(colored_labels)
ax2.set_title('Label hasil Watershed')
ax2.axis('off')
st.pyplot(fig2)
plt.close(fig2)


# Tuning eksperimen parameter h dan distance metric
with st.expander('Eksperimen & Analisis parameter (h-minima / distance metric)', expanded=False):
    h_values = st.multiselect('Nilai h yang dibandingkan', [0, 1, 2, 3, 4, 5, 7, 10, 15], default=[1, 2, 4, 7])
    metrics = st.multiselect('Distance metric yang dibandingkan', ['L1 (Manhattan)', 'L2 (Euclidean)'], default=['L1 (Manhattan)', 'L2 (Euclidean)'])
    if st.button('Jalankan eksperimen'):
        if not h_values or not metrics:
            st.warning('Pilih minimal satu nilai h dan satu distance metric.')
        else:
            records = []
            comparison_imgs = []
            comparison_titles = []
            for metric in metrics:
                for hv in h_values:
                    d, m, lab, bd, ov, mc = segment(rgb, mask, metric, float(hv), min_area)
                    records.append({'Distance': metric, 'h': hv, 'Marker count': mc, 'Final regions': int(lab.max())})
                    comparison_imgs.append(ov)
                    comparison_titles.append(f'{metric} | h={hv} | regions={int(lab.max())}')
            st.dataframe(records, use_container_width=True)
            cols = st.columns(2)
            for idx, (im, title) in enumerate(zip(comparison_imgs, comparison_titles)):
                with cols[idx % 2]:
                    st.image(im, caption=title, use_container_width=True)
            st.caption('Bandingkan jumlah region dan tampilan boundary. Jumlah region terbanyak tidak selalu berarti segmentasi terbaik; periksa apakah setiap objek terpisah tanpa pecah berlebihan.')


# Download outputs
st.subheader('Download hasil')
for name, array in [('binary_mask.png', (mask.astype(np.uint8)*255)), ('distance_transform.png', distance), ('watershed_boundary.png', overlay), ('watershed_labels.png', labels.astype(np.uint16))]:
    if name == 'watershed_boundary.png':
        save_arr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    elif name == 'distance_transform.png':
        save_arr = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        save_arr = array
    ok, enc = cv2.imencode('.png', save_arr)
    if ok:
        st.download_button(f'Download {name}', enc.tobytes(), file_name=name, mime='image/png')
