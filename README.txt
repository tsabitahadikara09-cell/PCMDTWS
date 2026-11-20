FISHnDISH - Distance Transform & Watershed Segmentation

1. Install Python (recommended 3.10-3.12).
2. Open terminal/cmd in this project folder.
3. Install dependencies:
   pip install -r requirements.txt
4. Run:
   streamlit run app.py
5. Upload a JPG/PNG image from the images folder, then adjust parameters in the sidebar.

Features:
- Grayscale or color-channel selection
- Otsu/manual thresholding
- Morphological opening and closing
- Distance Transform L1 (Manhattan) / L2 (Euclidean)
- Marker generation using h-maxima (peaks in distance map)
- Watershed segmentation and boundary overlay
- Parameter comparison experiment panel
- Download binary mask, distance transform, boundary, and labels

Note: The h-maxima method is used to find peaks in the distance map (object centers). This is the peak-based counterpart to minima detection on an inverted distance map.
