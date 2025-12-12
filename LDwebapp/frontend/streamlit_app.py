# frontend/streamlit_app.py
import streamlit as st
import requests
from PIL import Image
import io
import zipfile
from datetime import datetime

# === CONFIG ===
BACKEND_URL = "http://127.0.0.1:5000/predict"   # change if backend is deployed elsewhere
LOGS_URL = "http://127.0.0.1:5000/logs"        # optional admin logs endpoint
MAX_FILES = 5
TIMEOUT = 120  # seconds for backend requests

st.set_page_config(page_title="Leukemia Cell Detection", layout="wide")

# ---- Header ----
st.title("Leukemia Cell Detection")
st.subheader("Microscopy Image Segmentation and Classification")
st.markdown("FOR DEMONSTRATION PURPOSES ONLY. NOT FOR CLINICAL USE.")
st.markdown(
    "Upload 1–5 microscopy images. The backend will segment individual cells, "
    "classify each crop using a pretrained ResNet model, and return an annotated image "
    "plus per-cell probabilities."
)

# ---- Columns: left = instructions & sample, right = uploader & results ----
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Input guidelines")
    st.write(
        "- Preferred: cropped single-cell microscopic images (see sample).\n"
        "- The system can accept whole-smear images — it will segment and crop cells automatically, "
        "but results are best for single-cell crops.\n"
        f"- Upload up to **{MAX_FILES}** images per run.\n"
        "- Supported formats: JPG, JPEG, PNG, BMP."
    )

    # Show sample image (you should include sample_input.jpg in frontend folder)
    try:
        st.image("sample_input.jpg", caption="Sample: preferred single-cell crop", use_container_width=True)
    except Exception:
        st.info("Place a sample_input.jpg file in the frontend folder to show example input.")

    st.markdown("---")
    st.subheader("Run notes (optional)")
    user_notes = st.text_area(
        "Add any notes or comments for this run (saved locally in frontend only)",
        placeholder="e.g. sample ID, staining info, case notes..."
    )

    st.markdown("**Admin:**")
    if st.button("Download logs (admin)"):
        try:
            r = requests.get(LOGS_URL, timeout=10)
            if r.status_code == 200:
                st.download_button("Download CSV logs", r.content, file_name="predictions_log.csv")
            else:
                st.error("No logs available (backend returned non-200).")
        except Exception as e:
            st.error(f"Could not fetch logs: {e}")

with right_col:
    st.subheader("Upload images (1–5)")
    uploaded_files = st.file_uploader(
        "Choose microscopy images", 
        type=["jpg", "jpeg", "png", "bmp"], 
        accept_multiple_files=True
    )

    # Limit number of files
    if uploaded_files and len(uploaded_files) > MAX_FILES:
        st.error(f"Please upload at most {MAX_FILES} files. Showing first {MAX_FILES}.")
        uploaded_files = uploaded_files[:MAX_FILES]

    # Preview uploaded images
    if uploaded_files:
        preview_cols = st.columns(len(uploaded_files))
        for i, f in enumerate(uploaded_files):
            with preview_cols[i]:
                try:
                    img = Image.open(f)
                    st.image(img, use_container_width=True, caption=f.name)
                except Exception:
                    st.write(f.name)

    # Run button
    if uploaded_files and st.button("Run Detection"):
        # Prepare files for multipart upload
        files_payload = []
        for f in uploaded_files:
            content = f.getvalue()
            # guess mime type simply
            mime = "image/jpeg"
            if f.name.lower().endswith(".png"):
                mime = "image/png"
            elif f.name.lower().endswith(".bmp"):
                mime = "image/bmp"
            files_payload.append(("files", (f.name, content, mime)))

        # Send optional notes as form field
        data_payload = {"notes": user_notes} if user_notes else {}

        try:
            with st.spinner("Sending to backend and running segmentation + classification..."):
                resp = requests.post(BACKEND_URL, files=files_payload, data=data_payload, timeout=TIMEOUT)
        except requests.exceptions.RequestException as e:
            st.error(f"Request to backend failed: {e}")
            resp = None

        if resp is None:
            pass
        elif resp.status_code != 200:
            st.error(f"Backend error ({resp.status_code}): {resp.text}")
        else:
            try:
                result = resp.json()
            except Exception as e:
                st.error(f"Could not parse backend JSON response: {e}")
                result = None

            if result is not None:
                # Helper to fetch bytes for a returned URL or relative path
                def fetch_bytes(url):
                    if url.startswith("/"):
                        base = BACKEND_URL.replace("/predict", "")
                        full = base + url
                    else:
                        full = url
                    r = requests.get(full, timeout=20)
                    r.raise_for_status()
                    return r.content

                # Prepare in-memory ZIP
                zip_buffer = io.BytesIO()
                zipf = zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED)

                st.success("Results received. Review below.")

                # top-level container (prevents deep nesting)
                results_container = st.container()

                with results_container:
                    for item_i, item in enumerate(result.get("results", [])):
                        st.markdown(f"### Uploaded Image #{item_i+1}")

                        # Expander per image to keep the page compact
                        with st.expander(f"View image #{item_i+1} details", expanded=True):
                            # Left: annotated image (full width inside expander)
                            try:
                                annotated_bytes = fetch_bytes(item["annotated_url"])
                                st.image(annotated_bytes, caption="Annotated image", use_container_width=True)
                                zipf.writestr(f"image_{item_i+1}_annotated.png", annotated_bytes)
                            except Exception as e:
                                st.warning(f"Could not load annotated image: {e}")

                            cells = item.get("cells", [])
                            if not cells:
                                st.info("No cells detected by segmentation for this image.")
                            else:
                                st.info(f"Detected {len(cells)} cells")

                                # Show crops and metadata in rows of up to 3 thumbnails per row.
                                ncols = 3
                                for i in range(0, len(cells), ncols):
                                    row_cells = cells[i:i+ncols]
                                    cols = st.columns(len(row_cells))
                                    for cell, col in zip(row_cells, cols):
                                        with col:
                                            try:
                                                crop_bytes = fetch_bytes(cell["crop_url"])
                                                st.image(crop_bytes, width=140)
                                                label = cell.get("label", "").upper()
                                                p_leuk = cell.get("prob_leukemia", 0.0)
                                                p_norm = cell.get("prob_normal", 0.0)
                                                st.write(f"#{cell['cell_index']} • **{label}**")
                                                st.write(f"L: {p_leuk:.2f}  N: {p_norm:.2f}")
                                                zipf.writestr(f"image_{item_i+1}_cell_{cell['cell_index']}.png", crop_bytes)
                                            except Exception as e:
                                                st.write(f"Cell {cell.get('cell_index','?')}: failed to load ({e})")

                        st.markdown("---")

                # finalize zip
                zipf.close()
                zip_buffer.seek(0)
                now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                zip_filename = f"leukemia_results_{now}.zip"

                st.download_button(
                    label=" Download All Results (annotated + crops) as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=zip_filename,
                    mime="application/zip"
                )

# End of file
