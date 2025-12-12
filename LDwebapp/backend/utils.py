# backend/utils.py  (replace the old segment_cells with this)
import cv2
import numpy as np

def segment_cells(img, min_area=600):
    """
    Improved segmentation using HSV color threshold tuned for purple-stained cells.
    Returns list of (x,y,w,h,crop) for each detected cell (BGR crops).
    """
    # Work on a copy
    img_bgr = img.copy()
    # Convert to HSV to isolate purple staining (approx)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # These ranges are approximate for purple-ish stains; adjust if needed
    lower1 = np.array([120, 20, 20])   # lower hue for purple-ish (tweak if necessary)
    upper1 = np.array([170, 255, 255])

    # Create mask and clean it
    mask = cv2.inRange(hsv, lower1, upper1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fallback: if mask is too small, try Otsu on grayscale (helps when staining differs)
    if np.count_nonzero(mask) < 50:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area >= min_area and w > 12 and h > 12:
            # pad the box slightly (but keep inside image)
            pad = 4
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(img_bgr.shape[1], x + w + pad)
            y1 = min(img_bgr.shape[0], y + h + pad)
            crop = img_bgr[y0:y1, x0:x1]

            # skip crops that are mostly black or empty
            if crop.size == 0:
                continue
            if np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < 8:  # too dark
                continue

            detections.append((x0, y0, x1-x0, y1-y0, crop))

    detections = sorted(detections, key=lambda t: (t[1], t[0]))
    return detections


def draw_bounding_boxes_and_labels(img, boxes_with_labels, box_color=(0,0,255), text_color=(255,255,255)):
    annotated = img.copy()
    for item in boxes_with_labels:
        x, y, w, h = item["x"], item["y"], item["w"], item["h"]
        label = item.get("label", "")
        prob = item.get("prob", None)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), box_color, 2)
        text = f"{label}"
        if prob is not None:
            text = f"{label} {prob:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x, y - th - 6), (x + tw + 6, y), box_color, -1)
        cv2.putText(annotated, text, (x+3, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
    return annotated
