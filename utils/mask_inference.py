import numpy as np
import cv2
from PIL import Image


def _to_rgb_np(pil_img):
    return np.array(pil_img.convert("RGB"))


def _to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _postprocess(binary):
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    binary = np.where(binary > 0, 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    x = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(x)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
    return mask


def _mask_score(mask, rgb):
    h, w = mask.shape[:2]
    area = float((mask > 0).sum()) / float(h * w)

    if area < 0.03 or area > 0.95:
        return -1e9

    edges = cv2.Canny(_to_gray(rgb), 60, 160)
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    boundary = (boundary > 0).astype(np.uint8)

    edge_hit = (edges > 0).astype(np.uint8) * boundary
    edge_align = edge_hit.sum() / max(1, boundary.sum())

    cy, cx = np.mean(np.where(mask > 0)[0]), np.mean(np.where(mask > 0)[1])
    dx = abs(cx - w / 2) / (w / 2)
    dy = abs(cy - h / 2) / (h / 2)
    center_penalty = 0.25 * (dx + dy)

    return 1.5 * edge_align + 0.4 * (1.0 - abs(area - 0.25)) - center_penalty


def _otsu_mask(rgb):
    gray = _to_gray(rgb)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if (th == 255).mean() > 0.6:
        th = cv2.bitwise_not(th)

    return _postprocess(th)


def _adaptive_mask(rgb):
    gray = _to_gray(rgb)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    if (th == 255).mean() > 0.6:
        th = cv2.bitwise_not(th)

    return _postprocess(th)


def _canny_contour_mask(rgb):
    gray = _to_gray(rgb)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    if not contours:
        return mask

    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
    return _postprocess(mask)


def _grabcut_mask(rgb):
    h, w = rgb.shape[:2]
    rect = (int(w * 0.08), int(h * 0.08), int(w * 0.84), int(h * 0.84))

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(rgb, mask, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return np.zeros((h, w), np.uint8)

    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return _postprocess(out)


def get_object_mask(pil_img):
    rgb = _to_rgb_np(pil_img)

    candidates = []
    for fn in (_otsu_mask, _adaptive_mask, _canny_contour_mask, _grabcut_mask):
        m = fn(rgb)
        s = _mask_score(m, rgb)
        candidates.append((s, m))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]
    return Image.fromarray(best, mode="L")
