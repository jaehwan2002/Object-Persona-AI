from PIL import Image, ImageDraw
import numpy as np
import os
import random


def extract_dominant_color(pil_img):
    small = pil_img.resize((50, 50))
    arr = np.array(small)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]

    mean_color = arr.reshape(-1, 3).mean(axis=0)
    return tuple(map(int, mean_color))


def colorize_mask(mask_img, base_color):
    mask_arr = np.array(mask_img)
    alpha = (mask_arr > 0).astype(np.uint8) * 255
    alpha_img = Image.fromarray(alpha, mode="L")

    rgba = Image.new("RGBA", mask_img.size, base_color + (0,))
    rgba.putalpha(alpha_img)
    return rgba


def _create_simple_face(size=(140, 100), style="귀여움"):
    eyes = Image.new("RGBA", size, (0, 0, 0, 0))
    mouth = Image.new("RGBA", size, (0, 0, 0, 0))

    de = ImageDraw.Draw(eyes)
    dm = ImageDraw.Draw(mouth)

    w, h = size
    r = h // 4
    gap = w // 6

    de.ellipse((gap, h // 4, gap + r, h // 4 + r), fill=(0, 0, 0, 255))
    de.ellipse((w - gap - r, h // 4, w - gap, h // 4 + r), fill=(0, 0, 0, 255))

    if style == "귀여움":
        dm.arc((w // 3, h // 4, 2 * w // 3, 3 * h // 4), 0, 180, fill=(0, 0, 0, 255), width=5)
    elif style == "잔잔함":
        dm.line((w // 3, h // 2, 2 * w // 3, h // 2), fill=(0, 0, 0, 255), width=5)
    else:
        dm.arc((w // 3, h // 3, 2 * w // 3, h), 200, 340, fill=(0, 0, 0, 255), width=5)

    return eyes, mouth


def _style_key(style):
    return {"귀여움": "cute", "잔잔함": "calm", "액션": "action"}.get(style, "")


def _list_imgs(folder):
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]


def load_random_assets(style, assets_dir="assets"):
    key = _style_key(style)

    eyes_dir = os.path.join(assets_dir, "eyes")
    mouths_dir = os.path.join(assets_dir, "mouths")

    eyes_files = _list_imgs(eyes_dir)
    mouth_files = _list_imgs(mouths_dir)

    if not (eyes_files and mouth_files):
        return _create_simple_face(style=style)

    eyes_candidates = [f for f in eyes_files if key and f.lower().startswith(key + "_")]
    mouth_candidates = [f for f in mouth_files if key and f.lower().startswith(key + "_")]

    if not eyes_candidates:
        eyes_candidates = eyes_files
    if not mouth_candidates:
        mouth_candidates = mouth_files

    eyes_path = os.path.join(eyes_dir, random.choice(eyes_candidates))
    mouth_path = os.path.join(mouths_dir, random.choice(mouth_candidates))

    eyes = Image.open(eyes_path).convert("RGBA")
    mouth = Image.open(mouth_path).convert("RGBA")
    return eyes, mouth


def _mask_stats(mask_img):
    arr = np.array(mask_img)
    ys, xs = np.where(arr > 0)

    w, h = mask_img.size
    if len(xs) == 0:
        return (w // 2, h // 2), (0, 0, w - 1, h - 1)

    cx, cy = int(xs.mean()), int(ys.mean())
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    return (cx, cy), (x1, y1, x2, y2)


def compose_character(original_img, mask_img, style="귀여움"):
    original_resized = original_img.resize(mask_img.size).convert("RGBA")

    dom_color = extract_dominant_color(original_resized)

    overlay = colorize_mask(mask_img, dom_color)
    ov = np.array(overlay)
    ov[..., 3] = (ov[..., 3] * 0.25).astype(np.uint8)
    overlay = Image.fromarray(ov, mode="RGBA")

    canvas = Image.alpha_composite(original_resized, overlay)

    eyes, mouth = load_random_assets(style)

    (cx, cy), (x1, y1, x2, y2) = _mask_stats(mask_img)
    obj_w = max(1, x2 - x1)
    obj_h = max(1, y2 - y1)

    eyes_w = int(obj_w * 0.45)
    eyes_h = int(obj_h * 0.18)
    mouth_w = int(obj_w * 0.28)
    mouth_h = int(obj_h * 0.14)

    eyes = eyes.resize((max(30, eyes_w), max(20, eyes_h)))
    mouth = mouth.resize((max(20, mouth_w), max(15, mouth_h)))

    tall_ratio = obj_h / max(1, obj_w)
    y_bias = int(obj_h * (0.08 if tall_ratio > 1.2 else 0.0))

    eyes_pos = (cx - eyes.size[0] // 2, cy - int(obj_h * 0.18) - y_bias)
    mouth_pos = (cx - mouth.size[0] // 2, cy + int(obj_h * 0.05) - y_bias)

    canvas.alpha_composite(eyes, eyes_pos)
    canvas.alpha_composite(mouth, mouth_pos)

    return canvas
