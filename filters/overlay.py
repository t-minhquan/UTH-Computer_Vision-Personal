import cv2
import numpy as np

def apply_overlay(bg_image_np: np.ndarray, overlay_path: str, scale: float = 0.2, position: str = "bottom-right") -> np.ndarray:
    if bg_image_np is None or not overlay_path:
        return bg_image_np

    # Đọc sticker giữ nguyên kênh Alpha
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return bg_image_np

    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    elif overlay.shape[2] == 4:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
    elif overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported sticker image shape: {overlay.shape}")

    bg_h, bg_w = bg_image_np.shape[:2]
    if bg_image_np.ndim != 3 or bg_image_np.shape[2] not in (3, 4):
        raise ValueError(f"Background must be an RGB or RGBA image, got {bg_image_np.shape}")
    
    # Resize sticker
    new_w = max(1, int(bg_w * scale))
    new_h = max(1, int(overlay.shape[0] * (new_w / overlay.shape[1])))
    overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Tính vị trí
    margin_x = min(20, max(0, bg_w - new_w))
    margin_y = min(20, max(0, bg_h - new_h))
    x, y = margin_x, margin_y
    if position == "bottom-right":
        x = max(0, bg_w - new_w - 20)
        y = max(0, bg_h - new_h - 20)
    elif position == "top-right":
        x = max(0, bg_w - new_w - 20)
        y = margin_y
    elif position == "center":
        x = (bg_w - new_w) // 2
        y = (bg_h - new_h) // 2

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(bg_w, x + new_w), min(bg_h, y + new_h)
    if x0 >= x1 or y0 >= y1:
        return bg_image_np

    overlay_crop = overlay[y0 - y:y1 - y, x0 - x:x1 - x]
    bg_crop = bg_image_np[y0:y1, x0:x1]

    if overlay_crop.shape[2] == 4:
        src_alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0
    else:
        src_alpha = np.ones((*overlay_crop.shape[:2], 1), dtype=np.float32)

    if bg_crop.shape[2] == 4:
        dst_alpha = bg_crop[:, :, 3:4].astype(np.float32) / 255.0
    else:
        dst_alpha = np.ones((*bg_crop.shape[:2], 1), dtype=np.float32)

    dst_rgb = bg_crop[:, :, :3].astype(np.float32)
    src_rgb = overlay_crop[:, :, :3].astype(np.float32)
    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    premultiplied_rgb = (
        src_rgb * src_alpha + dst_rgb * dst_alpha * (1.0 - src_alpha)
    )
    out_rgb = np.divide(
        premultiplied_rgb,
        out_alpha,
        out=np.zeros_like(premultiplied_rgb),
        where=out_alpha > 0,
    )

    bg_image_np[y0:y1, x0:x1, :3] = np.rint(np.clip(out_rgb, 0, 255)).astype(np.uint8)
    if bg_image_np.shape[2] == 4:
        bg_image_np[y0:y1, x0:x1, 3] = np.rint(
            np.clip(out_alpha[:, :, 0] * 255.0, 0, 255)
        ).astype(np.uint8)

    return bg_image_np