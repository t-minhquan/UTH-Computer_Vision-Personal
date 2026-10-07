import cv2
import numpy as np

def apply_pixelate(image_np: np.ndarray, pixel_size: int = 10) -> np.ndarray:
    if image_np is None or pixel_size <= 1:
        return image_np
    
    # Đảm bảo mảng contigous trong bộ nhớ để tránh lỗi stride
    img = np.ascontiguousarray(image_np)
    h, w = img.shape[:2]
    
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    
    # Kiểm tra xem ảnh có kênh Alpha (4 kênh) hay không
    if img.ndim == 3 and img.shape[2] == 4:
        # Tách riêng 3 kênh màu (RGB) và 1 kênh trong suốt (Alpha)
        rgb = img[:, :, :3]
        alpha = img[:, :, 3]
        
        # Pixelate phần RGB
        small_rgb = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        pixel_rgb = cv2.resize(small_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Pixelate phần Alpha
        small_alpha = cv2.resize(alpha, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        pixel_alpha = cv2.resize(small_alpha, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Ghép lại thành ảnh 4 kênh RGBA chuẩn
        return np.dstack((pixel_rgb, pixel_alpha))
    elif img.ndim == 3:
        # Nếu là ảnh 3 kênh RGB thông thường
        small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small_img, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small_img, (w, h), interpolation=cv2.INTER_NEAREST)