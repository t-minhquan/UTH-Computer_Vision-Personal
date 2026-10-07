import os
import cv2
import tempfile
import numpy as np
import gradio as gr
from filters.pixelate import apply_pixelate
from filters.overlay import apply_overlay

STICKER_PATH = os.path.join("assets", "pixel_heart.png")

# ==================== 1. XỬ LÝ ẢNH ====================
def process_image(input_img, pixel_size, enable_overlay, sticker_scale, position):
    if input_img is None:
        return None
    
    # Tạo bản sao mảng NumPy để không ghi đè dữ liệu gốc
    img_copy = input_img.copy()

    # 1. Áp dụng Pixelate Filter
    result = apply_pixelate(img_copy, pixel_size=int(pixel_size))
    
    # 2. Áp dụng Overlay Sticker
    if enable_overlay and os.path.exists(STICKER_PATH):
        result = apply_overlay(
            bg_image_np=result,
            overlay_path=STICKER_PATH,
            scale=sticker_scale,
            position=position
        )
        
    return result


# ==================== 2. XỬ LÝ VIDEO (30 FPS) ====================
def process_video(video_path, pixel_size, enable_overlay, sticker_scale, position, progress=gr.Progress()):
    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)
    
    # Lấy thông số video gốc
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ép cố định Output Video chạy ở chuẩn 30 FPS
    target_fps = 30.0

    # Tạo file tạm lưu kết quả
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    output_path = temp_output.name
    temp_output.close()

    # Khởi tạo VideoWriter (mp4v codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV đọc khung hình dạng BGR -> Chuyển sang RGB để dùng với filter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Xử lý từng khung hình (Frame-by-Frame)
        processed_frame = process_image(
            frame_rgb, pixel_size, enable_overlay, sticker_scale, position
        )

        # Chuyển ngược RGB sang BGR để ghi ra file video
        frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

        frame_count += 1
        if total_frames > 0:
            progress(frame_count / total_frames, desc=f"Đang xử lý video... ({frame_count}/{total_frames} frames)")

    cap.release()
    out.release()

    return output_path


# ==================== 3. GIAO DIỆN GRADIO ====================
with gr.Blocks(title="Pixel Art Filter Studio") as demo:
    gr.Markdown("# 🎨 Pixel Art & Video Filter Studio (30 FPS)")
    gr.Markdown("Tải ảnh hoặc video của bạn lên để chuyển đổi sang phong cách Pixelate 8-bit.")

    with gr.Tab("🖼️ Xử lý Ảnh"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="numpy", image_mode="RGBA", label="Ảnh gốc")
                img_pixel_size = gr.Slider(minimum=2, maximum=50, value=12, step=1, label="Kích thước Pixel")
                img_enable_overlay = gr.Checkbox(label="Chèn Pixel Sticker", value=False)
                img_sticker_scale = gr.Slider(minimum=0.05, maximum=0.5, value=0.2, step=0.05, label="Tỉ lệ Sticker")
                img_position = gr.Dropdown(choices=["bottom-right", "top-right", "center", "top-left"], value="bottom-right", label="Vị trí Sticker")
                img_submit = gr.Button("Áp dụng cho Ảnh", variant="primary")
            with gr.Column():
                img_output = gr.Image(type="numpy", image_mode="RGBA", label="Kết quả")

        img_submit.click(
            fn=process_image,
            inputs=[img_input, img_pixel_size, img_enable_overlay, img_sticker_scale, img_position],
            outputs=img_output
        )

    with gr.Tab("🎥 Xử lý Video (30 FPS)"):
        with gr.Row():
            with gr.Column():
                vid_input = gr.Video(label="Video gốc")
                vid_pixel_size = gr.Slider(minimum=2, maximum=50, value=12, step=1, label="Kích thước Pixel")
                vid_enable_overlay = gr.Checkbox(label="Chèn Pixel Sticker", value=False)
                vid_sticker_scale = gr.Slider(minimum=0.05, maximum=0.5, value=0.2, step=0.05, label="Tỉ lệ Sticker")
                vid_position = gr.Dropdown(choices=["bottom-right", "top-right", "center", "top-left"], value="bottom-right", label="Vị trí Sticker")
                vid_submit = gr.Button("Áp dụng cho Video", variant="primary")
            with gr.Column():
                vid_output = gr.Video(label="Video kết quả (30 FPS)")

        vid_submit.click(
            fn=process_video,
            inputs=[vid_input, vid_pixel_size, vid_enable_overlay, vid_sticker_scale, vid_position],
            outputs=vid_output
        )

if __name__ == "__main__":
    demo.launch()