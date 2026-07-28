import os
import re
import cv2

def extract_assembly_clips(txt_path: str, video_path: str, output_dir: str, annotation_fps: float = 30.0):
    """
    Extracts video clips based on frame numbers from Assembly101 annotations.
    
    :param txt_path: Path to the annotation text file.
    :param video_path: Path to the source Assembly101 video file.
    :param output_dir: Directory where extracted clips will be saved.
    :param annotation_fps: Frame rate the annotations were indexed at (default: 30.0 FPS).
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Parse the annotation file
    clips_info = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t') if '\t' in line else line.split()
            
            if len(parts) >= 3:
                start_f30 = int(parts[0])
                end_f30 = int(parts[1])
                action_label = parts[2].strip()
                clips_info.append((start_f30, end_f30, action_label))

    # 2. Open the source video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video at {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate multiplier (e.g., 60 FPS video / 30 FPS annotation = 2x frame index)
    scale_factor = video_fps / annotation_fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print(f"Video FPS: {video_fps:.2f} | Annotation FPS: {annotation_fps}")
    print(f"Frame Scale Multiplier: {scale_factor:.2f}")
    print(f"Found {len(clips_info)} segment annotations.\n")

    # 3. Extract clips with scaled frame numbers
    for idx, (start_f30, end_f30, label) in enumerate(clips_info, start=1):
        # Convert 30 FPS indices to actual video frame indices
        actual_start_frame = int(round(start_f30 * scale_factor))
        actual_end_frame = int(round(end_f30 * scale_factor))

        safe_label = re.sub(r'[^a-zA-Z0-9_\-]', '_', label).strip('_')
        out_filename = f"{idx:02d}_{safe_label}_{start_f30}_{end_f30}.mp4"
        out_path = os.path.join(output_dir, out_filename)

        # Jump directly to the scaled start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start_frame)
        
        # Save output using the native video FPS
        writer = cv2.VideoWriter(out_path, fourcc, video_fps, (width, height))
        
        current_frame = actual_start_frame
        while current_frame <= actual_end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Reached end of video early at frame {current_frame}")
                break
            
            writer.write(frame)
            current_frame += 1

        writer.release()
        print(f"Saved: {out_filename} (Mapped 30fps frames [{start_f30}:{end_f30}] -> video frames [{actual_start_frame}:{actual_end_frame}])")

    cap.release()
    print("\nExtraction finished!")

if __name__ == "__main__":
    extract_assembly_clips(
        txt_path="./Data/annotations/a01/sample_2.txt",
        video_path="./Data/recordings/a01/sample_2.mp4",
        output_dir="./Data/extracted_clips/a01/sample_2/"
    )