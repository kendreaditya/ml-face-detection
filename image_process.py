import os
import cv2

def get_frames(name, video_path, output_dir, fps):  # 30 frame, from video to images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    step = frame_rate // fps if frame_rate >= fps else 1

    frame_idx = 0
    for i in range(0, frame_count):
        ret, frame = cap.read()
        if ret:
            if i % step == 0:
                cv2.imwrite(os.path.join(output_dir, f"{name}_{frame_idx}.png"), frame)  # caps at 4th digit, 01,02....
                frame_idx += 1
        else:
            break
    cap.release()  # name then frame ex) jinyoon_1, jinyoon_2,...

def resize_images(input_dir, output_dir, width, height):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, file)
            img = cv2.imread(img_path)
            resized_img = cv2.resize(img, (width, height))
            resized_img_path = os.path.join(output_dir, file)
            cv2.imwrite(resized_img_path, resized_img)

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./data", help="path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./images", help="path to output directory")

    args = parser.parse_args()

    names = os.listdir(args.dataset_dir)
    video_form = ['.mp4', '.mov', '.MOV']

    for name in names:
        dir_path = os.path.join(args.dataset_dir, name)
        output_images = os.path.join(args.output_dir, name)
        os.makedirs(dir_path, exist_ok=True)

        for file in tqdm(os.listdir(dir_path), desc=f"Processing {name}'s videos"):
            if file.endswith(tuple(video_form)):
                video_path = os.path.join(dir_path, file)
                desired_fps = 30
                get_frames(name, video_path, output_images, desired_fps)