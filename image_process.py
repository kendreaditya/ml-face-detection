import os
import cv2
import argparse
import multiprocessing
from tqdm import tqdm

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./data", help="path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./images", help="path to output directory")
    parser.add_argument("--width", type=int, default=256, help="output image width")
    parser.add_argument("--height", type=int, default=256, help="output image height")
    parser.add_argument("--fps", type=int, default=30, help="desired frame rate")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpu", action="store_true", help="use CPU")
    group.add_argument("--gpu", action="store_true", help="use GPU")

    args = parser.parse_args()

    device = "cuda" if args.gpu else "cpu"

    names = os.listdir(args.dataset_dir)
    video_form = ['.mp4', '.mov', '.MOV']

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for name in names:
            dir_path = os.path.join(args.dataset_dir, name)
            output_images = os.path.join(args.output_dir, name)
            os.makedirs(output_images, exist_ok=True)

            for file in tqdm(os.listdir(dir_path), desc=f"Processing {name}'s videos"):
                if file.endswith(tuple(video_form)):
                    video_path = os.path.join(dir_path, file)
                    pool.apply_async(get_frames, args=(name, video_path, output_images, args.fps))

        pool.close()
        pool.join()

    # resize_images(args.output_dir, args.output_dir, args.width, args.height)