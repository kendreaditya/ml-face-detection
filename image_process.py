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
    path = './data'
    names = ['aditya', 'jinyoon', 'Kenya', 'Peter', 'sami']
    video_form = ['.mp4', '.mov', '.MOV']
    for name in names:
        dir_path = os.path.join(path, name)
        output_images = os.path.join(dir_path, '0_images')
        output_resized = os.path.join(dir_path, '1_resized')
        os.makedirs(output_images, exist_ok=True)
        os.makedirs(output_resized, exist_ok=True)
        for file in os.listdir(dir_path):
            if file.endswith(tuple(video_form)):
                video_path = os.path.join(dir_path, file)
                desired_fps = 30
                get_frames(name, video_path, output_images, desired_fps)
        resize_images(output_images, output_resized, 270, 480)