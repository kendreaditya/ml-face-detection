import os
import random
from PIL import Image
import shutil
from random import shuffle

def sampling_images(src_path, dst_path, max_count):
    # List all image files in src_path
    image_files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]

    # Calculate the number of images to oversample
    num_images_to_add = max_count - len(image_files)

    # Remove any existing images in the dst_path directory
    for file in os.listdir(dst_path):
        file_path = os.path.join(dst_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Oversample images by copying existing images randomly
    for i in range(num_images_to_add):
        # Select a random image file from the source directory
        random_image = random.choice(image_files)

        # Construct the source image path
        random_image_path = os.path.join(src_path, random_image)

        # Construct the destination image path, adding a suffix to the filename to avoid overwriting
        dst_image_path = os.path.join(dst_path, f"{random_image.split('.')[0]}_copy{i}.{random_image.split('.')[1]}")

        # Open, copy and save the image to the destination path
        img = Image.open(random_image_path)
        img.save(dst_image_path)

    # Finally, copy all original images from src_path to dst_path
    for image_file in image_files:
        src_image_path = os.path.join(src_path, image_file)
        dst_image_path = os.path.join(dst_path, image_file)

        # Open, copy, and save the image to the destination path
        img = Image.open(src_image_path)
        img.save(dst_image_path)

def create_dataset_structure(src_dir, train_class_dir, test_class_dir, split_ratio):
    # get all image files in the src(aug) directory
    image_files = os.listdir(src_dir)
    # shuffle then split for whatever ratio you want
    shuffle(image_files)
    split_index = int(len(image_files) * split_ratio)
    train_images = image_files[:split_index]
    test_images = image_files[split_index:]

    # Copy train and test images to their distribute(dst) directories
    for image_file in train_images:
        src_image_path = os.path.join(src_dir, image_file)
        dst_image_path = os.path.join(train_class_dir, image_file)
        shutil.copy(src_image_path, dst_image_path)
    for image_file in test_images:
        src_image_path = os.path.join(src_dir, image_file)
        dst_image_path = os.path.join(test_class_dir, image_file)
        shutil.copy(src_image_path, dst_image_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./images")
    args = parser.parse_args()

    names = os.listdir(args.dataset_dir)

    image_counts = {}

    for name in names:
        image_path = os.path.join(args.dataset_dir, name)
        image_counts[name] = len(os.listdir(image_path))

    max_count = max(image_counts.values())

    for name in names:
        src_path = os.path.join(root, name, '1_resized')
        dst_path = os.path.join(root, name, '2_sampling')
        os.makedirs(dst_path, exist_ok=True)
        sampling_images(src_path, dst_path, max_count)

    # 2. now time to split, create train and test
    split_ratio = 0.8
    for name in names:
        src_path = os.path.join(root, name, '2_sampling')
        train_dst_path = os.path.join(root, "dataset/train", name)
        test_dst_path = os.path.join(root, "dataset/test", name)
        os.makedirs(train_dst_path, exist_ok=True)
        os.makedirs(test_dst_path, exist_ok=True)
        create_dataset_structure(src_path, train_dst_path, test_dst_path, split_ratio)

    # 3. do augmentation
    # for name in names:
    #     src_path = os.path.join(root, "train_before", name)
    #     dst_path = os.path.join("..", "..", "train")
    #     apply_augmentation(src_path, dst_path)


