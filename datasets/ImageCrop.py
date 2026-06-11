import os
from PIL import Image


def split_image_into_patches(input_dir, output_dir, patch_size=256):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".png"):
            continue

        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)

        width, height = img.size

        if width != 1024 or height != 1024:
            print(f"Warning: {filename} size is {width}x{height}, not 1024x1024. Skipped.")
            continue

        base_name = os.path.splitext(filename)[0]
        patch_id = 1

        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                patch = img.crop((x, y, x + patch_size, y + patch_size))

                output_filename = f"{base_name}_{patch_id}.png"
                output_path = os.path.join(output_dir, output_filename)

                patch.save(output_path)

                patch_id += 1

        print(f"Processed {filename}: generated {patch_id - 1} patches.")


def generate_split_txt(split_dir, split_name):
    """
    在 split_dir 下面生成 train.txt / val.txt / test.txt。

    默认从 A 文件夹中读取图片名。
    因为变化检测数据集中 A、B、label 的图片名通常是一一对应的。
    """

    image_dir = os.path.join(split_dir, "A")
    txt_path = os.path.join(split_dir, f"{split_name}.txt")

    if not os.path.exists(image_dir):
        print(f"Warning: {image_dir} does not exist. Cannot generate {split_name}.txt.")
        return

    image_names = [
        filename
        for filename in os.listdir(image_dir)
        if filename.lower().endswith(".png")
    ]

    image_names.sort()

    with open(txt_path, "w") as f:
        for name in image_names:
            f.write(name + "\n")

    print(f"Generated {txt_path}, total {len(image_names)} images.")


def process_levir_dataset(root_dir, output_root, patch_size=256):
    """
    处理类似 LEVIR-CD 的数据集结构：

    root_dir/
        train/
            A/
            B/
            label/
        val/
            A/
            B/
            label/
        test/
            A/
            B/
            label/

    输出结构：

    output_root/
        train/
            A/
            B/
            label/
            train.txt
        val/
            A/
            B/
            label/
            val.txt
        test/
            A/
            B/
            label/
            test.txt
    """

    splits = ["train", "val", "test"]
    folders = ["A", "B", "label"]

    for split in splits:
        for folder in folders:
            input_dir = os.path.join(root_dir, split, folder)
            output_dir = os.path.join(output_root, split, folder)

            if not os.path.exists(input_dir):
                print(f"Skip: {input_dir} does not exist.")
                continue

            print(f"\nProcessing {input_dir}")
            split_image_into_patches(input_dir, output_dir, patch_size)

        split_dir = os.path.join(output_root, split)
        generate_split_txt(split_dir, split)


if __name__ == "__main__":
    root_dir = "E:\Things_Of_Graduate\ECE_228\Proj\pycharm_project_396\datasets\LEVIR"
    output_root = "E:\Things_Of_Graduate\ECE_228\Proj\pycharm_project_396\datasets\LEVIR-256"

    process_levir_dataset(root_dir, output_root, patch_size=256)