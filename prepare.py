import os
import cv2
import shutil
import numpy as np
from PIL import Image, ImageOps
from dotenv import load_dotenv

#### Config ####
load_dotenv()
IMG_WH = int(os.getenv("IMG_WH"))
TRAIN_PERCENTAGE_SPLIT = int(os.getenv("TRAIN_PERCENTAGE_SPLIT")) / 100

#### Constants ####
OUTPUT_DIRECTORY = "./training_data_preprocessed/"
OUTPUT_TRAINING_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "training/")
OUTPUT_TRAINING_IMAGES_DIRECTORY = os.path.join(OUTPUT_TRAINING_DIRECTORY, "images/")
OUTPUT_TRAINING_LABELS_DIRECTORY = os.path.join(OUTPUT_TRAINING_DIRECTORY, "labels/")
OUTPUT_VALIDATION_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "validation/")
OUTPUT_VALIDATION_IMAGES_DIRECTORY = os.path.join(
    OUTPUT_VALIDATION_DIRECTORY, "images/"
)
OUTPUT_VALIDATION_LABELS_DIRECTORY = os.path.join(
    OUTPUT_VALIDATION_DIRECTORY, "labels/"
)


def find_and_convert_id_card_cropper_images(basePath):
    imagesPathBase = os.path.join(basePath, "images")
    annotationsPathBase = os.path.join(basePath, "labels")
    images = []
    for img_file_name in os.listdir(imagesPathBase):
        img_path = os.path.join(imagesPathBase, img_file_name)
        if os.path.isfile(img_path) is False or img_path.endswith(".jpg") is False:
            print(f"Image not ending with .jpg: {img_path}")
            continue

        annotation_path = os.path.join(
            annotationsPathBase, img_file_name.removesuffix(".jpg") + ".txt"
        )
        if os.path.isfile(annotation_path) is False:
            print(f"Annotation {annotation_path} missing, skipping...")
            continue

        imgPreprocessed = Image.open(img_path).resize((IMG_WH, IMG_WH), Image.LANCZOS)
        imgPreprocessed = ImageOps.exif_transpose(imgPreprocessed)
        with open(annotation_path) as f:
            images.append(
                {"image": imgPreprocessed, "annotation": "\n".join(f.readlines())}
            )
    return images


def find_frames_with_annotations(frames_path, annotations_path) -> list[dict]:
    images = []
    for img_file_name in os.listdir(frames_path):
        img_path = os.path.join(frames_path, img_file_name)
        if os.path.isfile(img_path) is False or img_path.endswith(".png") is False:
            continue

        annotation_path = os.path.join(annotations_path, img_file_name)
        if os.path.isfile(annotation_path) is False:
            print(f"Annotation {annotation_path} missing, skipping...")
            continue

        images.append({"baseImagePath": img_path, "maskImagePath": annotation_path})
    return images


def mask_image_to_yolov_annotations(annotationsImagePath) -> str:
    imgBase = Image.open(annotationsImagePath).resize((IMG_WH, IMG_WH), Image.LANCZOS)
    imgBaseExifAutoOrient = ImageOps.exif_transpose(imgBase)
    imgNP = np.array(imgBaseExifAutoOrient)
    imgOpenCV = cv2.cvtColor(cv2.cvtColor(imgNP, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

    h, w = imgOpenCV.shape
    _, thresh = cv2.threshold(imgOpenCV, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_lines = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        norm_bw = bw / w
        norm_bh = bh / h
        label_lines.append(
            f"0 {x_center:.6f} {y_center:.6f} {norm_bw:.6f} {norm_bh:.6f}"
        )

    return "\n".join(label_lines)


def train_validation_split(
    input_data: list[any], train_percentage=TRAIN_PERCENTAGE_SPLIT
):
    train_size = int(train_percentage * len(input_data))
    np.random.shuffle(input_data)
    return input_data[:train_size], input_data[train_size:]


def save_dataset_item(
    item,
    item_name: str,
    img_dest_path: str,
    label_dest_path: str,
):
    img = item["image"]
    yolo_annotation = item["annotation"]

    new_img_path = os.path.join(img_dest_path, f"{item_name}.png")
    img.save(new_img_path)

    new_yolo_annotation_path = os.path.join(label_dest_path, f"{item_name}.txt")
    yolo_annotation_file = open(new_yolo_annotation_path, "w")
    yolo_annotation_file.write(yolo_annotation)
    yolo_annotation_file.close()


if __name__ == "__main__":
    print("Preparing environment...")
    for path in [
        OUTPUT_DIRECTORY,
        OUTPUT_TRAINING_DIRECTORY,
        OUTPUT_TRAINING_IMAGES_DIRECTORY,
        OUTPUT_TRAINING_LABELS_DIRECTORY,
        OUTPUT_VALIDATION_DIRECTORY,
        OUTPUT_VALIDATION_IMAGES_DIRECTORY,
        OUTPUT_VALIDATION_LABELS_DIRECTORY,
    ]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

    print("Getting all images...")
    all_images_with_annotations = []
    all_images_with_annotations.extend(
        find_frames_with_annotations(
            "./resources/test_frames/image", "./resources/test_masks/image"
        )
    )
    all_images_with_annotations.extend(
        find_frames_with_annotations(
            "./resources/train_frames/image", "./resources/train_masks/image"
        )
    )
    all_images_with_annotations.extend(
        find_frames_with_annotations(
            "./resources/val_frames/image", "./resources/val_masks/image"
        )
    )

    print("Converting to a preferable format + extracting annotations...")
    images_preprocessed = []
    for i, image_with_annotation in enumerate(all_images_with_annotations):
        imgPreprocessed = Image.open(image_with_annotation["baseImagePath"]).resize(
            (IMG_WH, IMG_WH), Image.LANCZOS
        )
        imgPreprocessed = ImageOps.exif_transpose(imgPreprocessed)
        annotation = mask_image_to_yolov_annotations(
            image_with_annotation["maskImagePath"]
        )
        images_preprocessed.append({"image": imgPreprocessed, "annotation": annotation})

    print(
        "Converting id-card-cropper-1 to a preferable format + extracting annotations..."
    )
    images_preprocessed.extend(
        find_and_convert_id_card_cropper_images("./resources/id-card-cropper-1/test")
    )
    images_preprocessed.extend(
        find_and_convert_id_card_cropper_images("./resources/id-card-cropper-1/train")
    )
    images_preprocessed.extend(
        find_and_convert_id_card_cropper_images("./resources/id-card-cropper-1/valid")
    )

    print("Traing + validation split...")
    training_dataset, validation_dataset = train_validation_split(images_preprocessed)
    np.random.shuffle(training_dataset)
    np.random.shuffle(validation_dataset)

    print("Saving training dataset...")
    for i, item in enumerate(training_dataset):
        save_dataset_item(
            item, i, OUTPUT_TRAINING_IMAGES_DIRECTORY, OUTPUT_TRAINING_LABELS_DIRECTORY
        )

    print("Saving validation dataset...")
    for i, item in enumerate(validation_dataset):
        save_dataset_item(
            item,
            i,
            OUTPUT_VALIDATION_IMAGES_DIRECTORY,
            OUTPUT_VALIDATION_LABELS_DIRECTORY,
        )

    print("DONE :)")
