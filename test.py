import argparse
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()
IMG_WH = int(os.getenv("IMG_WH"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ALPR image manual tester",
        description="Manually nad visually view your AI's output",
    )
    parser.add_argument("ai_model", help="Path to your finetuned model")
    parser.add_argument("img_to_test", help="Path to image to test")
    args = parser.parse_args()

    print("Loading model")
    prediction_model = YOLO(args.ai_model, task="detect")

    print("Starting prediction")
    img_to_test = Image.open(args.img_to_test).resize((IMG_WH, IMG_WH), Image.LANCZOS)
    license_plates_as_boxes = prediction_model.predict(
        img_to_test, verbose=True, imgsz=IMG_WH
    )[0].boxes
    if len(license_plates_as_boxes) == 0:
        print("Didn't find any boxes/matches.")
        sys.exit(0)

    print("Plotting image")
    plt.figure()
    plt.imshow(img_to_test)

    print("Plotting boxes")
    ax = plt.gca()
    for i, box in enumerate(license_plates_as_boxes):
        x_min, y_min, x_max, y_max = box.xyxy.cpu().detach().numpy()[0]
        rect = Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color="red"
        )
        ax.add_patch(rect)

    plt.show()
