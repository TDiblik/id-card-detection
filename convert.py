import argparse
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

IMG_WH = int(os.getenv("IMG_WH"))

# https://github.com/ultralytics/ultralytics/issues/348
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="PT to ONNX convertor",
        description="Convert your model from .pt to .onnx for further use",
    )
    parser.add_argument("ai_model", help="Path to your finetuned model")
    args = parser.parse_args()

    print("Loading model...")
    model = YOLO(args.ai_model)

    model.export(format="tfjs", imgsz=IMG_WH)
