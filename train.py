import os
import torch
from ultralytics import YOLO, settings
from dotenv import load_dotenv

load_dotenv()

IMG_WH = int(os.getenv("IMG_WH"))
BASE_MODEL = os.getenv("TRAIN_BASE_MODEL")

# https://github.com/ultralytics/ultralytics/issues/348
if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Load model
    model = YOLO(BASE_MODEL)

    # https://docs.ultralytics.com/modes/train/#arguments
    # Specifically pay attention to `batch`, if you encouter the `torch.cuda.OutOfMemoryError: CUDA out of memory.`, just make the batches smaller.
    model.train(
        data="./data.yaml", imgsz=IMG_WH, epochs=100, patience=50, batch=16, cache=True
    )
