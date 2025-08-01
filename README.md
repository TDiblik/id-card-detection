# ID Card detection

## Prepare

```sh
pip install -r requirements.txt
python prepare.py
```

## Train

1. Run prepare steps.
2. Edit `.env` and `train.py` to your liking.
3. Run `python train.py`.

## Test

1. Run `python test.py {path_to_your_model} {path_to_image_to_test}`
2. Edit `.env` and `train.py` to your liking.
3. Run `python train.py`.

## Convert to web format

1. Run `python convert.py {path_to_your_file}`

## Notes

- You'll find your model after training at `./runs/detect/train/weights/best.pt`
