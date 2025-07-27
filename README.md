# Animal_Sound_Classification

This project provides a minimal CRNN training pipeline using the [ESC‑50 dataset](https://github.com/karoldvl/ESC-50). The dataset is located under `data/ESC-50-master`.

## Training

Run `train.py` to train the CRNN model. By default folds 1–4 are used for training and fold 5 for validation.

```bash
python train.py --epochs 10 --batch_size 16 --device cpu
```

Trained weights are saved to `crnn.pth`.

## Benchmarking

After training, evaluate the model with `evaluate.py` which loads the saved weights and reports accuracy on fold 5.

```bash
python evaluate.py --weights crnn.pth --device cpu
```
