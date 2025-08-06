# Animal_Sound_Classification
My VWA Project

## Training

`Training.py` now logs metrics with [Weights & Biases](https://wandb.ai/).
During training the script records loss, accuracy, learning rate and
activation histograms for convolutional and linear layers. To use it,
install the dependencies and run:

```bash
python Training.py --epochs 1 --batch_size 16
```

Make sure to set the `WANDB_API_KEY` environment variable beforehand to
enable logging to your W&B account.

