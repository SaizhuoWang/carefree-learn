# type: ignore

import cflearn
import argparse

# CI
parser = argparse.ArgumentParser()
parser.add_argument("--ci", type=int, default=0)
args = parser.parse_args()
is_ci = bool(args.ci)


num_classes = 10
data = cflearn.cv.MNISTData(
    root="../data",
    batch_size=4 if is_ci else 64,
    transform="for_generation",
)

m = cflearn.cv.CarefreePipeline(
    "siren_vae",
    {
        "in_channels": 1,
        "num_classes": num_classes,
    },
    callback_names="sized_generator",
    loss_metrics_weights={"kld": 0.001, "mse": 1.0},
    fixed_steps=1 if is_ci else None,
    valid_portion=0.0001 if is_ci else 1.0,
)
m.fit(data, cuda=None if is_ci else 3)
