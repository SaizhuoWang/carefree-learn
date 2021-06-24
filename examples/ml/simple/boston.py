import cflearn

import numpy as np

from cfdata.tabular import TabularDataset


metrics = ["mae", "mse"]
x, y = TabularDataset.boston().xy
y = (y - y.mean()) / y.std()
m = cflearn.make(
    "ml.simple",
    config={
        "output_dim": 1,
        "is_classification": False,
        "loss_name": "mae",
        "metric_names": metrics,
    },
)
m.fit(x, y)
assert isinstance(m, cflearn.ml.SimplePipeline)
cflearn.ml.evaluate(x, y, metrics=metrics, pipelines=m)

predictions = m.predict(x)[cflearn.PREDICTIONS_KEY]
m.save("boston")
m2 = cflearn.ml.SimplePipeline.load("boston")
assert np.allclose(predictions, m2.predict(x)[cflearn.PREDICTIONS_KEY])
m.to_onnx("boston_onnx")
m3 = cflearn.ml.SimplePipeline.from_onnx("boston_onnx")
assert np.allclose(predictions, m3.predict(x)[cflearn.PREDICTIONS_KEY], atol=1.0e-4)
