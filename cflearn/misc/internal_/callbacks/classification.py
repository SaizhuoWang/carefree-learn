import os

from .general import ImageCallback
from ...toolkit import to_device
from ...toolkit import save_images
from ...toolkit import eval_context
from ...toolkit import make_indices_visualization_map
from ....trainer import Trainer
from ....constants import INPUT_KEY


@ImageCallback.register("clf")
class ClassificationCallback(ImageCallback):
    def log_artifacts(self, trainer: Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[INPUT_KEY]
        with eval_context(trainer.model):
            logits = trainer.model.classify(original)
            labels_map = make_indices_visualization_map(logits.argmax(1))
        image_folder = self._prepare_folder(trainer)
        save_images(original, os.path.join(image_folder, "original.png"))
        save_images(labels_map, os.path.join(image_folder, "labels.png"))


__all__ = [
    "ClassificationCallback",
]
