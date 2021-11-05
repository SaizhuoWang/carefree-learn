import os
import sys

from typing import Any
from typing import Union
from typing import Optional
from typing import Dict
from cftool.misc import Saving, shallow_copy_dict

from cflearn.constants import META_CONFIG_NAME


class Task:
    def __init__(self, run_command: Optional[str] = None, **meta_kwargs: Any):
        self.run_command = run_command
        self.meta_kwargs = meta_kwargs

    def run(
        self,
        execute: str,
        config_folder: str,
        cuda: Optional[Union[int, str]],
    ) -> "Task":
        # Make command
        if self.run_command is not None:
            command = self.run_command
        else:
            command = f"{sys.executable} -m cflearn.dist.ml.runs.{execute}"

        # Make environment variables to export
        export = self.meta_kwargs.get('export', None)
        if export is not None:
            environ_list = [f'{k}={v}' for k, v in export.items()]
            env = ' '.join(environ_list)
        else:
            env = ''
        self.meta_kwargs.pop('export')

        meta_config = shallow_copy_dict(self.meta_kwargs)
        meta_config["cuda"] = cuda
        os.makedirs(config_folder, exist_ok=True)
        Saving.save_dict(meta_config, META_CONFIG_NAME, config_folder)
        os.system(f"{env} {command} --config_folder {config_folder}")
        return self

    def save(self, saving_folder: str) -> "Task":
        os.makedirs(saving_folder, exist_ok=True)
        meta_config = shallow_copy_dict(self.meta_kwargs)
        meta_config["run_command"] = self.run_command
        Saving.save_dict(meta_config, META_CONFIG_NAME, saving_folder)
        return self

    @classmethod
    def load(cls, saving_folder: str) -> "Task":
        meta_config = Saving.load_dict(META_CONFIG_NAME, saving_folder)
        return cls(**meta_config)


__all__ = ["Task"]
