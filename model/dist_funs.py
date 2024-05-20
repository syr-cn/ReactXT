import torch
from typing import Any, Iterable, Iterator, List, Optional, Sized, Tuple, Union, Dict
from pytorch_lightning import strategies
from lightning_fabric.utilities.types import _PATH
from deepspeed.runtime.data_pipeline.data_routing.helper import remove_random_ltd_state_dict


'''
overwrite the function in deepspeed
'''

### start overwrite ###
def module_state_dict(self, destination=None, prefix="", keep_vars=False, exclude_frozen_parameters=False):
    sd = self.module.state_dict(destination, prefix, keep_vars)
    # Remove frozen parameter weights from state_dict if specified
    if exclude_frozen_parameters:
        to_be_removed = []
        for n in sd:
            try: 
                if not self.module.get_parameter(n).requires_grad:
                    to_be_removed.append(n)
            except AttributeError:
                to_be_removed.append(n)
        for key in to_be_removed:
            sd.pop(key)
    if self.random_ltd_enabled():
        sd = remove_random_ltd_state_dict(sd)
    return sd
from deepspeed import DeepSpeedEngine
DeepSpeedEngine.module_state_dict = module_state_dict
### end overwrite ###

class MyDeepSpeedStrategy(strategies.DeepSpeedStrategy):
    def save_checkpoint_v1(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ):
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
            storage_options: parameter for how to save to st
            orage, passed to ``CheckpointIO`` plugin
        """
        if self.is_global_zero:
            self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    def load_model_state_dict(self, checkpoint):
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)
    def save_checkpoint(self, checkpoint: Dict, filepath: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: The checkpoint state dictionary
            filepath: write-target file's path
            storage_options: not used for ``DeepSpeedStrategy`` as ``CheckpointIO`` is not used

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in
        """
        # broadcast the filepath from rank 0 to ensure all the states are saved in a common filepath
        filepath = self.broadcast(filepath)
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}` as `CheckpointIO` is not used."
            )

        if self.zero_stage_3 and self._multi_device and self.is_global_zero:
            print(
                "Warning: When saving the DeepSpeed Stage 3 checkpoint, "
                "each worker will save a shard of the checkpoint within a directory. "
                "If a single file is required after training, "
                "see https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html#"
                "deepspeed-zero-stage-3-single-file for instructions."
            )
        # Use deepspeed's internal checkpointing function to handle partitioned weights across processes
        # dump states as a checkpoint dictionary object
        _exclude_keys = ["state_dict", "optimizer_states"]
        checkpoint = {k: v for k, v in checkpoint.items() if k not in _exclude_keys}
        self.deepspeed_engine.save_checkpoint(filepath, client_state=checkpoint, tag="checkpoint", exclude_frozen_parameters=True)
