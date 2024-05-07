from lavis.datasets.builders.base_dataset_builder import base_dataset_builder
from lavis.common.registry import registry
from lavis.dataset.datasets import (
    RXRCaptionDataset,
    RXRCaptionEvalDataset,
    RXRCaptionInstructDataset
)

@registry.register_builder("rxr")
class RXRCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = RXRCaptionDataset
    '''
    In captioning tasks, during test time, each data sample often 
    includes multiple ground-truth captions rather than just a single 
    ground-truth during training time.
    '''
    eval_dataset_cls = RXRCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rxr/rxr_caption.yaml",
    }

# TODO: pendiente resolver RXRCaptionInstructDataset