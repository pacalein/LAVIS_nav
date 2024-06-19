from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.common.registry import registry
from lavis.datasets.datasets.rxr_caption_datasets import (
    RXRCaptionDataset,
    RXRCaptionEvalDataset,
    RXRCaptionInstructDataset
)

@registry.register_builder("rxr_caption")
class RXRCaptionBuilder(BaseDatasetBuilder):
    # train_dataset_cls = RXRCaptionDataset
    train_dataset_cls = RXRCaptionInstructDataset
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