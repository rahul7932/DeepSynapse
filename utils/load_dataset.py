from detectron2.data import DatasetCatalog, MetadataCatalog
from models.model_cfg import get_cfg_mod
from utils.build_dataset import build_dataset


def load_dataset(dataDir, loadMode):
    # loadMode is train, test, val

    # used to register a custom dataset with detectron
    DatasetCatalog.register("spine_" + loadMode,
                            build_dataset(dataDir, loadMode))
    # registering the metadata of the dataset (each thing is a spine)
    MetadataCatalog.get("spine_" + loadMode).set(thing_classes=["spine"])
    # getting metadata
    spine_metadata = MetadataCatalog.get("spine_" + loadMode)
    # getting dataset
    dataset_dicts = DatasetCatalog.get('spine_' + loadMode)

    # get the configs of the model for val
    if loadMode == 'val':
        cfg = get_cfg_mod(val=True)
    # get the configs of the model for train, test
    else:
        cfg = get_cfg_mod()

    # data, metadata, config
    return dataset_dicts, spine_metadata, cfg