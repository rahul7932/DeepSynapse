from utils.build_dataset import build_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from models.model_cfg import get_cfg_mod


def load_dataset(dataDir, loadMode):
    """
    Load and register a custom dataset with Detectron2, retrieve its metadata, 
    and obtain the appropriate model configurations based on the mode.

    Parameters:
    - dataDir: The directory where the dataset is located.
    - loadMode: Mode of the dataset ("train", "val", or "test").

    Returns:
    - dataset_dicts: The structured dataset.
    - spine_metadata: Metadata of the dataset.
    - cfg: Model configuration.
    """

    dataset_name = f"spine_{loadMode}"

    # Register the custom dataset with Detectron2
    DatasetCatalog.register(
        dataset_name, lambda: build_dataset(dataDir, loadMode))

    # Assign and retrieve metadata for the registered dataset
    MetadataCatalog.get(dataset_name).set(thing_classes=["spine"])
    spine_metadata = MetadataCatalog.get(dataset_name)

    # Retrieve the dataset
    dataset_dicts = DatasetCatalog.get(dataset_name)

    # Retrieve model configuration based on the loadMode
    cfg = get_cfg_mod(val=True) if loadMode == 'val' else get_cfg_mod()

    return dataset_dicts, spine_metadata, cfg
