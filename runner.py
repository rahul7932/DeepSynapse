import argparse
from utils.load_dataset import load_dataset
from trainers.spineTrainer import SpineTrainer, train
from trainers.inference import inference
from models.model_cfg import get_cfg_mod
from detectron2.utils.logger import setup_logger


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run Detectron2 Spine Detection")
    parser.add_argument("--mode", choices=['train', 'inference'],
                        required=True, help="Whether to train or run inference")
    parser.add_argument("--dataDir", required=True,
                        help="Directory containing the dataset")
    parser.add_argument("--outputDir", default="output",
                        help="Directory to save outputs (weights for training or predictions for inference)")
    parser.add_argument("--weightPath", default=None,
                        help="Path to model weights (only needed for inference)")
    parser.add_argument("--visualize", action='store_true',
                        help="Visualize predictions")
    args = parser.parse_args()

    # Load the dataset
    dataset_dicts, spine_metadata, cfg = load_dataset(args.dataDir, args.mode)

    # Depending on the mode, train or run inference
    if args.mode == "train":
        train(dataset_dicts, spine_metadata, cfg, args.outputDir)
    elif args.mode == "inference":
        if args.weightPath is None:
            print(
                "Please provide a path to model weights using --weightPath for inference.")
            exit(1)
        inference(dataset_dicts, cfg, args.outputDir,
                  args.weightPath, val=True)

    if args.visualize:
        # Add visualization logic here if you want to visualize predictions or training data
        pass


if __name__ == "__main__":
    main()
