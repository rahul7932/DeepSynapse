import argparse
from models.model_cfg import *
import os
from utils.load_dataset import load_dataset

from detectron2.utils.logger import setup_logger
from trainers.inference import inference
from trainers.trainer import train
from trainers.validators.test import test

setup_logger()


if __name__ == '__main__':
    currDir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(
        description='Run training or inference with detectron2 on spine data')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='mode (train / resume / val / test) \n' +
                             'val/test: runs inference and then evaluates detections (no output file)')
    parser.add_argument('dataDir', type=str, default='/spineData/images2D_croppedBackbone',
                        help='Parent directory of dataset (starting from current "detection" directory)')
    parser.add_argument('outputDir', metavar='outDir', type=str,
                        help='output directory to write/read weights from, also where inference result will be output')
    parser.add_argument('--weightPath', type=str)
    parser.add_argument('--visualizeGT', action='store_true')
    parser.add_argument('--visualizeTestPreds', action='store_true')
    parser.add_argument('--generatePredFile', action='store_true')

    args = parser.parse_args()
    print(f'info: run args {args}')

    dataDir = os.path.join(currDir, args.dataDir)
    if args.mode in ['train', 'resume']:
        dataset, metadata, cfg = load_dataset(dataDir, 'train')
    else:
        dataset, metadata, cfg = load_dataset(dataDir, args.mode)

    print('info: running {}'.format(args.mode))

    if args.mode in ['train', 'resume']:
        toResume = args.mode != 'train'
        train(dataset, metadata, cfg, args.outputDir,
              toResume=toResume, vis=args.visualizeGT)

    elif args.mode in ['test', 'val']:
        assert args.weightPath is not None

        if args.generatePredFile:
            print('info: generating pred file')
            original_output = cfg.OUTPUT_DIR
            inference(dataset, cfg, args.outputDir, args.weightPath, val=args.mode == 'val')
            cfg.OUTPUT_DIR = original_output

        print('info: generating metric data through COCOEvaluator')
        test(dataset, metadata, cfg, args.outputDir, args.weightPath,
             vis=args.visualizeTestPreds, val=args.mode == 'val')