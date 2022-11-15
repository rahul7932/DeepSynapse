def build_spine_train_aug(cfg):
    # augmentations
    augs = [
        #T.ResizeShortestEdge(
        #            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        #        )
    ]
    #if cfg.INPUT.CROP.ENABLED:
    #    augs.append(
    #        T.RandomCrop(
    #            cfg.INPUT.CROP.TYPE,
    #            cfg.INPUT.CROP.SIZE
    #        )
    #    )
    #augs.append(T.RandomFlip())
    #augs.append(T.RandomRotation([0.0, 180.0]))
    #augs.append(T.RandomSaturation(0.3, 2))
    #augs.append(T.RandomContrast(0.3, 2))
    #augs.append(T.RandomBrightness(0.3, 2))
    return augs