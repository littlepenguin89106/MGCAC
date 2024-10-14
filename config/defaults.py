from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Directory
# -----------------------------------------------------------------------------
_C.DIR = CN()
_C.DIR.dataset = "FSC147" # the path of  train/test dataset
_C.DIR.source_dataset = "FSC147" # the path of original FSC147 dataset, only use for mosaic evaluation
_C.DIR.exp = "mgcac" # experiment name
_C.DIR.runs = "./" # the root folder for all experiments

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
# The path of data list
_C.DATASET.list_train = "./data/train.txt"
_C.DATASET.list_val = "./data/val.txt"
_C.DATASET.list_test = "./data/test.txt"
# preload dataset into the memory to speed up training
_C.DATASET.preload = True
# randomly horizontally flip images when training
_C.DATASET.random_flip = True
# The size of resized exemplar patches 
_C.DATASET.exemplar_size = (128, 128)
_C.DATASET.exemplar_number = 3

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.PRETRAINED= True
# The path of backbone pretrained weight
_C.MODEL.BACKBONE.PRETRAINED_PATH= './pretrained/CvT-21-384x384-IN-22k.pth'
_C.MODEL.BACKBONE.INIT= 'trunc_norm'
_C.MODEL.BACKBONE.NUM_STAGES= 3
_C.MODEL.BACKBONE.PATCH_SIZE= [ 7, 3, 3 ]
_C.MODEL.BACKBONE.PATCH_STRIDE= [ 4, 2, 2 ]
_C.MODEL.BACKBONE.PATCH_PADDING= [ 2, 1, 1 ]
_C.MODEL.BACKBONE.DIM_EMBED= [ 64, 192, 384 ]
_C.MODEL.BACKBONE.NUM_HEADS= [ 1, 3, 6 ]
_C.MODEL.BACKBONE.DEPTH= [ 1, 4, 16 ]
_C.MODEL.BACKBONE.MLP_RATIO= [ 4.0, 4.0, 4.0 ]
_C.MODEL.BACKBONE.ATTN_DROP_RATE= [ 0.0, 0.0, 0.0 ]
_C.MODEL.BACKBONE.DROP_RATE= [ 0.0, 0.0, 0.0 ]
_C.MODEL.BACKBONE.DROP_PATH_RATE= [ 0.0, 0.0, 0.1 ]
_C.MODEL.BACKBONE.QKV_BIAS= [ True, True, True ]
_C.MODEL.BACKBONE.CLS_TOKEN= [ False, False, False ]
_C.MODEL.BACKBONE.POS_EMBED= [ False, False, False ]
_C.MODEL.BACKBONE.QKV_PROJ_METHOD= [ 'dw_bn', 'dw_bn', 'dw_bn' ]
_C.MODEL.BACKBONE.KERNEL_QKV= [ 3, 3, 3 ]
_C.MODEL.BACKBONE.PADDING_KV= [ 1, 1, 1 ]
_C.MODEL.BACKBONE.STRIDE_KV= [ 2, 2, 2 ]
_C.MODEL.BACKBONE.PADDING_Q= [ 1, 1, 1 ]
_C.MODEL.BACKBONE.STRIDE_Q= [ 1, 1, 1 ]
_C.MODEL.BACKBONE.FREEZE_BN= True


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# restore training from a checkpoint
_C.TRAIN.resume = "model_ckpt.pth.tar"
# numbers of exemplar boxes
_C.TRAIN.exemplar_number = 3
# batch size
_C.TRAIN.batch_size = 1
# epochs to train for
_C.TRAIN.epochs = 20
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000
# optimizer and learning rate
_C.TRAIN.optimizer = "AdamW"
_C.TRAIN.lr_backbone = 0.01
_C.TRAIN.lr = 0.01
# milestone
_C.TRAIN.lr_drop = 200
# momentum
_C.TRAIN.momentum = 0.95
# weights regularizer
_C.TRAIN.weight_decay = 5e-4
# gradient clipping max norm
_C.TRAIN.clip_max_norm = 0.1
# number of data loading workers
_C.TRAIN.num_workers = 0
# manual seed
_C.TRAIN.seed = 2020
_C.TRAIN.start_epoch = 0
_C.TRAIN.device = 'cuda:0'

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# If set test_norm_thres and test_norm_scale_factor, enable test normalization during evaluation.
_C.VAL.test_norm_thres = None
_C.VAL.test_norm_scale_factor = None
# the checkpoint to evaluate on
_C.VAL.resume = "model_best.pth.tar"
# currently only supports 1
_C.VAL.batch_size = 1
# frequency to display
_C.VAL.disp_iter = 10
# frequency to validate
_C.VAL.val_epoch = 10
# evaluate_only
_C.VAL.evaluate_only = False
_C.VAL.visualization = False

_C.LOSS = CN()
_C.LOSS.name = 'GLoss'
_C.LOSS.crop_size = 384
_C.LOSS.downsample_ratio = 4
_C.LOSS.cost = 'exp'
_C.LOSS.scale = 0.6
_C.LOSS.reach = 0.5
_C.LOSS.blur = 0.01
_C.LOSS.scaling = 0.5
_C.LOSS.tau = 0.1
_C.LOSS.p = 1
_C.LOSS.d_point = 'l1'
_C.LOSS.d_pixel = 'l2'