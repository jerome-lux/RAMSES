import json
from pathlib import Path
import os


class Config:

    def __init__(self, **kwargs):

        self.load_backbone = False
        self.backbone_params = {}
        self.backbone = ("resnext50",)

        # Specific params
        self.ncls = 1
        self.imshape = (4096, 6144, 3)
        self.mask_stride = 4  # It must match with the backbone and the param 'output_level'

        # General layers params
        self.activation = "gelu"
        self.normalization = "gn"  # gn for Group Norm
        self.normalization_kw = {"groups": 32}
        self.model_name = "SOLOv2-Resnext50"

        # FPN
        self.connection_layers = {
            "C2": "stage1_block3Convblock",
            "C3": "stage2_block4Convblock",
            "C4": "stage3_block6Convblock",
            "C5": "stage4_block3Convblock",
        }
        self.FPN_filters = 256
        self.extra_FPN_layers = 1  # layers after P5. Strides must correspond to the number of FPN layers !

        # SOLO head
        self.head_filters = [256, 256, 256, 256]  # Filters per stage
        self.strides = [4, 8, 16, 32, 64]  # strides of FPN levels
        self.head_layers = 4  # Number of repeats of head conv layers
        self.head_filters = 256
        self.kernel_size = 1
        self.grid_sizes = [64, 36, 24, 16, 12]
        self.scale_ranges = [[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]]  # if P2 level is stride 4
        self.offset_factor = 0.25

        # SOLO MASK head
        self.point_nms = False
        self.mask_mid_filters = 128
        self.mask_output_filters = 256
        self.geom_feat_convs = 2  # number rof convs in the geometry factor branch
        self.geom_feats_filters = 128
        self.mask_output_level = 0  # size of the unified mask (in level of the FPN)
        self.sigma_nms = 0.5
        self.min_area = 0

        # loss and training parameters
        self.lossweights = [1.0, 1.0, 1.0]
        self.max_pos_samples = 512  # limit the number of positive gt samples when computing loss to limit memory footprint
        self.threshold_metrics = 0.0
        self.compute_cls_loss = True
        self.compute_seg_loss = True
        self.compute_density_loss = True

        # Update defaults parameters with kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        s = ""

        for k, v in self.__dict__.items():
            s += "{}:{}\n".format(k, v)

        return s

    def save(self, filename):

        # data = {k:v for k, v in self.__dict__.items()}

        p = Path(filename).parent.absolute()
        if not os.path.isdir(p):
            os.mkdir(p)

        with open(filename, "w") as f:
            json.dump(self.__dict__, f)
