import argparse

import cv2
import numpy as np
import re
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, VisImage
from UniT.configs import add_config

def test_model(base_model, images):
    cfg: CfgNode = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_model))
    # add_config(cfg)
    # cfg.merge_from_file(base_model)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_model)
    cfg.MODEL.DEVICE = 'cpu'
    predictor: DefaultPredictor = DefaultPredictor(cfg)

    result_images = []

    image_file: str
    for image_file in images:
        print("image file: ", image_file)
        img: np.ndarray = cv2.imread(image_file)

        output: Instances = predictor(img)["instances"]
        v = Visualizer(img[:, :, ::-1],
                       MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                       scale=1.0)
        result: VisImage = v.draw_instance_predictions(output.to("cpu"))
        result_image: np.ndarray = result.get_image()[:, :, ::-1]

        # get file name without extension, -1 to remove "." at the end
        out_file_name: str = re.search(r"(.*)\.", image_file).group(0)[:-1]
        out_file_name += "_processed.png"

        cv2.imwrite(out_file_name, result_image)
        result_images.append(result_image)

    return result_images

if __name__ == "__main__":
    base_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    # base_model = "../configs/VOC/VOC-RCNN-101-C4-split1.yaml"
    # base_model = "../configs/Base-RCNN-C4.yaml"

    images = [r'..\demo\images\select_images\bus.jpg']
    result_images = test_model(base_model, images)