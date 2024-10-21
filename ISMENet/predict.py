import os
import json
from pathlib import Path
import datetime
from copy import deepcopy
from PIL import Image
import pandas
import numpy as np
import skimage as sk
from tensorflow.keras.layers import ZeroPadding2D
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.measure import find_contours, approximate_polygon
import ISMENet
from . import utils
from .visualization import _COLORS
from scipy.ndimage import distance_transform_edt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf


now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

VALID_IMAGE_FORMATS = [".jpg", ".png", ".tif", ".bmp", ".jpeg"]
MINAREA = 64
MINSIZE = 8
BGCOLOR = (90, 140, 200)
deltaL = 10


def get_masks(seg_preds, scores, threshold=0.5, weight_by_scores=False):
    """Get labeled masks from segmentation prediction and scores
    threshold: only pixels with value > threshold are considered foreground
    weight_by_scores: \n
    - if True, each predicted masks (each slice of seg_preds) is weighted by its score before argmax
    Final instance are the argmax tensor of this weighted tensor. Note that we keep only masks with values > threshold.
    \n- if False, we just take the argmax of the segmentation score tensor.
    """

    # Sort instance by scores
    if weight_by_scores:
        thresholded_masks = np.where(seg_preds >= threshold, seg_preds, 0)
        weighted_masks = thresholded_masks * scores[:, tf.newaxis, tf.newaxis]
        # add bg slice
        bg_slice = np.zeros((1, thresholded_masks.shape[1], thresholded_masks.shape[2]))
        labeled_masks = np.concatenate([bg_slice, weighted_masks], axis=0)
        # Take argmax (e.g. mask swith higher mask_scores * cls_scores, when two masks overlap)
        labeled_masks = np.argmax(labeled_masks, axis=0)

    # Just take the argmax
    else:
        filt_seg = np.where(seg_preds >= threshold, seg_preds, 0)
        bg_slice = np.zeros((1, filt_seg.shape[1], filt_seg.shape[2]))
        labeled_masks = np.concatenate([bg_slice, filt_seg], axis=0)
        labeled_masks = np.argmax(labeled_masks, axis=0)
        # predicted_instances = np.unique(labeled_masks).size - 1

    return labeled_masks


def box_to_coco(boxes):
    cocoboxes = np.zeros_like(boxes)
    cocoboxes[..., 0:2] = boxes[..., 0:2]
    cocoboxes[..., 2:4] = boxes[..., 2:4] - boxes[..., 0:2]
    return cocoboxes


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, level=0.5, tolerance=0, x_offset=0, y_offset=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode="constant", constant_values=0)
    contours = find_contours(padded_binary_mask, 0.5, fully_connected="high")
    contours = [c - 1 for c in contours]
    for contour in contours:
        contour = close_contour(contour)
        contour = approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            print("Polyshape must have at least 2 points. Skipping")
            continue
        contour = np.flip(contour, axis=1)
        contour[..., 0] += y_offset
        contour[..., 1] += x_offset
        seg = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        seg = [0 if i < 0 else i for i in seg]
        polygons.append(seg)

    return polygons


def predict(
    coco,
    output_dir,
    input_size,
    resolution,
    input_dir,
    model=None,
    model_file=None,
    thresholds=(0.5, 0.5, 0.6),
    crop_to_aspect_ratio=True,
    weight_by_scores=False,
    max_detections=400,
    bgcolor=BGCOLOR,
    minarea=64,
    minsize=4,
    subdirs=False,
    save_imgs=True,
):
    """Instance segmentation + mass estimation on an image or a series of images.

    Inputs:
    coco: a dict containing info, licence and categories in coco format

    input_dir: all the images in input_dir will be processed
    input_size: input siz eof the network. image will be padded/cropped and resized to input_shape
    output_dir: where to put teh results
    model: tensorflow model. if None, the model_file must be provided
    model_file: path to tensorflow saved model
    resolution: res of the input images
    thresholds: a list of thresolds: (t1, t2, t3)
        {"score_threshold": 0.5, "seg_threshold": 0.5, "nms_threshold": 0.6}
    crop_to_aspect_ratio: wether to crop before resizing the image to input_size. If false, the image is padded.
    subdirs [False]: if True, all images in subfolders are processed
    max_detections: lmax number of detected instances (beware the masks tensor shape is [max_instances, Nx, Ny]
    minarea and minsize are the min area of instances (min area of each connected part of an instance) and minsize of boxes. Useful for filtering noise.

    Outputs:
    -COCO object (dict -> saved as json)
    -data dict (dict -> saved as csv)

    Creates on disk:
    - A coco file (json)
    - A csv file containing instance boxes resolution, mass, area, class or other information (if available)
    - label images  in output_dir/labels folder
    - image superimposed with colored labels for vizualisation in output_dir/vizu folder (low-res images)
    - individual instances in output_dir/crops folder

    """
    # TODO: maybe compute regionprops on low res images and scale up results to speed things up ?

    # Load model config and weights
    # tf.config.run_functions_eagerly(True)

    bgcolor = np.array(bgcolor) / 255.0

    if model is None:
        tf.keras.backend.clear_session()
        model_directory = os.path.dirname(model_file)
        with open(os.path.join(model_directory, "config.json"), "r", encoding="utf-8") as configfile:
            config = json.load(configfile)

        # Creating architecture using config file
        modelconf = ISMENet.Config(**config)

        model = ISMENet.model.ISMENetModel(modelconf)
        print(f"Loading model {model_file}...", end="")
        # Loading weights
        model.load_weights(model_file, by_name=False, skip_mismatch=False).expect_partial()
        print("OK")

    # Retrieve images (either in the input dir only or also in all the subdirectiories)
    img_dict = {}

    if not subdirs:
        for entry in os.scandir(input_dir):
            f = entry.name
            if entry.is_file() and os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS:
                img_dict[f] = os.path.join(input_dir, f)

    else:
        for root, _, files in os.walk(input_dir):
            for f in files:
                if os.path.isfile(os.path.join(root, f)) and os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS:
                    img_dict[f] = os.path.join(root, f)

    print("Found", len(img_dict), "images in ", input_dir)

    img_counter = -1

    default_kwargs = {
        "score_threshold": thresholds[0],
        "seg_threshold": thresholds[1],
        "nms_threshold": thresholds[2],
        "max_detections": max_detections,
        "min_arera": minarea,
        "point_nms": False,
        "sigma_nms": 0.5,
    }

    print("Resolution of original images:", resolution, "pixels/mm")

    coco["info"]["description"] = str(input_dir)
    coco["annotations"] = []
    coco["images"] = []

    OUTPUT_DIR = Path(output_dir)
    # CROPS_DIR = OUTPUT_DIR / Path("crops")
    VIZU_DIR = OUTPUT_DIR / Path("vizu")
    LABELS_DIR = OUTPUT_DIR / Path("labels")

    # os.makedirs(CROPS_DIR, exist_ok=True)
    os.makedirs(VIZU_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    data = {
        "baseimg": [],
        "label": [],
        "res": [],
        "class": [],
        "x0": [],
        "x1": [],
        "y0": [],
        "y1": [],
        "area": [],
        "mass": [],
        "filename": [],
        "axis_major_length": [],
        "axis_minor_length": [],
        "feret_diameter_max": [],
        "max_inscribed_radius": [],
    }

    # network input size
    nx, ny = input_size[:2]

    keys = sorted(list(img_dict.keys()))
    # np.random.shuffle(keys)

    for counter, imgname in enumerate(keys):

        impath = img_dict[imgname]
        PILimg = Image.open(impath)

        image = np.array(PILimg) / 255.0
        ini_nx, ini_ny = image.shape[0:2]

        print(
            "Processing image {} ({}/{}), size {}x{}".format(imgname, counter + 1, len(keys), ini_nx, ini_ny),
            end="",
        )

        #  crop the image to the prescribed size before resizing
        if crop_to_aspect_ratio:
            image, _ = utils.crop_to_aspect_ratio(input_size, image)
            if image.shape[0:2] != (ini_nx, ini_ny):
                print(f". Cropping to shape {image.shape[0:2]}", end="")

        fullsize_nx, fullsize_ny = image.shape[:2]

        # Resize without modifying aspect ratio
        resized_image = tf.image.resize(image, (nx, ny), antialias=True, preserve_aspect_ratio=True)

        ratio = resized_image.shape[0] / fullsize_nx
        if resized_image.shape[1] / fullsize_ny != resized_image.shape[0] / fullsize_nx:
            print(
                f"\nWarning: downsampling ratio is not the same in both directions: ({ratio}, {image.shape[1] / fullsize_ny})"
            )

        # Pad if needed
        padding = ((0, 0), (0, 0))
        if fullsize_ny * nx != fullsize_nx * ny:
            resized_image, padding = utils.pad_to_aspect_ratio((nx, ny), resized_image)

        img_counter += 1
        coco["images"].append(
            {
                "file_name": imgname,
                "coco_url": "",
                "height": PILimg.height,
                "width": PILimg.width,
                "date_captured": "",
                "id": img_counter,
            }
        )

        # Get network predictions
        seg_preds, scores, cls_labels, densities = model(
            resized_image[tf.newaxis, ...], training=False, **default_kwargs
        )

        seg_preds = seg_preds[0].numpy()
        scores = scores[0].numpy()
        cls_labels = cls_labels[0].numpy()
        densities = densities[0].numpy()

        if scores.size <= 0:  # No detection !
            print("...OK. No instance detected !")
            continue

        # Get labeled mask (int)
        labeled_masks = get_masks(
            seg_preds,
            scores,
            threshold=default_kwargs["seg_threshold"],
            weight_by_scores=weight_by_scores,
        )

        final_labels = np.unique(labeled_masks)[1:] - 1  # skipping 0
        # some slices are "empty" -> all pixel values are < sge_threshold
        if final_labels.size != cls_labels.size:
            scores = scores[final_labels]
            cls_labels = cls_labels[final_labels]
            densities = densities[final_labels]

        # mask is usually smaller than input_size.
        # We compute the actual downsampling from the input image size to get the resolution and extract the unpadded mask
        mask_stride = nx // labeled_masks.shape[0]
        if mask_stride != ny // labeled_masks.shape[1]:
            print(
                f"Warning: mask stride is not equal in both directions: ({mask_stride}, {ny / labeled_masks.shape[1]})"
            )

        # Resize and delete padding
        if np.array(padding).sum() > 0:
            # if the image is padded, we must first upscale the mask to imsize then delete padding and upscale to input image size
            fullsize_masks = tf.image.resize(
                labeled_masks[..., np.newaxis], (nx, ny), antialias=False, method="nearest"
            )[..., 0]
            if padding[0][1] > 0:
                fullsize_masks = fullsize_masks[padding[0][0] : -padding[0][1], ...]
            if padding[1][1] > 0:
                fullsize_masks = fullsize_masks[..., padding[1][0] : -padding[1][1]]

            fullsize_masks = tf.image.resize(
                fullsize_masks[..., np.newaxis], (fullsize_nx, fullsize_ny), antialias=False, method="nearest"
            )[..., 0]

        else:
            fullsize_masks = tf.image.resize(
                labeled_masks[..., np.newaxis], (fullsize_nx, fullsize_ny), antialias=False, method="nearest"
            )[..., 0]

        fullsize_masks = fullsize_masks.numpy()

        # Extract bboxes
        region_properties = regionprops(fullsize_masks, extra_properties=(max_inscribed_radius_func,))
        labels = np.array([prop["label"] for prop in region_properties])

        if labels.shape[0] <= 0:
            print("...OK. No instance detected !")
            continue

        if labels.size != cls_labels.size:
            print(labels.size, cls_labels.size)
            print("WARNING: number of detected regions is not the same as number of predicted instances")

        # filtering small instances [beware, it means that labels may not be continous]
        # filtered_properties = [(i, prop) for i, prop in enumerate(region_properties) if prop["area"] >= minarea]
        # idx, filtered_properties = zip(*filtered_properties)
        # idx = np.array(idx)
        # idx = np.array([i for i, prop in enumerate(region_properties) if prop["area"] >= minarea])
        # labels = labels[idx]
        fullsize_masks = np.where(np.isin(fullsize_masks, labels), fullsize_masks, 0)
        boxes = np.array([prop["bbox"] for prop in region_properties])

        # Saving props
        mask_resolution = resolution * ratio / mask_stride
        area = [prop["area"] for prop in region_properties]
        axis_major_length = [prop["axis_major_length"] for prop in region_properties]
        axis_minor_length = [prop["axis_minor_length"] for prop in region_properties]
        feret_diameter_max = [prop["feret_diameter_max"] for prop in region_properties]
        max_inscribed_radius = [prop["max_inscribed_radius_func"] for prop in region_properties]

        data["area"].extend(area)
        data["axis_major_length"].extend(axis_major_length)
        data["axis_minor_length"].extend(axis_minor_length)
        data["feret_diameter_max"].extend(feret_diameter_max)
        data["max_inscribed_radius"].extend(max_inscribed_radius)

        masses = densities / (100 * mask_resolution**2)
        classes = cls_labels

        data["mass"].extend(masses.tolist())
        data["class"].extend(classes.tolist())

        if save_imgs:
            # resize masks and image for vizualisation
            small_masks = tf.image.resize(
                fullsize_masks[..., np.newaxis],
                (fullsize_nx // mask_stride, fullsize_ny // mask_stride),
                antialias=False,
                method="nearest",
            )[..., 0].numpy()
            small_image = tf.image.resize(
                image, (fullsize_nx // mask_stride, fullsize_ny // mask_stride), antialias=True
            ).numpy()
            vizuname = "VIZU-{}.jpg".format(os.path.splitext(imgname)[0])
            bd = sk.segmentation.find_boundaries(small_masks, connectivity=2, mode="inner", background=0)
            vizu = label2rgb(small_masks, small_image, alpha=0.25, bg_label=0, colors=_COLORS, saturation=1)
            vizu = np.where(bd[..., np.newaxis], (0, 0, 0), vizu)
            vizu = np.around(255 * vizu).astype(np.uint8)
            Image.fromarray(vizu).save(os.path.join(VIZU_DIR, vizuname))

        print(f"...OK. Found {len(labels)} instances. ")

        cocoboxes = box_to_coco(boxes)

        # saving coco instance data
        for i, prop in enumerate(region_properties):

            box = prop["bbox"]

            data["baseimg"].append(imgname)
            data["label"].append(labels[i])
            data["res"].append(resolution)
            data["x0"].append(box[0])
            data["x1"].append(box[2])
            data["y0"].append(box[1])
            data["y1"].append(box[3])

            # Create COCO annotation
            polys = binary_mask_to_polygon(
                prop["image"],
                level=0.5,
                x_offset=cocoboxes[i, 0],
                y_offset=cocoboxes[i, 1],
            )

            coco["annotations"].append(
                {
                    "segmentation": polys,
                    "area": int(data["area"][i]),
                    "iscrowd": 0,
                    "image_id": img_counter,
                    "bbox": [int(b) for b in cocoboxes[i]],
                    "category_id": int(classes[i]),
                    "id": i,
                }
            )

        # Save labels
        labelname = "{}.png".format(os.path.splitext(imgname)[0])
        Image.fromarray(fullsize_masks.astype(np.uint16)).save(os.path.join(LABELS_DIR, labelname))

    info_filepath = os.path.join(OUTPUT_DIR, "info.json")
    config = {"SEGNET": str(model_file), "DATA_DIR": str(input_dir)}
    with open(info_filepath, "w", encoding="utf-8") as jsonconfig:
        json.dump(config, jsonconfig)

    print("Saving COCO in ", OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "coco_annotations.json"), "w", encoding="utf-8") as jsonfile:
        json.dump(coco, jsonfile)

    df = pandas.DataFrame().from_dict(data)
    df = df.set_index("filename")
    df.to_csv(os.path.join(OUTPUT_DIR, "annotations.csv"), na_rep="nan", header=True)

    # return COCO and data dict
    return coco, data


def stream_predict(
    imsize,
    input_dir,
    output_dir,
    resolution,
    model=None,
    model_file=None,
    crop_to_aspect_ratio=True,
    deltaL=5,
    thresholds=(0.5, 0.5, 0.6),
    weight_by_scores=False,
    max_detections=400,
    bgcolor=BGCOLOR,
    minarea=MINAREA,
    minsize=MINSIZE,
    subdirs=True,
    save_imgs=True,
):
    """Segmentation d'une série d'images qui se suivent selon la direction x
    A SOLOv2 network is used to preddict the instance masks and class.
    Mass of object is predicted as the product between a new shared head and a new unified mask feature map
    Objects straddling two frames are used to determine the two overlap area.
    An image containing only objects touching the edges is formed from the overlap area.
    Entire objects are then extracted from this intermediate image.
    Dans le cas où il n'y a qu'une image, les objets touchant le bord inférieur ne sont donc pas traités.
    TODO:Optimisation: pour des raisons de simplicité, les objets sont mesurés sur l'image des masques redimensionnée à la taille de l'image d'entrée (!= dimensions d'enntrée du réseau)
        - on pourrait envisager d'utiliser cucim.skimage pour accélérer les calculs des regionprops
        - On pourrait également réduire la taille des visualisations sauvegardées
    TODO:implement minsize thresholding

    Inputs:
    imsize: size of input images
    input_dir: all the images in input_dir will be processed
    output_dir: where to put teh results
    resolution: res of the input images
    model: an instance of a tf model
    model_file: path to a tensorflow saved model
    crop_to_aspect_ratio: wether to crop before resizing the image to input_size. If false, the image is only padded if needed.
    deltaL: if a box is less than deltaL from the image edge, the object is considered to be touching the edge.
    trhesolds: a list of thresolds: (t1, t2, t3)
        {"score_threshold": 0.5, "seg_threshold": 0.5, "nms_threshold": 0.6}
    subdirs [False]: if True, all images in subfolders are processed
    max_detections: lmax number of detected instances (beware the masks tensor shape is [max_instances, Nx, Ny]
    minarea and minsize are the min area of instances (min area of each connected part of an instance) and minsize of boxes. Useful for filtering noise.

    Outputs:
    -data dict

    Creates on disk:
    - A csv file containing instance boxes resolution, mass, area, class or other information (if available)
    - label images in output_dir/labels folder
    - image superimposed with colored labels for vizualisation in output_dir/vizu folder (low-res images)
    - newly created overlap bands images (full resolution) in output_dir/images
    """

    # Load model config and weights
    # tf.config.run_functions_eagerly(True)
    tf.keras.backend.clear_session()

    bgcolor = np.array(bgcolor) / 255.0

    if model is None:
        tf.keras.backend.clear_session()
        model_directory = os.path.dirname(model_file)
        if model_directory is None:
            print("Please give a model file or tf model")
            return
        model_directory = os.path.dirname(model_file)
        with open(os.path.join(model_directory, "config.json"), "r", encoding="utf-8") as configfile:
            config = json.load(configfile)

        # Creating architecture using config file
        modelconf = ISMENet.Config(**config)

        detector = ISMENet.model.ISMENetModel(modelconf)
        print(f"Loading model {model_file}...", end="")
        # Loading weights
        detector.load_weights(model_file, by_name=False, skip_mismatch=False).expect_partial()
        print("OK")

    else:
        detector = model

    # Retrieve images
    img_dict = {}
    if not subdirs:
        for entry in os.scandir(input_dir):
            f = entry.name
            if entry.is_file() and os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS:
                img_dict[f] = os.path.join(input_dir, f)

    else:
        for root, _, files in os.walk(input_dir):
            for f in files:
                if os.path.isfile(os.path.join(root, f)) and os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS:
                    img_dict[f] = os.path.join(root, f)

    print("Found", len(img_dict), "images in ", input_dir)

    img_counter = -1

    default_kwargs = {
        "score_threshold": thresholds[0],
        "seg_threshold": thresholds[1],
        "nms_threshold": thresholds[2],
        "max_detections": max_detections,
        "min_area": minarea,
    }

    print("Resolution of original image:", resolution, "pixels/mm")

    OUTPUT_DIR = Path(output_dir)
    VIZU_DIR = OUTPUT_DIR / Path("vizu")
    LABELS_DIR = OUTPUT_DIR / Path("labels")
    OVERLAPPING_IMGS_DIR = OUTPUT_DIR / Path("overlap")

    os.makedirs(VIZU_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(OVERLAPPING_IMGS_DIR, exist_ok=True)

    data = {
        "baseimg": [],
        "label": [],
        "res": [],
        "class": [],
        "x0": [],
        "x1": [],
        "y0": [],
        "y1": [],
        "area": [],
        "mass": [],
        "filename": [],
        "axis_major_length": [],
        "axis_minor_length": [],
        "feret_diameter_max": [],
        "max_inscribed_radius": [],
    }

    nx, ny = imsize[:2]

    # keys = sorted(list(img_dict.keys()))
    img_paths = sorted(list(img_dict.values()))
    # np.random.shuffle(keys)

    for counter, full_path in enumerate(img_paths):

        imgname = os.path.basename(full_path)

        print(
            "Processing image {} ({}/{})".format(imgname, counter, len(img_dict)),
            end="",
        )

        # impath = img_dict[imgname]
        # PILimg = Image.open(impath)
        # fullsize_image = np.array(PILimg) / 255.0

        fullsize_image = tf.io.decode_image(tf.io.read_file(full_path), channels=3, dtype=tf.float32).numpy()

        if crop_to_aspect_ratio:
            fullsize_image, _ = utils.crop_to_aspect_ratio(imsize, fullsize_image)

        fullsize_nx, fullsize_ny = fullsize_image.shape[:2]

        image = tf.image.resize(fullsize_image, (nx, ny), antialias=True, method="bilinear", preserve_aspect_ratio=True)

        # downsampling ratio between input image and the network input size
        ratio = image.shape[0] / fullsize_nx
        if image.shape[1] / fullsize_ny != ratio:
            print(
                f"Warning: downscaling ratio is not the same in both directions: ({ratio}, {image.shape[1] / fullsize_ny})"
            )
        padding = ((0, 0), (0, 0))
        if fullsize_ny / fullsize_nx != ny / nx:
            image, padding = utils.pad_to_aspect_ratio((nx, ny), image)

        img_counter += 1

        # Get network predictions
        seg_preds, scores, cls_labels, densities = detector(image[tf.newaxis, ...], training=False, **default_kwargs)

        # because batch size = 1
        seg_preds = seg_preds[0].numpy()
        scores = scores[0].numpy()
        cls_labels = cls_labels[0].numpy()
        densities = densities[0].numpy()

        if scores.size <= 0:  # No detection !
            continue

        # Get labeled mask (int)
        labeled_masks = get_masks(
            seg_preds,
            scores,
            threshold=default_kwargs["seg_threshold"],
            weight_by_scores=weight_by_scores,
        )

        mask_stride = nx // labeled_masks.shape[0]

        if mask_stride != ny // labeled_masks.shape[1]:
            print(
                f"Warning: mask stride is not equal in both directions: ({mask_stride}, {ny / labeled_masks.shape[1]})"
            )

        # Resize and delete padding
        if np.array(padding).sum() > 0:
            # if the image is padded, we must first upscale the mask to imsize then delete padding and upscale to input image size
            fullsize_masks = tf.image.resize(
                labeled_masks[..., np.newaxis], (nx, ny), antialias=False, method="nearest"
            )[..., 0]
            if padding[0][1] > 0:
                fullsize_masks = fullsize_masks[padding[0][0] : -padding[0][1], ...]
            if padding[1][1] > 0:
                fullsize_masks = fullsize_masks[..., padding[1][0] : -padding[1][1]]

            fullsize_masks = tf.image.resize(
                fullsize_masks[..., np.newaxis], (fullsize_nx, fullsize_ny), antialias=False, method="nearest"
            )[..., 0]

        else:
            fullsize_masks = tf.image.resize(
                labeled_masks[..., np.newaxis], (fullsize_nx, fullsize_ny), antialias=False, method="nearest"
            )[..., 0]

        fullsize_masks = fullsize_masks.numpy()

        region_properties = regionprops(fullsize_masks, extra_properties=(max_inscribed_radius_func,))
        pred_boxes = np.array([prop["bbox"] for prop in region_properties])
        labels = np.array([prop["label"] for prop in region_properties])

        # middle indexes: indexes of boxes not touching edges,
        # up indexes: boxes touching upper edge (i.e. x->0)
        # TODO: check if an object touches both edges (normalement non !)

        if counter == 0:
            # on extrait toutes les boites sauf celle touchant le "bas" de l'image (x=nx-padx1)
            middle_indexes = np.where(pred_boxes[:, 2] < fullsize_nx - deltaL)
        elif counter == len(img_dict) - 1:
            # On extrait toutes les boites sauf celles touchant le "haut" (x=0+padx0)
            middle_indexes = np.where(pred_boxes[:, 0] > deltaL)
        else:
            # On extrait toutes les boites sauf celles touchant le "haut" (x=0) ET le bas (x=nx)
            middle_indexes = np.where((pred_boxes[:, 2] < fullsize_nx - deltaL) & (pred_boxes[:, 0] > deltaL))

        middle_labels = []
        if middle_indexes[0].size > 0:
            middle_labels = labels[middle_indexes]

        # le "haut de l'image" correspond à nx=0
        up_indexes = np.where(pred_boxes[:, 0] <= deltaL)
        up_labels = labels[up_indexes]
        bottom_indexes = np.where(pred_boxes[:, 2] >= fullsize_nx - deltaL)
        print(f"...OK. Found {scores.size} instances. ", end="")
        print(f"{up_indexes[0].size} objects touching the upper edge and {bottom_indexes[0].size} touching the bottom.")

        if middle_indexes[0].size > 0:

            # delete labels of objects touching the edges
            fullsize_masks = np.where(np.isin(fullsize_masks, middle_labels), fullsize_masks, 0)

            # Save labels without objects touching the edges
            labelname = "{}.png".format(os.path.splitext(imgname)[0])
            Image.fromarray(fullsize_masks.astype(np.uint16)).save(os.path.join(LABELS_DIR, labelname))

            # Delete objects *not* touching the edges in the fullsize image
            filtered_image = np.where(fullsize_masks[..., np.newaxis] == 0, fullsize_image, BGCOLOR)

            middle_pred_boxes = pred_boxes[middle_indexes]

            # Save properties of objects not touching the edges
            mask_ratio = ratio / mask_stride
            mask_resolution = resolution * mask_ratio
            area = np.array([prop["area"] for prop in region_properties])[middle_indexes]
            axis_major_length = np.array([prop["axis_major_length"] for prop in region_properties])[middle_indexes]
            axis_minor_length = np.array([prop["axis_minor_length"] for prop in region_properties])[middle_indexes]
            feret_diameter_max = np.array([prop["feret_diameter_max"] for prop in region_properties])[middle_indexes]
            max_inscribed_radius = np.array([prop["max_inscribed_radius_func"] for prop in region_properties])[
                middle_indexes
            ]
            masses = densities[middle_indexes] / (100 * mask_resolution**2)
            classes = cls_labels[middle_indexes]

            data["baseimg"].extend([imgname] * labels.size)
            data["label"].extend(middle_labels.tolist())
            data["res"].extend([resolution] * labels.size)
            data["x0"].extend(middle_pred_boxes[:, 0].tolist())
            data["x1"].extend(middle_pred_boxes[:, 2].tolist())
            data["y0"].extend(middle_pred_boxes[:, 1].tolist())
            data["y1"].extend(middle_pred_boxes[:, 3].tolist())
            data["area"].extend(area.tolist())
            data["mass"].extend(masses.tolist())
            data["class"].extend(classes.tolist())
            data["axis_major_length"].extend(axis_major_length.tolist())
            data["axis_minor_length"].extend(axis_minor_length.tolist())
            data["feret_diameter_max"].extend(feret_diameter_max.tolist())
            data["max_inscribed_radius"].extend(max_inscribed_radius.tolist())

            if save_imgs:
                vizuname = "VIZU-{}.jpg".format(os.path.splitext(imgname)[0])
                bd = sk.segmentation.find_boundaries(fullsize_masks, connectivity=2, mode="inner", background=0)
                vizu = label2rgb(fullsize_masks, filtered_image, alpha=0.25, bg_label=0, colors=_COLORS, saturation=1)
                vizu = np.where(bd[..., np.newaxis], (0, 0, 0), vizu)
                vizu = np.around(255 * vizu).astype(np.uint8)
                Image.fromarray(vizu).save(os.path.join(VIZU_DIR, vizuname))

        elif up_indexes.size > 0:

            print(", all detected instances touch the edges")

        # Create overlapping image using previous bottom image and current up image
        # note that we create an image with the same resolution as the input image !

        if counter > 0:
            # hauteur bande de recouvrement haute et basse
            # ici bottom_boxes correspond aux objets détectée sur l'image précédente
            fullsize_boxes_up = pred_boxes[up_indexes]

            # S'il n'y a pas d'objets touchant les bords en haut de l'image actuelle ou en bas de la précédente alors on passe
            if not (fullsize_boxes_up.size == 0 and prev_bottom_boxes.size == 0):
                # Up correspond au haut de l'image cad coords à partir de 0
                if fullsize_boxes_up.size > 0:
                    lxup = fullsize_boxes_up[:, 2].max()
                else:
                    lxup = 0

                # bottom correspond au "bas" de l'image (vers les coords croissantes)
                if prev_bottom_boxes.size > 0:
                    lxdown = prev_bottom_boxes[:, 0].min()
                else:
                    lxdown = fullsize_nx

                if lxup == 0 and lxdown < fullsize_nx:
                    fullsize_overlap_image = prev_filtered_image[lxdown:, ...]

                elif lxup > 0 and lxdown == fullsize_nx:
                    fullsize_overlap_image = filtered_image[0:lxup, ...]
                else:
                    fullsize_overlap_image = np.concatenate(
                        [
                            prev_filtered_image[lxdown:, ...],
                            filtered_image[0:lxup, ...],
                        ],
                        0,
                    )

                o_fullsize_nx, o_fullsize_ny = fullsize_overlap_image.shape[0], fullsize_overlap_image.shape[1]

                # resize the image to the network's input size
                overlap_image, o_padding = utils.pad_to_aspect_ratio((nx, ny), fullsize_overlap_image)
                overlap_image = tf.image.resize(overlap_image, (nx, ny), antialias=True)
                o_ratio = overlap_image.shape[0] / o_fullsize_nx

                # Get network predictions
                o_seg_preds, o_scores, o_cls_labels, o_densities = detector(image[tf.newaxis, ...], training=False, **default_kwargs)

                # Because batchsize=1
                o_seg_preds = o_seg_preds[0].numpy()
                o_scores = o_scores[0].numpy()
                o_cls_labels = o_cls_labels[0].numpy()
                o_densities = o_densities[0].numpy()

                o_labeled_masks = get_masks(
                    o_seg_preds,
                    o_scores,
                    threshold=default_kwargs["seg_threshold"],
                    weight_by_scores=weight_by_scores,
                )

                if o_scores.size > 0:

                    o_imgname = "OVERLAP_{}-{}.jpg".format(
                        os.path.splitext(prev_imgname)[0], os.path.splitext(imgname)[0]
                    )
                    print("Processing overlapping image {}...".format(o_imgname), end="")

                    # Upscale the mask image and delete padding
                    if np.array(padding).sum() > 0:
                        # if the image is padded, we must first upscale the mask to imsize then delete padding and upscale to input image size
                        o_fullsize_masks = tf.image.resize(
                            o_labeled_masks[..., np.newaxis], (nx, ny), antialias=False, method="nearest"
                        )[..., 0]
                        if o_padding[0][1] > 0:
                            o_fullsize_masks = o_fullsize_masks[o_padding[0][0] : -o_padding[0][1], ...]
                        if o_padding[1][1] > 0:
                            o_fullsize_masks = o_fullsize_masks[..., o_padding[1][0] : -o_padding[1][1]]
                        o_fullsize_masks = tf.image.resize(
                            o_fullsize_masks[..., np.newaxis],
                            (o_fullsize_nx, o_fullsize_ny),
                            antialias=False,
                            method="nearest",
                        )[..., 0]
                    else:
                        o_fullsize_masks = tf.image.resize(
                            o_labeled_masks[..., np.newaxis],
                            (o_fullsize_nx, o_fullsize_ny),
                            antialias=False,
                            method="nearest",
                        )[..., 0]

                    o_fullsize_masks = o_fullsize_masks.numpy()
                    # save the fullsize  overlap image
                    o_PILimg = Image.fromarray(np.around(fullsize_overlap_image * 255).astype(np.uint8))
                    o_PILimg.save(os.path.join(OVERLAPPING_IMGS_DIR, o_imgname), quality=95)

                    # Save labels
                    o_labelname = "{}.png".format(os.path.splitext(o_imgname)[0])
                    Image.fromarray(o_fullsize_masks.astype(np.uint16)).save(os.path.join(LABELS_DIR, o_labelname))

                    # Compute regionprops
                    o_region_properties = regionprops(o_fullsize_masks, extra_properties=(max_inscribed_radius_func,))
                    o_pred_boxes = np.array([prop["bbox"] for prop in o_region_properties])
                    o_labels = np.array([prop["label"] for prop in o_region_properties])

                    # Save properties of objects not touching the edges
                    mask_ratio = o_ratio / mask_stride
                    mask_resolution = resolution * mask_ratio
                    area = np.array([prop["area"] for prop in o_region_properties])
                    axis_major_length = np.array([prop["axis_major_length"] for prop in o_region_properties])
                    axis_minor_length = np.array([prop["axis_minor_length"] for prop in o_region_properties])
                    feret_diameter_max = np.array([prop["feret_diameter_max"] for prop in o_region_properties])
                    max_inscribed_radius = np.array([prop["max_inscribed_radius_func"] for prop in o_region_properties])
                    masses = o_densities[middle_indexes] / (100 * mask_resolution**2)
                    classes = cls_labels[middle_indexes]

                    data["baseimg"].extend([o_imgname] * o_labels.size)
                    data["label"].extend(o_labels.tolist())
                    data["res"].extend([resolution] * o_labels.size)
                    data["x0"].extend(o_pred_boxes[:, 0].tolist())
                    data["x1"].extend(o_pred_boxes[:, 2].tolist())
                    data["y0"].extend(o_pred_boxes[:, 1].tolist())
                    data["y1"].extend(o_pred_boxes[:, 3].tolist())
                    data["area"].extend(area.tolist())
                    data["mass"].extend(masses.tolist())
                    data["class"].extend(classes.tolist())
                    data["axis_major_length"].extend(axis_major_length.tolist())
                    data["axis_minor_length"].extend(axis_minor_length.tolist())
                    data["feret_diameter_max"].extend(feret_diameter_max.tolist())
                    data["max_inscribed_radius"].extend(max_inscribed_radius.tolist())

                    if save_imgs:
                        vizuname = "VIZU-{}.jpg".format(os.path.splitext(o_imgname)[0])
                        bd = sk.segmentation.find_boundaries(
                            o_fullsize_masks, connectivity=2, mode="inner", background=0
                        )
                        vizu = label2rgb(
                            o_fullsize_masks,
                            fullsize_overlap_image,
                            alpha=0.25,
                            bg_label=0,
                            colors=_COLORS,
                            saturation=1,
                        )
                        vizu = np.where(bd[..., np.newaxis], (0, 0, 0), vizu)
                        vizu = np.around(255 * vizu).astype(np.uint8)
                        Image.fromarray(vizu).save(os.path.join(VIZU_DIR, vizuname))

        # paramètres utilisés dans l'itération suivante
        bottom_indexes = np.where(pred_boxes[:, 2] >= fullsize_nx - deltaL)
        # print("bottom boxes", pred_boxes[bottom_indexes])
        # bottom_labels = labels[bottom_indexes]
        prev_bottom_boxes = pred_boxes[bottom_indexes]
        prev_filtered_image = filtered_image
        prev_imgname = imgname

    info_filepath = os.path.join(OUTPUT_DIR, "info.json")
    config = {"SEGNET": str(model_file), "DATA_DIR": str(input_dir)}
    with open(info_filepath, "w", encoding="utf-8") as jsonconfig:
        json.dump(config, jsonconfig)

    df = pandas.DataFrame().from_dict(data)
    # df = df.set_index("filename")
    df.to_csv(os.path.join(OUTPUT_DIR, "annotations.csv"), na_rep="nan", header=True)

    # return data dict
    return data


def max_inscribed_radius_func(mask):

    return distance_transform_edt(np.pad(mask, 1)).max()
