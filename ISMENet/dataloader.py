import os
from PIL import Image
from copy import copy
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
import datetime
import glob
import pandas
from . import utils

VALID_IMAGE_FORMATS = [".jpg", ".png", ".tif", ".bmp"]


class csvDataLoader:

    def __init__(
        self,
        annotations,
        input_shape,
        crop_to_aspect_ratio=True,
        mask_stride=4,
        cls_to_idx=None,
        mininst=1,
        maxinst=400,
        max_inst_per_cls=None,
        exclude=[],
        minres=20,
        shuffle=True,
        seed=None,
        filter_annotations=True,
        augmentation_func=None,
        num_parallel_calls=tf.data.AUTOTUNE,
        **kwargs,
    ):
        """
        DataLoader to create a train and a validation tf dataset from a pandas dataframe file\n
        The csv file must contains the following columns:\n
        label, folder, baseimg, area, x0, y0, x1, y1, class, res, mass, gt_mass
        Params:
        annotations: a pandas dataframe
        input_shape [nx, ny]: image are cropped to the target shape.
            The input image is cropped from both opposite side im[cropx:-cropx, cropy:-cropy, :].
        crop_to_aspect_ratio: keep the original aspect ratio
        mask_stride: strides of mask image (depend on the model architecture)
        cls_to_idx: a dictionnary mapping class names to an integer. It is generated automatically if cls_to_idx is None
        Note that the class idx must begin at 1 (0 is background) and must be successive integers
        mininst: minimum number of instances. Images with a lower number are discarded
        maxinst: discard image containing more than maxinst objects
        max_inst_per_cls: ensure that the final datset does not contain more than max_inst_per_cls objects of each class
        exclude: [] discard images than contain the classes in the list
        augmentation_func: function to augment image (no cropping, resizing or geometric transform !!! Only brightness/color/noise)
        """

        self.annotations = annotations.reset_index(drop=True)  # store current filtered annotations
        self.base_annotations = annotations.copy(deep=True)  # store all unfiltered annotations
        self.input_shape = input_shape
        self.cls_to_idx = cls_to_idx
        self.exclude = exclude
        self.mininst = mininst
        self.maxinst = maxinst
        self.max_inst_per_cls = max_inst_per_cls
        self.num_parallel_calls = num_parallel_calls
        self.mask_stride = mask_stride
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.ratio = input_shape[0] / input_shape[1]
        self.minres = minres
        self.seed = seed
        self.augmentation_func = augmentation_func

        self.train_basenames = []
        self.valid_basenames = []
        self.train_dataset = None
        self.valid_dataset = None

        self.train_class_counts = {}
        self.valid_class_counts = {}

        if filter_annotations:
            self.filter_annotations()

        self.classes = self.annotations["class"].unique().tolist()
        self.classes.sort()

        # define class index if not given
        if self.cls_to_idx is None:
            self.cls_to_idx = {c: i + 1 for i, c in enumerate(self.classes)}

    @classmethod
    def from_file(cls, annfile, filename, augmentation_func=None, num_parallel_calls=tf.data.AUTOTUNE):
        """
        Load annotations from csv file and create train and valid datasets based on the file containing image names
        """

        with open(filename, "r", encoding="utf-8") as jsonfile:
            data = json.load(jsonfile)

        anns = pandas.read_csv(annfile, sep=None, engine="python")

        instance = cls(
            annotations=anns,
            filter_annotations=False,
            augmentation_func=augmentation_func,
            num_parallel_calls=num_parallel_calls,
            **data,
        )

        instance.train_class_counts = data.get("train_class_counts", {})
        instance.valid_class_counts = data.get("valid_class_counts", {})

        instance.train_basenames = data.get("train", [])
        instance.valid_basenames = data.get("valid", [])
        instance.train_dataset = instance.build(names=instance.train_basenames)
        instance.valid_dataset = instance.build(names=instance.valid_basenames)

        instance.basenames = data.get("train", []) + data.get("valid", [])

        return instance

    def save(self, filename, id=None):
        """Save current annotations and filenames in train and valid dataset"""

        if id is None:
            id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        data = {"dataset_id": id}
        data["exclude"] = self.exclude
        data["ratio"] = self.ratio
        data["crop_to_ar"] = self.crop_to_aspect_ratio
        data["mask_stride"] = self.mask_stride
        data["minres"] = self.minres
        data["input_shape"] = list(self.input_shape)
        data["cls_to_idx"] = self.cls_to_idx
        data["mininst"] = self.mininst
        data["maxinst"] = self.maxinst
        data["max_inst_per_cls"] = self.max_inst_per_cls
        data["seed"] = self.seed
        data["train_class_counts"] = {str(k): int(v) for k, v in self.train_class_counts.items()}
        data["valid_class_counts"] = {str(k): int(v) for k, v in self.valid_class_counts.items()}

        if self.train_basenames is not None:
            data["train"] = self.train_basenames
        if self.valid_basenames is not None:
            data["valid"] = self.valid_basenames

        with open(filename + ".json", "w", encoding="utf-8") as jsonfile:
            # json.dump(data, jsonfile, indent=4)
            jsonfile.write(json.dumps(data, indent=4))

        self.annotations.to_csv(
            filename + ".csv", index=False, index_label="baseimg", na_rep="nan", mode="w", header=True
        )

    def update_info(self):
        """Recompute the number of instances per class in train and valid ds"""

        self.train_class_counts = {c: 0 for c in self.classes}

        for basename in self.train_basenames:
            inst_dataframe = self.annotations.loc[self.annotations["baseimg"] == basename]
            inst_cls_value_count = {
                c: n + self.train_class_counts.get(c, 0) for c, n in inst_dataframe["class"].value_counts().items()
            }
            self.train_class_counts.update(inst_cls_value_count)

        self.valid_class_counts = {c: 0 for c in self.classes}

        for basename in self.valid_basenames:
            inst_dataframe = self.annotations.loc[self.annotations["baseimg"] == basename]
            inst_cls_value_count = {
                c: n + self.valid_class_counts.get(c, 0) for c, n in inst_dataframe["class"].value_counts().items()
            }
            self.valid_class_counts.update(inst_cls_value_count)

    def _populate_set(self, basenames, annotations, nmax, exclude=[], not_counting=[], max_reuse=3, shuffle=True):

        """ Génère un ensemble d'images (provenant du dataframe annotations) selon differents critères
        renvoie la liste des noms et un dictionnaire du nombre d'instance par classe
        """

        cls_inst_counter = {c: 0 for c in self.classes}
        img_reuse_counter = {b: 0 for b in basenames}

        rng = np.random.default_rng(self.seed)

        results_basenames = []

        add_instances = True
        it_counter = 0
        while add_instances:
            it_counter += 1
            add_instances = False
            img_left_per_class = {c: 0 for c in self.classes}

            if shuffle:
                rng.shuffle(basenames)

            print(f"Adding images, iteration {it_counter}     ")  # , end="\r")

            for basename in basenames:

                if img_reuse_counter[basename] <= max_reuse:

                    inst_dataframe = annotations.loc[annotations["baseimg"] == basename]
                    inst_cls_value_count = {c: n for c, n in inst_dataframe["class"].value_counts().items()}
                    temp_dict = {}
                    skipit = False

                    for c, n in inst_cls_value_count.items():
                        if c in exclude:
                            skipit = True
                            break
                        temp_dict[c] = n + cls_inst_counter.get(c, 0)
                        if temp_dict[c] > nmax[c]:
                            skipit = True
                            break

                    if not skipit:
                        results_basenames.append(basename)
                        cls_inst_counter.update(temp_dict)
                        # print(f"Adding image {basename} {temp_dict}")
                        img_reuse_counter[basename] += 1
                        if img_reuse_counter[basename] < max_reuse:
                            for c, n in cls_inst_counter.items():
                                img_left_per_class[c] += 1

            # s'il reste des images on vérifie si on doit encore ajouter des objets pour atteindre l'objectif
            for c, n in cls_inst_counter.items():
                if c not in not_counting and n < nmax[c] and img_left_per_class[c] > 0:
                    add_instances = True
                    break

        if len(results_basenames) > 0:
            print(
                f"Ended in {it_counter} iterations. Added {len(results_basenames)} images. Instances per class: {cls_inst_counter} \n"
            )

        return results_basenames, cls_inst_counter

    def filter_set(self, ds_id=None, hasmass=True, method='notin'):

        """filter an existing train/valid dataset using ds id and mass
        """
        # Filter annotation dataframe

        if hasmass:
            filtered_anns = self.annotations.loc[np.isfinite(self.annotations["mass"])]
        else:
            filtered_anns = self.annotations

        if  ds_id is not None:
            if method == 'notin':
                filtered_anns = filtered_anns.loc[filtered_anns["id"] != ds_id]
            else:
                filtered_anns = filtered_anns.loc[filtered_anns["id"] == ds_id]

        filtered_basenames = filtered_anns["baseimg"].unique().tolist()

        new_train_basenames = [b for b in self.train_basenames if b in filtered_basenames]
        new_valid_basenames = [b for b in self.valid_basenames if b in filtered_basenames]

        self.train_basenames = new_train_basenames
        self.valid_basenames = new_valid_basenames
        self.update_info()
        self.build(self.train_basenames)
        self.build(self.valid_basenames)

    def create_set(
        self,
        n=100,
        set='valid',
        exclude=[],
        not_counting=[],
        max_reuse=3,
        seed=None,
        append=False,
        dataset_name=None,
        constraint='in',
        mass=False,
        shuffle=True,
    ):
        """
        Create training or validation set with n elements by oversampling if needed.
        n:
        exclude: class to exclude from the dataset
        not_counting: do not oversample this class [but still limit the max number of instance to ntrain or nval !]
        it is useful for Coin class which has only one copy in each image.
        max_reuse: number of time an image can be reused
        append: image names are appended to existing basenames
        filter : a tuple with column name and value
        mass: if True, keep only instance with a defined mass
        dataset_name : tuple with name of the column where is stored the dataset name and name the dataset (col_id, name)
        constraint : use either images 'in' the dataset 'dataset_name' or 'not_in'
        Note that the oversampling is limited for some classes when they are mixed with other classes...
        """

        if constraint.lower() not in ["in", "not_in"]:
            print("constraint must be either 'in' or 'not_in'. Set it to default 'in'")
            constraint = 'in'

        if seed is not None:
            self.seed = seed

        if shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.basenames)

        if dataset_name is not None:
            text = f"{dataset_name[1]}"
            if constraint == 'in':
                filtered_annotations = self.annotations.loc[self.annotations[dataset_name[0]] == dataset_name[1]]
            else:
                filtered_annotations = self.annotations.loc[self.annotations[dataset_name[0]] != dataset_name[1]]
            filtered_basenames = filtered_annotations["baseimg"].unique().tolist()
        else:
            text = "whole dataset"
            filtered_annotations = self.annotations
            filtered_basenames = self.basenames

        # Note: the following filter deletes images where there is no mass data.
        # Beware that it keeps an image if there is at least one objet with mass data.
        if mass:
            ann = np.isfinite(filtered_annotations["mass"])
            idx = ann[~ann].index
            filtered_annotations = filtered_annotations.drop(idx)
            filtered_basenames = filtered_annotations["baseimg"].unique().tolist()

        if set == 'train':
            temp_basenames = [f for f in filtered_basenames if f not in self.valid_basenames]
        else:
            temp_basenames = [f for f in filtered_basenames if f not in self.train_basenames]

        max_inst_per_cls = {c: n for c in self.classes}

        print(f"Creating {set} set with an objective of {n} training intances {constraint} {text}")
        print(f"Using {len(temp_basenames)} images")

        basenames, class_counts = self._populate_set(
            temp_basenames,
            annotations=filtered_annotations,
            nmax=max_inst_per_cls,
            exclude=exclude,
            not_counting=not_counting,
            max_reuse=max_reuse,
            shuffle=shuffle
        )

        if append:
            print("Appending images and instances to train/valid sets")
            if set == 'train':
                self.train_basenames.extend(basenames)
                self.train_class_counts = {c: self.train_class_counts.get(c, 0) + n for c, n in class_counts.items()}
            else:
                self.valid_basenames.extend(basenames)
                self.valid_class_counts = {c: self.valid_class_counts.get(c, 0) + n for c, n in class_counts.items()}

        else:
            print("creating new train/valid sets")
            if set == 'train':
                self.train_basenames = basenames
                self.train_class_counts = class_counts
                self.train_dataset = self.build(self.train_basenames)
            else:
                self.valid_basenames = basenames
                self.valid_class_counts = class_counts
                self.valid_dataset = self.build(self.valid_basenames)

    def create_sets(
        self,
        ntrain=100,
        nval=0,
        exclude=[],
        not_counting=[],
        max_reuse=3,
        seed=None,
        append=False,
        dataset_name=None,
        mass=False,
        shuffle=True,
    ):
        """
        Create training and validation sets with ntrain and nval elements by oversampling if needed.
        ntrain:
        exclude: class to exclude from the dataset
        not_counting: do not oversample this class [but still limit the max number of instance to ntrain or nval !]
        it is useful for Coin class which has only one copy in each image.
        max_reuse: number of time an image can be reused
        append: image names are appended to existing basenames
        filter : a tuple with column name and value
        mass: if True, keep only instance with a defined mass
        Note that the oversampling is limited for some classes when they are mixed with other classes...
        """

        self.seed = seed

        if shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.basenames)

        train_frac = ntrain / (ntrain + nval)

        if dataset_name is not None:
            text = f"{dataset_name[1]}"
            filtered_annotations = self.annotations.loc[self.annotations[dataset_name[0]] == dataset_name[1]]
            filtered_basenames = filtered_annotations["baseimg"].unique().tolist()
        else:
            text = "whole dataset"
            filtered_annotations = self.annotations
            filtered_basenames = self.basenames

        # Note: the following filter deletes images where there is no mass data.
        # Beware that it keeps an image if there is at least one objet with mass data.
        if mass:
            ann = np.isfinite(filtered_annotations["mass"])
            idx = ann[~ann].index
            filtered_annotations = filtered_annotations.drop(idx)
            filtered_basenames = filtered_annotations["baseimg"].unique().tolist()

        temp_train_basenames = filtered_basenames[: int(np.around(train_frac * len(filtered_basenames)))]
        temp_valid_basenames = filtered_basenames[int(np.around(train_frac * len(filtered_basenames))) :]

        max_inst_per_cls = {c: ntrain for c in self.classes}

        print(f"Creating set with an objective of {ntrain} training intances in {text}")
        print(f"Using {len(temp_train_basenames)} images")
        train_basenames, train_class_counts = self._populate_set(
            temp_train_basenames,
            annotations=filtered_annotations,
            nmax=max_inst_per_cls,
            exclude=exclude,
            not_counting=not_counting,
            max_reuse=max_reuse,
            shuffle=shuffle
        )
        print(f"Creating set with an objective of {nval} valid intances in {text}")
        print(f"Using {len(temp_valid_basenames)} images")
        max_inst_per_cls = {c: nval for c in self.classes}
        valid_basenames, valid_class_counts = self._populate_set(
            temp_valid_basenames,
            annotations=filtered_annotations,
            nmax=max_inst_per_cls,
            exclude=exclude,
            not_counting=not_counting,
            max_reuse=max_reuse,
            shuffle=shuffle
        )

        if append:
            print("Appending images and instances to train/valid sets")
            self.train_basenames.extend(train_basenames)
            self.train_class_counts = {c: self.train_class_counts.get(c, 0) + n for c, n in train_class_counts.items()}
            self.valid_basenames.extend(valid_basenames)
            self.valid_class_counts = {c: self.valid_class_counts.get(c, 0) + n for c, n in valid_class_counts.items()}

        else:
            print("creating new train/valid sets")
            self.train_basenames = train_basenames
            self.train_class_counts = train_class_counts
            self.valid_basenames = valid_basenames
            self.valid_class_counts = valid_class_counts

        self.train_dataset = self.build(self.train_basenames)
        self.valid_dataset = self.build(self.valid_basenames)

    def create_unbalanced_sets(self, train_frac=1, exclude=[], seed=None, append=False, dataset_name=None,
        mass=False):
        """
        Create datasets containing train_frac * number of instances of the class per class (idem for valid)
        """

        self.valid_basenames = []
        self.train_basenames = []

        self.seed = seed

        if shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.basenames)

        if dataset_name is not None:
            text = f"{dataset_name[1]}"
            filtered_annotations = self.annotations.loc[self.annotations[dataset_name[0]] == dataset_name[1]]
            filtered_basenames = filtered_annotations["baseimg"].unique().tolist()
        else:
            text = "whole dataset"
            filtered_annotations = self.annotations
            filtered_basenames = self.basenames

        # Note: the following filter deletes images where there is no mass data.
        # Beware that it keeps an image if there is at least one objet with mass data.
        if mass:
            ann = np.isfinite(filtered_annotations["mass"])
            idx = ann[~ann].index
            filtered_annotations = filtered_annotations.drop(idx)
            filtered_basenames = filtered_annotations["baseimg"].unique().tolist()

        temp_train_basenames = filtered_basenames[: int(np.around(train_frac * len(filtered_basenames)))]
        temp_valid_basenames = filtered_basenames[int(np.around(train_frac * len(filtered_basenames))) :]

        max_inst_per_cls = (self.annotations["class"].value_counts() * train_frac).to_dict()

        print(f"Creating training set with using in {text}")
        print(f"Using {len(temp_train_basenames)} images")
        train_basenames, train_class_counts = self._populate_set(
            temp_train_basenames, annotations=self.annotations, nmax=max_inst_per_cls, exclude=exclude, max_reuse=0
        )

        max_inst_per_cls = (self.annotations["class"].value_counts() * (1 - train_frac)).to_dict()
        print(f"Creating training set with using in {text}")
        print(f"Using {len(temp_valid_basenames)} images")
        valid_basenames, valid_class_counts = self._populate_set(
            temp_valid_basenames, annotations=self.annotations, nmax=max_inst_per_cls, exclude=exclude, max_reuse=0
        )

        if append:
            print("Appending images and instances to train/valid sets")
            self.train_basenames.extend(train_basenames)
            self.train_class_counts = {c: self.train_class_counts.get(c, 0) + n for c, n in train_class_counts.items()}
            self.valid_basenames.extend(valid_basenames)
            self.valid_class_counts = {c: self.valid_class_counts.get(c, 0) + n for c, n in valid_class_counts.items()}

        else:
            print("creating new train/valid sets")
            self.train_basenames = train_basenames
            self.train_class_counts = train_class_counts
            self.valid_basenames = valid_basenames
            self.valid_class_counts = valid_class_counts

        self.train_dataset = self.build(self.train_basenames)
        self.valid_dataset = self.build(self.valid_basenames)

    def filter_annotations(self):
        """Remove images which do match the input constraints from the annotations dataframe (excluded classes, objects in crop zone, not enough instances, etc.)
        also verify that both image and label files exist
        """

        # Reset annotations
        self.annotations = self.base_annotations.copy(deep=True)
        self.annotations.reset_index(drop=True, inplace=True)
        baseimgnames = self.annotations["baseimg"].unique().tolist()

        self.skip = []

        # if self.shuffle:
        #     rng = np.random.default_rng(self.seed)
        #     rng.shuffle(baseimgnames)

        temp_cls_list = self.annotations["class"].unique().tolist()
        cls_inst_counter = {c: 0 for c in temp_cls_list}
        del_indexes = []

        for basename in baseimgnames:
            # Ensure that the image and labels exists !
            inst_dataframe = self.annotations.loc[self.annotations["baseimg"] == basename]
            indexes = self.annotations[self.annotations["baseimg"] == basename].index
            if len(inst_dataframe) == 0:
                print(f"Cannot find annotations for image {basename}")
                continue

            folder = inst_dataframe["folder"].to_numpy()[0]
            imgname = glob.glob(os.path.join(folder, "images", inst_dataframe["baseimg"].to_numpy()[0] + ".*"))
            labelname = glob.glob(os.path.join(folder, "labels", inst_dataframe["baseimg"].to_numpy()[0] + ".*"))
            # imgname = next((folder / Path("images")).glob(inst_dataframe["baseimg"][0] + ".*"))
            # labelname = next((folder / Path("images")).glob(inst_dataframe["baseimg"][0] + ".*"))

            if len(imgname) == 0:
                print(
                    f"Warning: image {basename} in folder {inst_dataframe['folder'].values[0] + '/images'} not found, while its annotations exists ! Skipping"
                )
                del_indexes.extend(indexes)
                self.skip.append(basename)
                continue

            if len(labelname) == 0:
                print(
                    f"Warning: image {basename} in folder {inst_dataframe['folder'].values[0] + '/labels'} not found, while its annotations exists ! Skipping"
                )
                del_indexes.extend(indexes)
                self.skip.append(basename)
                continue

            if inst_dataframe["label"].to_numpy().size < self.mininst:
                print(f"Skipping image {basename}: too few objects ({inst_dataframe['label'].to_numpy().size})")
                del_indexes.extend(indexes)
                self.skip.append(basename)
                continue

            if inst_dataframe["label"].to_numpy().size > self.maxinst:
                print(f"Skipping image {basename}: too much objects ({inst_dataframe['label'].to_numpy().size})")
                del_indexes.extend(indexes)
                self.skip.append(basename)
                continue

            if inst_dataframe["res"].to_numpy()[0] < self.minres:
                print(f"Skipping image {basename}: resolution {inst_dataframe['res'].to_numpy()[0]} < {self.minres}")
                del_indexes.extend(indexes)
                self.skip.append(basename)
                continue

            deleted = False
            for c in inst_dataframe["class"]:
                if c in self.exclude:
                    print(f"Exclude image {basename} containing class {c}")
                    del_indexes.extend(indexes)
                    self.skip.append(basename)
                    deleted = True
                    break

            if deleted:
                continue

            # Check if a bbox fall into the cropped zone. If this is the case, the image is dismissed
            im = Image.open(imgname[0])  # Just need dimensions here
            ny = im.width
            nx = im.height
            im.close()
            target_nx = min(int(np.around(ny * self.ratio)), nx)
            target_ny = int(np.around(target_nx / self.ratio))

            if self.crop_to_aspect_ratio:
                if target_nx != nx or target_ny != ny:

                    cropy = ny - target_ny
                    cropx = nx - target_nx
                    cx = cropx // 2
                    rx = cropx % 2
                    cy = cropy // 2
                    ry = cropy % 2

                    # if nx or ny is < 0, we just need to pad the image after opening it whenbuilding the dataset, so no pb
                    # if not, we must verify that there are no instances in the crop zone
                    # we crop like this: [cropx//2 + cropx%2:-cropx//2, cropy//2 + cropy%2:-cropy//2,:]

                    x0 = inst_dataframe["x0"]
                    y0 = inst_dataframe["y0"]
                    x1 = inst_dataframe["x1"]
                    y1 = inst_dataframe["y1"]

                    y0min = y0.min()
                    y0max = y0.max()
                    y1min = y1.min()
                    y1max = y1.max()

                    x0min = x0.min()
                    x0max = x0.max()
                    x1min = x1.min()
                    x1max = x1.max()

                    if cropy > 0 and not (
                        y0min > ry
                        and y0min < ny - cy - ry
                        and y0max > cy
                        and y0max < ny - cy - ry
                        and y1min > cy
                        and y1min < ny - cy - ry
                        and y1max > cy
                        and y1max < ny - cy - ry
                    ):
                        print(f"skipping image {basename} containing box outside the cropped area")
                        # del_indexes.extend(indexes)
                        del_indexes.extend(indexes)
                        self.skip.append(basename)
                        continue

                    elif cropx > 0 and not (
                        x0min > cx
                        and x0min < nx - cx - rx
                        and x0max > cx
                        and x0max < nx - cx - rx
                        and x1min > cx
                        and x1min < nx - cx - rx
                        and x1max > cx
                        and x1max < nx - cx - rx
                    ):
                        print(f"skipping image {basename} containing box outside the cropped area")
                        del_indexes.extend(indexes)
                        self.skip.append(basename)
                        continue

            # Check if the number of instances/class is not exceeded
            if self.max_inst_per_cls is not None:
                skipit = False
                inst_cls_value_count = {c: n for c, n in inst_dataframe["class"].value_counts().items()}
                temp_dict = {}

                for c, n in cls_inst_counter.items():
                    temp_dict[c] = n + inst_cls_value_count.get(c, 0)
                    if temp_dict[c] > self.max_inst_per_cls:
                        skipit = True
                        break

                if skipit:
                    print(f"Skipping image {basename}: too much instances (>{self.max_inst_per_cls})")
                    del_indexes.extend(indexes)
                    self.skip.append(basename)
                    continue

                else:
                    cls_inst_counter.update(temp_dict)

        self.annotations = self.annotations.drop(del_indexes)
        self.basenames = self.annotations["baseimg"].unique().tolist()

    def build(self, names, shuffle=True):
        """Build the tf.data.Dataset
        each element contains:
        basename, image, mask, bboxes, classes, labels, normalized masses
        """

        if len(names) == 0:
            return None

        dsnames = names
        if shuffle:
            rng = np.random.default_rng(self.seed)
            dsnames = rng.choice(names, size=len(names), replace=False, p=None, axis=0, shuffle=True)

        dataset = tf.data.Dataset.from_tensor_slices(dsnames)

        dataset = dataset.map(
            lambda x: tf.py_function(
                func=self.parse_dataset,
                inp=[x],
                Tout=[
                    tf.TensorSpec(shape=(None,), dtype=tf.string),
                    tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                    tf.RaggedTensorSpec(shape=(None, None, 4), ragged_rank=1, dtype=tf.float32),
                    tf.RaggedTensorSpec(
                        shape=(
                            None,
                            None,
                        ),
                        dtype=tf.int32,
                    ),
                    tf.RaggedTensorSpec(
                        shape=(
                            None,
                            None,
                        ),
                        dtype=tf.int32,
                    ),
                    tf.RaggedTensorSpec(
                        shape=(
                            None,
                            None,
                        ),
                        dtype=tf.float32,
                    ),
                ],
            ),
            num_parallel_calls=self.num_parallel_calls,
        )
        # Squeeze the unecessary dimension for boxes, class ids and box labels
        dataset = dataset.map(
            lambda a, b, c, d, e, f, g: (
                a,
                b,
                c,
                tf.squeeze(d, 0),
                tf.squeeze(e, 0),
                tf.squeeze(f, 0),
                tf.squeeze(g, 0),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        return dataset

    def parse_dataset(self, basename):
        """For an unkown reason, it is not possible to batch tensors with different lengths using
        tf.data.experimental.dense_to_ragged_batch
        It is necessary to return RaggedTensor for boxes, class indices and labels, but
        to do so, we must add a leading dimension
        if not, tensorflow throw an error "The rank of a RaggedTensor must be greater than 1"
        the extra dim must be deleted later...
        Warning: it is assumed that images are in jpg format and labels in png !!
        """
        # Extract the annotations of the current image
        str_name = basename.numpy().decode("utf-8")
        data = self.annotations.loc[self.annotations["baseimg"] == str_name]

        # Image is already normalized when using tf.io
        imgname = os.path.join(np.array(data["folder"])[0], "images", str_name + ".jpg")
        labelname = os.path.join(np.array(data["folder"])[0], "labels", str_name + ".png")
        image = tf.io.decode_image(tf.io.read_file(imgname), channels=3, dtype=tf.float32)
        # mask = tf.io.decode_image(tf.io.read_file(labelname), channels=1, dtype=tf.uint16)
        mask = np.array(Image.open(labelname)).astype(np.int32)[..., tf.newaxis]  # PIL automatically finds the dtype
        nx, ny = tf.shape(image)[0].numpy(), tf.shape(image)[1].numpy()

        # crop or pad images and labels and translate bboxes
        target_nx = min(int(np.around(ny * self.ratio)), nx)
        target_ny = int(np.around(target_nx / self.ratio))

        x0 = data["x0"]
        x1 = data["x1"]
        y0 = data["y0"]
        y1 = data["y1"]

        if self.crop_to_aspect_ratio:
            image, cropval = utils.crop_to_aspect_ratio((target_nx, target_ny), image)
            mask, _ = utils.crop_to_aspect_ratio((target_nx, target_ny), mask)
            x0 = x0 - cropval[0][0]
            x1 = x1 - cropval[0][0]
            y0 = y0 - cropval[1][0]
            y1 = y1 - cropval[1][0]

        else:
            image, padval = utils.pad_to_aspect_ratio((target_nx, target_ny), image)
            mask, _ = utils.pad_to_aspect_ratio((target_nx, target_ny), mask)
            x0 = x0 + padval[0][0]
            x1 = x1 + padval[0][0]
            y0 = y0 + padval[1][0]
            y1 = y1 + padval[1][0]

        new_nx, new_ny = image.shape[:2]

        # method = np.random.choice(["nearest", "bilinear", "bicubic"])
        mlist = ["nearest", "bilinear", "bicubic"]
        method = mlist[tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=self.seed)]
        # resize to input_shape
        if self.input_shape[0] != nx or self.input_shape[1] != ny:
            image = tf.image.resize(
                image, size=(self.input_shape[0], self.input_shape[1]), method=method, antialias=True
            )
            mask = tf.image.resize(
                mask,
                size=(self.input_shape[0] // self.mask_stride, self.input_shape[1] // self.mask_stride),
                method="nearest",
            )
        if self.augmentation_func is not None:
            image = self.augmentation_func(image)
        # reduction ratio between original image size and mask output, where the mass is computed
        ratio = (self.input_shape[0] // self.mask_stride) / new_nx
        # Normalized density in g.pix**2 / cm**2 [in pix of the mask !]
        norm_density = 100 * ((data["res"] * ratio) ** 2) * data["mass"]

        # translate the bboxes and normalize

        bboxes = np.stack([x0, y0, x1, y1], axis=-1)
        bboxes = utils.normalize_bboxes(bboxes, self.input_shape[0] - 1, self.input_shape[1] - 1)
        classes = [self.cls_to_idx[c] for c in data["class"]]
        classes = tf.convert_to_tensor(classes, dtype=tf.int32)
        bboxes = tf.convert_to_tensor(bboxes, dtype=tf.float32)
        labels = tf.convert_to_tensor(data["label"].to_numpy(), dtype=tf.int32)
        masses = tf.convert_to_tensor(norm_density.to_numpy(), dtype=tf.float32)

        # tf.print(basename, image.shape, mask.shape, bboxes.shape, classes.shape, labels.shape, masses.shape)

        return (
            basename,
            image,
            mask[..., 0],
            tf.RaggedTensor.from_tensor(bboxes[tf.newaxis, ...]),
            tf.RaggedTensor.from_tensor(classes[tf.newaxis, ...]),
            tf.RaggedTensor.from_tensor(labels[tf.newaxis, ...]),
            tf.RaggedTensor.from_tensor(masses[tf.newaxis, ...]),
        )
