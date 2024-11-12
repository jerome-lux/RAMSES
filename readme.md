# Recycled Aggregates Mass Estimation and Segmentation RAMSES

Add mass estimation to SOLOv2  model (Segmenting Objects by LOcations, https://arxiv.org/pdf/2003.10152.pdf)).
Implemented using tensorflow (tf version must be <2.16, because of changes introduced in keras 3>)

## Creating the model
First, create a config object

    config = RAMSES.Config() #default config

You can also customize the config:

    params = {
    "load_backbone":False,
    "backbone_params":{},
    "backbone":'resnext50',
    "ncls":1,
    "imshape":(768, 1536, 3),
    "mask_stride":4,
    "activation:"gelu",
    "normalization":"gn",
    "normalization_kw":{'groups': 32},
    "model_name":"RAMSES-Resnext50",
    "connection_layers":{'C2': 'stage1_block3Convblock', 'C3': 'stage2_block4Convblock', 'C4': 'stage3_block6Convblock', 'C5': 'stage4_block3Convblock'},
    "FPN_filters":256,
    "extra_FPN_layers":1,
    "head_filters":256,
    "strides":[4, 8, 16, 32, 64],
    "head_layers":4,
    "kernel_size":1,
    "grid_sizes":[64, 36, 24, 16, 12],
    "scale_ranges":[[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]],
    "offset_factor":0.25,
    "mask_mid_filters":128,
    "mask_output_filters":256,
    "geom_feat_convs":2,
    "geom_feats_filters":128,
    "mask_output_level":0,
    "sigma_nms":0.5,
    "min_area":0,
    "lossweights":[1.0, 1.0, 1.0],
    "max_pos_samples":2048,
    "threshold_metrics":0.0,
    "compute_cls_loss":True,
    "compute_seg_loss":True,
    "compute_density_loss":False,
    "nms_sigma":0.5,
    }

     config = RAMSES.Config(**params)

The backbone can be loaded using load_backbone=True and backbone="path_to_your_backbone". It is a resnext50 by default.<br>
Then create the model:

    myRAMSESmodel = RAMSES.model.RAMSESModel(config)

When using a custom backbone, you have to put the name of the layers that will be connected to the FPN in the dict "connection_layers"

The model architecture can be accessed using the .model attribute

## Training with custom dataset: <br>
By default, the dataset is loaded using a custom DataLoader class<br>
The dataset files should contains two folders and an annotations.csv file:<br>
/images: RGB images <br>
/labels: labeled masks grey-level images (8, 16 or 32 bits int / uint) in **non compressed** format (png or bmp). names must be the same as in /image folder <br>
annotations.csv: contains instance data. At least image base name (without extension), labels in image, class, resolution (pixels/mm), box coordinates in [x0, y0, x1, y1], mass (g) and dataset folder on disk. Here are the required headers: <br>

    label, baseimg, area, x0, y0, x1, y1, class, res, mass, folder

Note that each corresponding image, label and annotation file must have the same base name<br>

First create a dict with class names and index <br>

    cls_ind = {
    "background":0,
    "cat":1,
    "dog":2,
    ...
    }

Then create a dataloader object using the annotations.csv opened in a pandas dataframe <br>

    dataloader = RAMSES.csvDataLoader(dataframe,
                        input_shape=config.imshape[:2],
                        cls_to_idx=cls_to_idx,
                        mininst=1,
                        maxinst=600,
                        minres=0,
                        mask_stride=config.mask_stride,
                        exclude=exclude,
                        augmentation_func=None,#partial(aug_func, transform=transform),
                        num_parallel_calls=1,
                        shuffle=True)

Note that, as the dataloader will output the mask targets, you must provide the mask strides (it can be changed later, but you must re-build the tf.datasets)

You can create train/valid tf datasets using create_sets() or create_set() methods. You can use a column values to filter the image used when generating the datasets.

hese methods first create the list images in train/valid sets using the given constraints (reusing images, maximum number of instancess per class, etc.), then they create the tf.Dataset with the build() method.

    dataloader.create_sets(ntrain=5000, nval=500, exclude=[], not_counting=[], 
                       max_reuse=2, append=False, seed=None, dataset_name=("id", "MYDATASET_1"), mass=False)

The dataset attribute train_basenames and valid_basenames contains the image names (with possible duplicates if some image are used several times). You can save and load your dataset using save() method and load() class method.

The dataloader.dataset attribute is a tf.Dataset and it will output:
- image name,
- image [H, W, 3],
- masks [H, W]: integer labeled instance image
- box [N]: box in xyxy formt for each instance
- cls_ids [N]: class id of each instance
- labels [N]: label (in the mask image) of each instance
- normalized mass (mass * mask_resolution**2), where mask_resoltuin is in pixels/mm

Note that the outputs are always _ragged tensors_

It should be easy to create a DataLoader for other formats like COCO.

To train the model, use the "train" function, with the chosen optimizer, batch size and callbacks: <br>

    RAMSES.model.train(myRAMSESmodel,
                       trainset.dataset,
                       epochs=epochs,
                       val_dataset=None,
                       steps_per_epoch=len(trainset.dataset) // batch_size,
                       validation_steps= 0,
                       batch_size=batch_size,
                       callbacks = callbacks,
                       optimizer=optimizer,
                       prefetch=tf.data.AUTOTUNE,
                       buffer=150)

## Inference
A call to the model with a [1, H, W, 3] image returns the N masks tensor (one slice per instance [1, N, H/2, W/2]) and corresponding classes [1, N] and scores [1, N] and normalized masses [1, N]. <br>
The model ALWAYS return ragged tensors, and should work with batchsize > 1.
The final labeled prediction can be obtained by the RAMSES.utils.decode_predictions function

    seg_peds, cls_labels, scores, masses = myRAMSESmodel(input)
    labeled_masks = RAMSES.utils.decode_predictions function(seg_preds, scores, threshold=0.5, by_scores=True)

Results can be vizualised using the RAMSES.visualization.draw_instances function:

    
    img = RAMSES.visualization.draw_instances(input, 
            labeled_masks.numpy(), 
            cls_ids=cls_labels[0,...].numpy() + 1, 
            cls_scores=scores[0,...].numpy(), 
            class_ids_to_name=id_to_cls, 
            show=True, 
            fontscale=0., 
            fontcolor=(0,0,0),
            alpha=0.5, 
            thickness=0)

or using matplotlib:

    cls_ids = [idx_to_cls[id + 1] for id in cls_labels|0, ...].numpy()]
    fig = ISMENet.plot_instances(
                    input,
                    labeled_masks.numpy()[0, ...],
                    cls_ids=cls_ids,
                    cls_scores=scores.numpy()[0,...],
                    alpha=0.2,
                    fontsize=2.5,
                    fontcolor="black",
                    draw_boundaries=True,
                    dpi=300,
                    show=False,
    )
    plt.show()


Note that all inputs to this function must have a batch dimension and should be converted to numpy arrays.
    


