import os
import glob
import shutil
import subprocess
import logging

import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm

import imgaug as ia
from imgaug import augmenters as iaa
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold

from trains import Task

logging.basicConfig(level=logging.INFO)
def main():
    #read yaml file
    with open('config.yaml') as file:
        config= yaml.safe_load(file)


    # trains init
    task = Task.init(task_name="Data Preprocess", auto_connect_arg_parser=False)
   
    #logging.info("Hypotenuse of {a}, {b} is {c}".format(a=3, b=4, c=hypotenuse(a,b))) 
    logging.info("Preparing Data") 
        # trains hyperparameters record
    config = task.connect(config)

    logging.info("Using config file with following parameters: {a}".format(a=config))
 
    
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Flipud(0.5), # vertical flips
        sometimes(iaa.Crop(percent=(0, 0.05))), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        #sometimes(iaa.Affine(
        #        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #        rotate=(-45, 45),
                #shear=(-16, 16),
                #order=[0, 1],
                #cval=(0, 255),
        #        mode=ia.ALL
        #    )), 
        iaa.SomeOf((0, 3),
                [
                    # Convert some images into their superpixel representation,
                # sample between 20 and 200 superpixels per image, but do
                # not replace all superpixels with their average, only
                # some of them (p_replace).
                
                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    #iaa.AverageBlur(k=(2, 7)),
                    #iaa.MedianBlur(k=(3, 11)),
                ]),

                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                # Same as sharpen, but for an embossing effect.
                #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                # Search in some images either for all edges or for
                # directed edges. These edges are then marked in a black
                # and white image and overlayed with the original image
                # using an alpha of 0 to 0.7.
                #sometimes(iaa.OneOf([
                #    iaa.EdgeDetect(alpha=(0, 0.7)),
                #    iaa.DirectedEdgeDetect(
                #        alpha=(0, 0.7), direction=(0.0, 1.0)
                #    ),
                #])),

                # Add gaussian noise to some images.
                # In 50% of these cases, the noise is randomly sampled per
                # channel and pixel.
                # In the other 50% of all cases it is sampled once per
                # pixel (i.e. brightness change).
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),

             
                # Invert each image's channel with 5% probability.
                # This sets each pixel value v to 255-v.

                # Add a value of -10 to 10 to each pixel.
                iaa.Add((-3, 3), per_channel=0.5),

                # Change brightness of images (50-150% of original value).
                iaa.Multiply((0.85, 1.15), per_channel=0.5),

                # Improve or worsen the contrast of images.
                iaa.ContrastNormalization((0.75, 1.5)),

                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                #iaa.Grayscale(alpha=(0.0, 1.0)),

                # In some images move pixels locally around (with random
                # strengths).
                #sometimes(
                #    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                #),

                # In some images distort local areas with varying strength.
                #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ],
            # do all of the above augmentations in random order
            random_order=True
        )
    ],
    random_order=True) # apply augmenters in random order

    names = []
    labels = []
    for cls in config["dataset"]["clss"]:
        files = os.listdir(config["dataset"]["base"]+cls)
        
        for f in files:
            names.append(f)
            labels.append(cls)


    cv = RepeatedStratifiedKFold(n_splits=config["dataset"]["splits"], n_repeats=config["dataset"]["repeats"],
                             random_state=config["seed"])

    split = 0
    base_dir = os.path.join(config["dataset"]["base"])
    for train_index, val_index in cv.split(names, labels):
        
        split_dir = config["processed"]+str(split)+"_split/"
        # Get train-validation split from base dir
        for dir in ['raw/', 'val/']:
            for cls in config["dataset"]["clss"]:
                if not os.path.exists(split_dir+dir+cls):
                    os.makedirs(split_dir+dir+cls)

        print("TRAIN:", train_index, "TEST:", val_index)
        for idx in train_index:
                shutil.copy(os.path.join(base_dir, labels[idx], names[idx]), 
                    os.path.join(split_dir, 'raw', labels[idx], names[idx]))
        for idx in val_index:
            shutil.copy(os.path.join(base_dir, labels[idx], names[idx]), 
                    os.path.join(split_dir, 'val', labels[idx], names[idx]))

        aug_dir = split_dir+"aug/"

        for cls in tqdm(config["dataset"]["clss"]):
            counter = 0
            if not os.path.exists(aug_dir+cls):
                os.makedirs(aug_dir+cls)

            
            while (counter < config["dataset"]["s_per_class"]):
                files = os.listdir(split_dir+"raw/"+cls)
                for f in files:
                    imo = np.asarray(Image.open(split_dir+"raw/"+cls+'/'+f))
                    im = seq.augment_image(imo)
                    im = Image.fromarray(im)
                    if im.size[0]>im.size[1]:
                        im = im.resize((512,384), Image.ANTIALIAS)
                    elif im.size[0]<im.size[1] == 4032:
                        im = im.resize((384,512), Image.ANTIALIAS)
                    else:
                        print (im.size)
                        print ("Invalid Image")
                    im.save(aug_dir+cls+'/'+str(counter)+f, quality=100)          
                    counter += 1
                    if counter >= config["dataset"]["s_per_class"]:
                        break
        
        split += 1
        
if __name__ == "__main__":
    main()