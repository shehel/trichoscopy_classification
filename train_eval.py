import logging
import os
import random
import yaml
from time import sleep
from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path
from sklearn.metrics import confusion_matrix
from fastai.callbacks import SaveModelCallback

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from trains import Task
import pdb

logging.basicConfig(level=logging.INFO)

def evaluate_model(config, path, t_dir, v_dir, model, bs, size, epochs):

    tfms = get_transforms(do_flip=True, flip_vert=True)
    data = ImageDataBunch.from_folder(path, train=t_dir, valid=v_dir, ds_tfms = None, bs=bs, size=size)

    round_acc = []
    round_stats = []

    learn = cnn_learner(data, model,metrics=[error_rate, accuracy, Recall(), Precision(), FBeta(beta=1)])
    # First round of training with frozen conv layers
    learn.fit_one_cycle(epochs, max_lr=1e-3, callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy', name=path/'best_resnet19_run2')])
    learn.load(path/"best_resnet19_run2")
    acc = accuracy(*learn.get_preds(ds_type=DatasetType.Valid))
    b = learn.get_preds(ds_type=DatasetType.Valid)
    stat = (precision_recall_fscore_support(b[1], torch.argmax(b[0], dim=1),
                                                    labels=list(range(10)),average='micro'))
    round_acc.append(acc)
    round_stats.append(stat)

    # Second round of training with frozen conv layers
    learn.fit_one_cycle(epochs, max_lr=1e-4, callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy', name=path/'best_resnet19_run2')])
    learn.load(path/"best_resnet19_run2")
    acc = accuracy(*learn.get_preds(ds_type=DatasetType.Valid))
    b = learn.get_preds(ds_type=DatasetType.Valid)
    stat = (precision_recall_fscore_support(b[1], torch.argmax(b[0], dim=1),
                                                    labels=list(range(10)),average='micro'))
    round_acc.append(acc)
    round_stats.append(stat)

    # Third round of training with fine-tuning conv layers
    learn.unfreeze()
    learn.fit_one_cycle(epochs, max_lr=slice(1e-06, 1e-05), callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy', name=path/'best_resnet19_run2')])

    acc = accuracy(*learn.get_preds(ds_type=DatasetType.Valid))
    print (acc)
    b = learn.get_preds(ds_type=DatasetType.Valid)
    stat = (precision_recall_fscore_support(b[1], torch.argmax(b[0], dim=1),
                                                    labels=list(range(10)),average='micro'))
    round_acc.append(acc)
    round_stats.append(stat)
    flat = [round_stats[x][2] for x in range(len(round_stats))]
    max_index = flat.index(max(flat))
    print ("Round acc ", round_acc)
    print ("Round Stats ", round_stats)
    return round_acc[max_index], round_stats[max_index]

def main():
    # trains init
    task = Task.init(task_name="Train Eval", auto_connect_arg_parser=False)

    #read yaml file
    with open('config.yaml') as file:
        config= yaml.safe_load(file)

    # trains hyperparameters record
    config = task.connect(config)


    logging.info("Using config file with following parameters: {a}".format(a=config))

    stats = []
    accs = []
    #sleep(40.0)

    if config["train"]["train_dir"]:
        path = pathlib.Path(os.getcwd()) / config["train"]["data_dir"]

        acc, stat = evaluate_model(config, path, config["train"]["train_dir"], config["train"]["val_dir"],
                        models.resnet18, config["train"]["batch_size"], config["train"]["im_size"],
                        config["train"]["epochs"])
    else:
        splits = config["dataset"]["splits"]
        repeats = config["dataset"]["repeats"]
        for split in range(splits*repeats):
            split_dir = config["data"]+str(split)+"_split/"
            path = pathlib.Path(os.getcwd()) / split_dir

            acc, stat = evaluate_model(config, path, config["train"]["train_dir"], config["train"]["val_dir"],
                        models.resnet18, config["train"]["batch_size"], config["train"]["im_size"],
                        config["train"]["epochs"])
            stats.append(stat)
            accs.append(acc)

            print ("Results: ", sum(accs) / len(accs))
            print (stats)
            print (accs)


if __name__ == "__main__":
    main()
