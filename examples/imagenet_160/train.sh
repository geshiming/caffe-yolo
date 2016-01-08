#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/imagenet_160/solver.prototxt --snapshot=/data/ImageNet/vgg_160_train_val_iter_25000.solverstate --gpu=0
