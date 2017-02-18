# cnn-torch

Basic CNN(Convolutional Neural Networks) which is implemented by torch

## Explanation

Classify images by 10 classes.
Use CIFAR-10 classification datasets.

## Results

### train-CPU

#### learningRate 0.001 / epoch 1

1. correct: 3748(37.48%) / total elapsed time: 785.00

#### learningRate 0.001 / epoch 5

1. correct: 4878(48.78%) / total elapsed time: 805.00
1. correct: 4784(47.84%) / total elapsed time: 1040.00
1. correct: 4841(48.41%) / total elapsed time: 520.00
1. correct: 4635(46.35%) / total elapsed time: 680.00

#### learningRate 0.001 / epoch 20

1. correct: 4834(48.34%) / total elapsed time: 10695.00

#### learningRate 0.005 / epoch 5

1. correct: 4480(44.80%) / total elapsed time: 2881.00

### train-GPU

#### learningRate 0.001 / epoch 1

1. correct: 3663(36.63%) / total elapsed time: 24.00
1. correct: 3848(38.48%) / total elapsed time: 24.00
1. correct: 3755(37.55%) / total elapsed time: 25.00
1. correct: 3433(34.33%) / total elapsed time: 24.00

#### learningRate 0.001 / epoch 5

1. correct: 4736(47.36%) / total elapsed time: 52.00
1. correct: 4675(46.75%) / total elapsed time: 55.00
1. correct: 4319(43.19%) / total elapsed time: 57.00
1. correct: 4690(46.90%) / total elapsed time: 56.00

#### learningRate 0.001 / epoch 20

1. correct: 4944(49.44%) / total elapsed time: 164.00

## Reference

- [Deep Learning with Torch: the 60-minute blitz](http://nbviewer.jupyter.org/github/soumith/talks/blob/master/gtc2015/Deep%20Learning%20with%20Torch.ipynb) by Soumith