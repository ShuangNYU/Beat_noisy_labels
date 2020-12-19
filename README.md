# Beat_noisy_labels
Applied semi-supervision on the noisy-detected data via self-supervised technique: generate randomly rotation in training set and trained a ResNet model to predict both classification labels and rotation labels, and designed the loss function to focus more on self-supervision task for examples detected as noisy, while more on the original classification task for examples detected as clean. 
To run the model, please run the notebook 'run.ipynb'.

The code of dataloader is developed and modified from DivideMix (https://github.com/LiJunnan1992/DivideMix).
