import itertools
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from sklearn.metrics import confusion_matrix


class AccumLoss(object):
    """
    Tool class for loss accumulation and visualization.

    In some cases the loss will be very close to zero and hence not comparable
    to previous losses (for example, only one example in the batch). This will
    make the loss curve not continuous in the visualization. This class can
    handle this situation and replace the small loss with the loss from last step.

    Args:
        n: number of losses

    Notes:
        This class is purely for visualization purpose. In most cases you don't
        need this tool (for example, you are using "mean" as your reduction method).
        But it is not harmful as the class will produce the same values if no
        outliers.
    """
    def __init__(self, n):
        # type: (int) -> None
        self.n = n
        self.arr = np.zeros(n)
        self.last_step = np.zeros(n)

    def clear_arr(self):
        """clear the loss array for re-allocation"""
        self.arr.fill(0)

    def clear_last_step(self):
        """clear the loss of last step"""
        self.last_step.fill(0)

    def accum(self, loss_list):
        # type: (Iterable) -> np.array
        # accumulate the losses from `loss_list`
        self.arr += np.array(loss_list)
        return self.arr

    def update(self):
        # update the stored losses
        self.last_step = self.arr.copy()
        self.clear_arr()

    def cut(self):
        # replace the losses from last step if too small
        self.arr = np.where(self.arr < 1e-9, self.last_step, self.arr)
        return self.arr

    def average(self, n):
        # type: (int) -> np.array
        # return the loss array averaged by n
        return np.nan_to_num(self.arr / n)


def images_denorm(images, mean=None, std=None):
    """
    De-normalize image array.

    Args:
        images (np.array): the image array in the shape [nb, 3, w, h]
        mean (Union[float, Iterable]): the mean values of each channel, if float,
                                       it will repeat for all the 3 channels
        std (Union[float, Iterable]): the standard deviation of each channel, if float,
                                      it will repeat for all the 3 channels

    Returns:
        @rtype: np.array
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.485, 0.456, 0.406]

    if isinstance(mean, float):
        mean = [mean] * 3
    if isinstance(std, float):
        std = [std] * 3

    mean = np.array(mean).reshape((1, 3, 1, 1))
    std = np.array(std).reshape((1, 3, 1, 1))
    return images * std + mean


def plot_attn_maps(images, attn_maps, size=2.5):
    """
    Make pyplot figure for attention maps.

    Args:
        images (np.array): the image array in the shape [nb, w, h, 3]
        attn_maps (np.array): the attention array in the shape [nb, N, w, h]
        size (int): image size in figure

    Returns:
        @rtype: plt.figure.Figure
    """
    num_image = images.shape[0]
    num_attn = attn_maps.shape[1]
    fig, axes = plt.subplots(nrows=num_image, ncols=num_attn,
                             figsize=(size*num_attn, size*num_image),
                             squeeze=False)
    for i, j in itertools.product(range(num_image), range(num_attn)):
            image = np.clip(images[i], 0, 1)
            attn = attn_maps[i, j]
            attn = resize(attn, image.shape[:2], order=0)
            ax = axes[i, j]
            ax.imshow(image)
            ax.imshow(attn, cmap='hot', interpolation='nearest', alpha=.4)
            ax.margins(.1)
            ax.axis('off')
    plt.tight_layout()
    return fig

def plot_attn_maps_or_images(images, attn_maps, size=2.5, ifattn = False):
    """
    Make pyplot figure for attention maps.

    Args:
        images (np.array): the image array in the shape [nb, w, h, 3]
        attn_maps (np.array): the attention array in the shape [nb, N, w, h]
        size (int): image size in figure

    Returns:
        @rtype: plt.figure.Figure
    """
    num_image = images.shape[0]
    num_attn = attn_maps.shape[1]
    fig, axes = plt.subplots(nrows=num_image, ncols=num_attn,
                             figsize=(size*num_attn, size*num_image),
                             squeeze=False)
    for i, j in itertools.product(range(num_image), range(num_attn)):
            image = np.clip(images[i], 0, 1)
            attn = attn_maps[i, j]
            attn = resize(attn, image.shape[:2], order=0)
            ax = axes[i, j]
            if ifattn == False:
                ax.imshow(image)
            else:
                ax.imshow(attn, cmap='hot', interpolation='nearest', alpha=.4)
            ax.margins(.1)
            ax.axis('off')
    plt.tight_layout()
    return fig


def compute_metrics(preds, gts, num_class):
    """
    Compute the classification metrics

    Args:
        preds (np.array): the predicted class labels
        gts (np.array): the ground truth class labels
        num_class (int): number of classes

    Returns:
        acc (float): overall accuracy
        acc_class_avg (float): class average accuracy
        acc_per_class (np.array): accuracy for each class
        conf_mat (np.array): the confusion matrix
    """
    conf_mat = confusion_matrix(gts, preds, np.arange(num_class))
    conf_mat = np.nan_to_num(conf_mat)
    acc = conf_mat.diagonal().sum() / len(gts)
    acc_per_class = conf_mat.diagonal() / conf_mat.sum(axis=1)
    acc_per_class = np.nan_to_num(acc_per_class)
    acc_class_avg = acc_per_class.mean()
    return acc, acc_class_avg, acc_per_class, conf_mat


def plot_conf_mat(conf_mat, class_names, fig_size, label_font_size=11,
                  show_num=False, show_color_bar=False, xrotate=45):
    """
    Make pyplot figure for confusion matrix

    Args:
        conf_mat (np.array): the confusion matrix
        class_names (Iterable): the class names
        fig_size (Tuple[int, int]): the plot figure size
        label_font_size (int): the font size for x axis tick labels
        show_num (bool): whether to show the number on the figure
        show_color_bar (bool): whether to show the color bar
        xrotate (int): rotation degree for x axis tick labels
    Returns:
        @rtype: plt.figure.Figure
    """
    fig, ax = plt.subplots(figsize=fig_size)
    cm = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    cm = np.nan_to_num(cm)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix')
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontdict={'fontsize': label_font_size})
    ax.set_yticklabels(class_names, fontdict={'fontsize': label_font_size})
    plt.setp(ax.get_xticklabels(), rotation=xrotate)
    plt.margins(0.5)

    if show_num:
        threshold = cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = 'white' if cm[i, j] > threshold else 'black'
            ax.text(j, i, cm[i, j], horizontalalignment='right', color=color)

    ax.set_ylabel('GT')
    ax.set_xlabel('pred')
    if show_color_bar:
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def compute_feat_shape(input_size, backbone):
    """
    Compute the final conv feature map shape for different backbone and input size

    Args:
        input_size (Tuple[int, int]): input image size
        backbone (str): backbone network

    Returns:
        @rtype: Tuple[int, int]
    """
    if backbone == 'vgg19' or backbone == 'resnet12':
        return input_size[0]/16, input_size[1]//16
    else:
        return input_size[0]/32, input_size[1]//32
