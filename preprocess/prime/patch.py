from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


'''
 Binary Classification (3D)
 Train: Patch Extraction

 Update: 15/07/2019
 Contributors: ft42, as1044
 
 // Target Organ (1):     
     - Lungs

 // Classes (5):             
     - Normal
     - Edema
     - Atelectasis
     - Pneumonia
     - Nodules
'''



def extract_class_balanced_example_array(image,
                                         label,
                                         example_size=[1, 64, 64],
                                         n_examples=1,
                                         classes=2,
                                         class_weights=None):
    """Extract training examples from an image (and corresponding label) subject
        to class balancing. Returns an image example array and the
        corresponding label array.

    Args:
        image (np.ndarray): image to extract class-balanced patches from
        label (np.ndarray): labels to use for balancing the classes
        example_size (list or tuple): shape of the patches to extract
        n_examples (int): number of patches to extract in total
        classes (int or list or tuple): number of classes or list of classes
            to extract

    Returns:
        np.ndarray, np.ndarray: class-balanced patches extracted from full
            images with the shape [batch, example_size..., image_channels]

    Reference: DLTK
    """
    assert image.shape[:-1] == label.shape, 'Image and label shape must match'
    assert image.ndim - 1 == len(example_size), \
        'Example size doesnt fit image size'

    rank = len(example_size)

    if isinstance(classes, int):
        classes = tuple(range(classes))
    n_classes = len(classes)


    if class_weights is None:
        n_ex_per_class = np.ones(n_classes).astype(int) * int(np.round(n_examples / n_classes))
    else:
        assert len(class_weights) == n_classes, \
            'Class_weights must match number of classes'
        class_weights = np.array(class_weights)
        n_ex_per_class = np.round((class_weights / class_weights.sum()) * n_examples).astype(int)

    # Compute an example radius to define the region to extract around center location
    ex_rad = np.array(list(zip(np.floor(np.array(example_size) / 2.0),
                               np.ceil(np.array(example_size) / 2.0))),
                      dtype=np.int)

    class_ex_images = []
    class_ex_lbls = []
    min_ratio = 1.
    for c_idx, c in enumerate(classes):
        # Get valid, random center locations belonging to that class
        idx = np.argwhere(label == c)

        ex_images = []
        ex_lbls = []

        if len(idx) == 0 or n_ex_per_class[c_idx] == 0:
            class_ex_images.append([])
            class_ex_lbls.append([])
            continue

        # Extract random locations
        r_idx_idx = np.random.choice(len(idx),
                                     size=min(n_ex_per_class[c_idx], len(idx)),
                                     replace=False).astype(int)
        r_idx = idx[r_idx_idx]

        # Shift the random to valid locations if necessary
        r_idx = np.array(
            [np.array([max(min(r[dim], image.shape[dim] - ex_rad[dim][1]),
                           ex_rad[dim][0]) for dim in range(rank)])
             for r in r_idx])

        for i in range(len(r_idx)):
            # Extract class-balanced examples from the original image
            slicer = [slice(r_idx[i][dim] - ex_rad[dim][0], r_idx[i][dim] + ex_rad[dim][1]) for dim in range(rank)]

            ex_image = image[slicer][np.newaxis, :]

            ex_lbl = label[slicer][np.newaxis, :]

            # Concatenate them and return the examples
            ex_images = np.concatenate((ex_images, ex_image), axis=0) \
                if (len(ex_images) != 0) else ex_image
            ex_lbls = np.concatenate((ex_lbls, ex_lbl), axis=0) \
                if (len(ex_lbls) != 0) else ex_lbl

        class_ex_images.append(ex_images)
        class_ex_lbls.append(ex_lbls)

        ratio = n_ex_per_class[c_idx] / len(ex_images)
        min_ratio = ratio if ratio < min_ratio else min_ratio

    indices = np.floor(n_ex_per_class * min_ratio).astype(int)

    ex_images = np.concatenate([cimage[:idxs] for cimage, idxs in zip(class_ex_images, indices)
                                if len(cimage) > 0], axis=0)
    ex_lbls = np.concatenate([clbl[:idxs] for clbl, idxs in zip(class_ex_lbls, indices)
                              if len(clbl) > 0], axis=0)

    return ex_images, ex_lbls
