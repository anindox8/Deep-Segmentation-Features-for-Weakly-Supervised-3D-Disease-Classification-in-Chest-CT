import os
import numpy as np
from numpy.linalg import inv, det, norm
from math import sqrt, pi
from functools import partial
import operator
import SimpleITK as sitk
import pandas as pd


'''
 3D Binary Classification
 Resampling Volume Resolution

 Update: 07/08/2019
 Contributors: as1044, ft42
 
 // Target Organ (1):     
     - Lungs

 // Classes (5):             
     - Normal
     - Emphysema
     - Pneumonia-Atelectasis
     - Mass
     - Nodules

'''

def resample_img1mm(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    # Resample Images to 1mm Spacing with SimpleITK

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def resample_img2mm(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    # Resample Images to 2mm Spacing with SimpleITK

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def normalise(itk_image):
    # Normalize and Clip Image Value Range with SimpleITK

    np_img = sitk.GetArrayFromImage(itk_image)
    np_img = np.clip(np_img, -1000., 800.).astype(np.float32)
    np_img = (np_img + 1000.) / 900. - 1.
    s_itk_image = sitk.GetImageFromArray(np_img)
    s_itk_image.CopyInformation(itk_image)
    return s_itk_image


img_path     = "/DataFolder/NIFTI_00/MultipleDiseases/"
destination  = "/DataFolder/Current_1190/MultipleDiseases/"


mylist = os.listdir(img_path)

for im in range(0,len(mylist)):
    
    img_fn       = img_path + mylist[im]
    img_name     = mylist[im]

    NII         = sitk.ReadImage(img_fn,sitk.sitkFloat32)
    NII_R       = resample_img2mm(NII)
    NII_S       = normalise(NII_R)

    sitk.WriteImage(NII_S, os.path.join(destination,img_name))
    print('MD:{}'.format(img_name))
