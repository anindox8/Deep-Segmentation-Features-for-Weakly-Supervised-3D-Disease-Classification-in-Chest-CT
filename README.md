# Deep Segmentation Features for Weakly Supervised Disease Classification via 3D Dual-Stage CNN 

**Problem Statement**: Weakly supervised 3D classification of multi-organ, multi-disease CT scans. 

**Data** (*proprietary to Duke University Medical Center*): *Class A*: Diseased Lungs Cases (Pneumonia-Atelectasis, Emphysema, Nodules, Mass); *Class B:* Healthy Lungs Cases. 

**Note**: The following repository currently serves as an archive. It is functional, but requires a full cleanup.

**Directories**  
  ● Convert DICOM to NIfTI Volumes: `preprocess/prime/DICOM_NIFTI.py`  
  ● Resample NIfTI Volume Resolutions: `preprocess/prime/resampleRes.py`  
  ● Infer StFA/DyFA Segmentation Sub-Model (DenseVNet): `python net_segment.py inference -c '../config.ini'`  
  ● Preprocess Full Dataset to Optimized I/O HDF5 Training Patch-Volumes: `preprocess/prime/preprocess_alpha.py`  
  ● Preprocess Full Dataset to Optimized I/O HDF5 Deployment Whole-Volumes: `preprocess/prime/preprocess_deploy.py`  
  ● Generate Data-Directory Feeder List: `feed/prime/feed_metadata.py`  
  ● Train StFA Classification Sub-Model: `train/prime/train_StFA.py`  
  ● Train DyFA Classification Sub-Model: `train/prime/train_DyFA.py`  
  ● Deploy Model (Validation): `deploy/prime/deployBinary.py`  
  ● Average Predictions for Same Patient: `deploy/prime/average_predictions.py`  
  ● Calculate AUC: `notebooks/binary_AUC.ipynb`
  


**Related Publication(s):**  
  ● A. Saha, F.I. Tushar, K. Faryna, V.D. Anniballe, R. Hou, M.A. Mazurowski, G.D. Rubin, J.Y. Lo (2020), "Weakly Supervised 3D   
    Classification of Chest CT using Aggregated Multi-Resolution Deep Segmentation Features", 2020 SPIE Medical Imaging: Computer-Aided 
    Diagnosis, Houston, TX, USA. DOI:10.1117/12.2550857
                 


## Network Architecture  
  
  
![Network Architecture](reports/images/network_architecture.png)*Figure 1.  Integrated model architecture for reusing segmentation feature maps in 3D binary classification. The segmentation sub-model is a DenseVNet, taking a variable input volume with a single channel and the classification sub-model is a 3D ResNet, taking an input volume patch of size [112,112,112] with 2 channels. Final output is a tensor with the predicted class probabilities.*  
  
    
    
## Multi-Resolution Deep Segmentation Features  
  
  
![Multi-Resolution Deep Segmentation Features](reports/images/segmentation_features.png)*Figure 2.  From left-to-right: input CT volume (axial view), 3 out of 61 segmentation feature maps extracted from the pretrained DenseVNet model, at different resolutions, and their corresponding static aggregated feature maps (StFA) in the case of diseased lungs with atelectasis (top row), mass (middle row) and emphysema (bottom row).*  
  
    
    
## Experimental Results  
  
  
![Binary AUC](reports/images/auc.png)*Figure 3.  ROC curves for each disease class against all normal cases and all disease classes against all normal cases for the independent (left),  StFA (center) and DyFA (right) models for binary lung disease classification.*
