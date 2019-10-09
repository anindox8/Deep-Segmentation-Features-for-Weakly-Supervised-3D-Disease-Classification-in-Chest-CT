# Prerequisites  

For full functionality, this directory requires a few additional files that can be accessed at the following links:  
  ● `niftynet_local`: [https://drive.google.com/drive/folders/1mWf3hppzSlEBgT-GyjLZHZrQdyLHwoxh?usp=sharing](https://drive.google.com/drive/folders/1mWf3hppzSlEBgT-GyjLZHZrQdyLHwoxh?usp=sharing)     
  

# Modifications  

For full functionality, off-the-shelf scripts require a few modifications as follows:  

## Multi-Resolution Segmentation Feature Maps Extraction
  ● `niftynet_local/niftynet/network/dense_vnet.py`: Line 303: `return output` -> `return all_features`  
  ● `niftynet_local/niftynet/application/segmentation_application.py`: Line 411-422: *comment-out*  
  ● `config.ini`: Line 68: `output_prob = False`  
  
## Class Probabilities Extraction
  ● `config.ini`: Line 68: `output_prob = True`  