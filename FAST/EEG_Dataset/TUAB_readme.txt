Data loader 

This document provides instructions for running the preprocessing pipeline on the TUAB dataset. The code is designed to execute several preprocessing steps to prepare EEG data for analysis. 

  
Preprocessing Steps 

The preprocessing pipeline performs the following operations on the EEG data: 

Downsampling: The data are downsampled to 250 Hz to standardize the sampling rate across all recordings. 

Noise Removal: A notch filter is applied to remove power line noise at 60 Hz. 

Bandpass Filtering: The data are filtered using a bandpass filter with a passband of 0.5 to 100 Hz to retain frequencies within the typical EEG range. 

Channel Renaming: The channel names are corrected to remove the 'EEG' prefix and '-REF' suffix, leaving only the channel names as identifiers. 

  
Transfer the name in 10-20 to 10-10 

'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'  


Channel Selection: A predefined list of 62 EEG channels is used. Channels not present in the dataset are added via zero-padding. 

Re-reference: Apply an average reference  

Saving the Preprocessed Data: The preprocessed data are saved into a new folder named TUAB_Abnormal_preprocessed for future use. 


Channel List 

The predefined channel list for the preprocessing includes the following EEG channels: 


desired_channels = [ 

    "AF7", "AF3", "AFz", "AF4", "AF8", 

    "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", 

    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", 

    "T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10", 

    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", 

    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", 

    "PO7", "PO3", "POz", "PO4", "PO8", 

    "O1", "O2" 

] 


Zero Padding 

Channels not included in the dataset but present in the desired_channels list will be added using zero-padding to maintain a consistent data structure across all recordings. 

Data Storage 

After preprocessing, the data are stored in a directory named TUAB_Abnormal_preprocessed. This directory structure is created to facilitate easy access and identification of the processed files for future research and analysis. 

The data can be saved for two formats: .edf and .h5, 

For h5 reading: 

 
def read_h5(raw,h5_file_path): 
h5_file_path = output_file_path[:-4] + '.h5'# replace .edf to .h5, optional
with h5py.File(h5_file_path, 'r') as h5file: 
# Read EEG data 
eeg_data = h5file['data'][:] 
 
# Read channel names 
ch_names = h5file['info/ch_names'][:] 
ch_names = [ch.decode('utf-8') for ch in ch_names] # Decode to utf-8 if necessary 
 
# Read sampling frequency 
sfreq = h5file['info/sfreq'][()] # Use [()] for single value datasets 
 
# Optionally, read channel types if you saved them 
ch_types = h5file['info/ch_types'][:] 
ch_types = [ch.decode('utf-8') for ch in ch_types] # Decode to utf-8 if necessary 
# basic = h5file['info/basic'][()] 
times = h5file['times'][()] 
return eeg_data, times, ch_names, ch_types, sfreq 

Note:  

1. The reference and ground channels are not included in the channel list but are necessary for the recording setup. 

2. when save edf, the pyedflib require the physical_max and physical_min of channel are not equal, which conflicts with zero-pading. So when writing zero-padded channel, the physical_max and physical_min will be set as 0.0001 and -0.0001.  This change was only set for zero-padded channel. If all channels are set like this, it will cause problem, making no the physical value less than –0.0001.