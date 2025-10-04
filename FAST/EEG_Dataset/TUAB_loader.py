import os
import mne
import numpy as np
import pyedflib
import h5py

# Base directory for the input EDF files
base_directory = "D:\\ZSL\\BCI_Data\\TUH\\TUAB_Abnormal"
# Update this path according to your environment

# List of desired channels (Add your list of channels here)
desired_channels = ["Fp1","Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2"
]

# Function to rename EEG channels

def rename_and_update_eeg_channels(raw):
    # First, remove 'EEG ' prefix and '-REF' suffix and capitalize
    rename_dict = {}
    for ch_name in raw.ch_names:
        if ch_name.startswith('EEG ') and ch_name.endswith('-REF'):
            # Remove 'EEG ' prefix and '-REF' suffix
            new_name = ch_name[4:-4].capitalize()  # Capitalize the cleaned name
            rename_dict[ch_name] = new_name
        else:
            rename_dict[ch_name] = ch_name.capitalize()

    # Apply the renaming to clean up the names
    raw.rename_channels(rename_dict)

    # Mapping for updating specific 10-20 system channels to 10-10 system names
    channel_mapping_10_20_to_10_10 = {
        'T3': 'T7',
        'T4': 'T8',
        'T5': 'P7',
        'T6': 'P8'
    }

    # Update the channel names from 10-20 to 10-10 system
    updated_rename_dict = {}
    for ch_name in raw.ch_names:
        updated_name = channel_mapping_10_20_to_10_10.get(ch_name, ch_name)
        updated_rename_dict[ch_name] = updated_name

    # Apply the second renaming to update channel names to 10-10 system
    raw.rename_channels(updated_rename_dict)
    return raw

# Function to filter and zero-pad channels
def filter_and_zero_pad_channels(raw, desired_channels):
    current_channels = raw.ch_names
    channels_to_add = [ch for ch in desired_channels if ch not in current_channels]
    for ch in channels_to_add:
        data = np.zeros((1, len(raw.times)))
        raw.add_channels([mne.io.RawArray(data, mne.create_info([ch], raw.info['sfreq'], ch_types='eeg'))], force_update_info=True)
    channels_to_drop = [ch for ch in current_channels if ch not in desired_channels]
    raw.drop_channels(channels_to_drop)
    raw.reorder_channels(desired_channels)
    return raw

def extract_eeg_channel(raw, desired_channels):
    current_channels = raw.ch_names
    channels_to_drop = [ch for ch in current_channels if ch not in desired_channels]
    raw.drop_channels(channels_to_drop)
    return raw

def zero_pad_channels(raw, desired_channels):
    current_channels = raw.ch_names
    channels_to_add = [ch for ch in desired_channels if ch not in current_channels]
    for ch in channels_to_add:
        data = np.zeros((1, len(raw.times)))
        raw.add_channels([mne.io.RawArray(data, mne.create_info([ch], raw.info['sfreq'], ch_types='eeg'))], force_update_info=True)
    raw.reorder_channels(desired_channels)
    return raw


# Function to preprocess the data
def preprocess_eeg_data(raw):
    if raw.info['sfreq'] > 250:
        raw.resample(250, npad="auto")
    raw.notch_filter(freqs=60, notch_widths=2)
    raw.filter(0.5, 100, fir_design='firwin')
    raw.set_eeg_reference(ref_channels='average', projection=False)
    # data = raw.get_data()
    # mean = data.mean(axis=1, keepdims=True)
    # std = data.std(axis=1, keepdims=True)
    # z_transformed_data = (data - mean) / std
    # raw._data = z_transformed_data
    return raw


# Function to dynamically create the output directory
def create_preprocessed_directory(file_path):
    # Replace the base part of the file path
    preprocessed_path = file_path.replace(base_directory, base_directory + '_preprocessed')
    directory = os.path.dirname(preprocessed_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return preprocessed_path


# Function to save as EDF using pyEDFlib
# Function to save as EDF
def save_as_edf(raw, output_file_path):
    with pyedflib.EdfWriter(output_file_path, n_channels=len(raw.ch_names), file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
        channel_info = []
        data_list = []
        for i, ch_name in enumerate(raw.ch_names):
            channel_data = raw._data[i]
            phys_min = round(channel_data.min(),4)
            phys_max = round(channel_data.max(),4)
            if phys_min == phys_max:
                phys_min -= 0.0001  # Arbitrary small value to set a non-zero range
                phys_max += 0.0001  # Arbitrary small value to set a non-zero range
            ch_type = 'EEG' if 'eeg' in raw.get_channel_types(picks=[i]) else 'Other'
            channel_info.append({
                'label': ch_name, 'dimension': 'uV', 'sample_rate': int(raw.info['sfreq']),
                'physical_min': phys_min, 'physical_max': phys_max,
                'digital_min': -32768, 'digital_max': 32767, 'transducer': '', 'prefilter': ''
            })

            data_list.append(channel_data)
        writer.setSignalHeaders(channel_info)
        writer.writeSamples(data_list)
        print(f"Processed and saved: {output_file_path}")


def save_as_h5(raw,output_file_path):
    output_file_path = output_file_path[:-4] + '.h5'# replace .edf to .h5
    with h5py.File(output_file_path, 'w') as h5file:
        # Save EEG data
        # This saves the data as a dataset named 'EEG', adjust the name as necessary
        h5file.create_dataset('data', data=raw.get_data())

        # Save channel names
        # Channel names are saved as an array of strings
        dt = h5py.special_dtype(vlen=str)  # Special dtype for variable-length strings
        h5file.create_dataset('info/ch_names', data=np.array(raw.ch_names, dtype=object), dtype=dt)

        # Save sample rate
        h5file.create_dataset('info/sfreq', data=raw.info['sfreq'])

        # Optionally, save additional information
        # For example, save the channel types
        h5file.create_dataset('info/ch_types', data=np.array(raw.get_channel_types(), dtype=object), dtype=dt)
        h5file.create_dataset('times', data=raw.times)
        # h5file.create_dataset('info/basic', data=raw.info())


        print(f"Processed and saved: {output_file_path}")


def read_h5(raw,h5_file_path):
    h5_file_path = output_file_path[:-4] + '.h5'# replace .edf to .h5
    with h5py.File(h5_file_path, 'r') as h5file:
        # Read EEG data
        eeg_data = h5file['data'][:]

        # Read channel names
        ch_names = h5file['info/ch_names'][:]
        ch_names = [ch.decode('utf-8') for ch in ch_names]  # Decode to utf-8 if necessary

        # Read sampling frequency
        sfreq = h5file['info/sfreq'][()]  # Use [()] for single value datasets

        # Optionally, read channel types if you saved them
        ch_types = h5file['info/ch_types'][:]
        ch_types = [ch.decode('utf-8') for ch in ch_types]  # Decode to utf-8 if necessary
        # basic = h5file['info/basic'][()]
        times = h5file['times'][()]
        return eeg_data, times, ch_names, ch_types, sfreq
# Main processing loop
for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file.endswith('.edf'):
            file_path = os.path.join(root, file)
            try:
                # Load the .edf file
                raw_data = mne.io.read_raw_edf(file_path, preload=True)
                raw_data = rename_and_update_eeg_channels(raw_data)
                raw_data = extract_eeg_channel(raw_data,desired_channels)
                raw_data = preprocess_eeg_data(raw_data)
                raw_data = zero_pad_channels(raw_data,desired_channels)
                # Create the output path
                output_file_path = create_preprocessed_directory(file_path)
                # Save the preprocessed data
                # save_as_edf(raw_data, output_file_path)
                save_as_h5(raw_data, output_file_path)
                [eeg_data, times, ch_names, ch_types, sfreq]=read_h5(raw_data, output_file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


