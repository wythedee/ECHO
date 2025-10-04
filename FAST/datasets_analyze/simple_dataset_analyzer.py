#!/usr/bin/env python3
"""
Simplified dataset analyzer - directly analyzes h5 files to get the first sample length information
"""

import os
import h5py
import numpy as np

# Dataset root directory
DATASET_ROOT = '/path/to/your/dataset_root'

# Dataset names built from actual file list (without .h5 extension)
# Dataset names built from actual file list (without .h5 extension)
DATASET_NAMES = [
    # 'ADHD_01_AliMotie',
    # 'CS_03_Large_Spanish_EEG_perception',
    # 'CS_03_Large_Spanish_EEG_production',
    # 'CS_04_BCIC_Track3',
    # 'EMO_01_DEAP',
    'EMO_02_SEED_IV',
    # 'EMO_03_SEED_V',
    # 'EMO_04_SEED',
    # 'EMO_05_THU-EP',
    # 'EMO_06_FACED',
    # 'MI_01_KoreaU',
    # 'MI_02_ShanghaiU',
    # 'MI_03_Shin2017A',
    # 'MI_04_BCI_IV_2a',
    # 'MI_05_Weibo2014',
    # 'MI_06_Schirrmeister2017',
    # 'MI_07_Cho2017',
    # 'MI_08_Track1_Few_shot',
    # 'MI_09_Track4_Upper_limb',
    # 'MI_10_HeBin2021_LR',
    # 'MI_10_HeBin2021_UD',
    # 'MI_11_HeBin2024',
    # 'MI_11_HeBin2024_LR',
    # 'MI_11_HeBin2024_UD',
    # 'MI_12_PhysioNet',
    # 'MI_BCI_IV_2a',
    # 'SSVEP_01_KoreaU',
    # 'SSVEP_02_TSinghuaU_Benchmark',
    # 'SSVEP_03_TSinghuaU_eldBETA',
    # 'SSVEP_04_TSinghuaU_BETA',
    # 'SLEEP_05_isruc_S1',
    # 'SLEEP_05_isruc_S3',
    # 'EP_01_CHBMIT',
    # 'MDD_01_Mumtaz',
    # 'TUH_04_TUEV_Events',
    # 'STR_01_MentalArithmetic',
]


def analyze_single_dataset(dataset_name):
    """Analyze the first sample information of a single dataset"""
    # First, try to find the file in the subdirectory
    sub_dir = dataset_name.split('_')[0]
    h5_path_subdir = f"{DATASET_ROOT}/{sub_dir}/{dataset_name}.h5"
    h5_path_root = f"{DATASET_ROOT}/{dataset_name}.h5"

    # Check if the file in the subdirectory exists
    if os.path.exists(h5_path_subdir):
        h5_path = h5_path_subdir
        print(f"    📁 Using subdirectory path: {h5_path}")
    else:
        h5_path = h5_path_root
        print(f"    📁 Using root directory path: {h5_path}")

    try:
        if not os.path.exists(h5_path):
            return None, f"File does not exist: {h5_path}"

        with h5py.File(h5_path, 'r') as f:
            # Get all subject keys
            sub_keys = list(f.keys())
            if not sub_keys:
                return None, "No subject data found"

            # Use the first subject
            first_sub = sub_keys[0]

            if 'X' not in f[first_sub] or 'Y' not in f[first_sub]:
                return None, f"Subject {first_sub} is missing X or Y data"

            # Get data shape
            X_shape = f[first_sub]['X'].shape
            Y_shape = f[first_sub]['Y'].shape

            n_trials = X_shape[0]
            n_channels = X_shape[1]
            n_timepoints = X_shape[2]

            # Get the actual shape of the first sample
            first_sample = f[first_sub]['X'][0]
            actual_channels = first_sample.shape[0]
            actual_timepoints = first_sample.shape[1]

            # Calculate time length (sampling rate 250Hz)
            time_length_seconds = actual_timepoints / 250.0

            # Get some statistics for the first sample
            sample_min = np.min(first_sample)
            sample_max = np.max(first_sample)
            sample_mean = np.mean(first_sample)
            sample_std = np.std(first_sample)

            result = {
                'Dataset Name': dataset_name,
                'Number of Subjects': len(sub_keys),
                'First Subject': first_sub,
                'Total Trials': n_trials,
                'Number of Channels': actual_channels,
                'Number of Timepoints': actual_timepoints,
                'Time Length (seconds)': round(time_length_seconds, 2),
                'Sampling Rate': 250,
                '1st Sample Shape': f"({actual_channels}, {actual_timepoints})",
                'Data Range': f"[{sample_min:.3f}, {sample_max:.3f}]",
                'Mean': round(sample_mean, 4),
                'Standard Deviation': round(sample_std, 4),
                'Number of Labels': len(np.unique(f[first_sub]['Y'][()])),
                'File Size (MB)': round(os.path.getsize(h5_path) / (1024*1024), 1)
            }

            return result, None

    except Exception as e:
        return None, f"Error reading file: {str(e)}"

def _resolve_h5_path(dataset_name):
    sub_dir = dataset_name.split('_')[0]
    h5_path_subdir = f"{DATASET_ROOT}/{sub_dir}/{dataset_name}.h5"
    h5_path_root = f"{DATASET_ROOT}/{dataset_name}.h5"
    return h5_path_subdir if os.path.exists(h5_path_subdir) else h5_path_root

def print_subject_label_distribution(dataset_name):
    """Prints the label distribution table for each subject's trials"""
    h5_path = _resolve_h5_path(dataset_name)
    if not os.path.exists(h5_path):
        print(f"  ⚠️ File does not exist: {h5_path}")
        return
    try:
        with h5py.File(h5_path, 'r') as f:
            sub_keys = list(f.keys())
            if not sub_keys:
                print("  ⚠️ No subject data found")
                return
            # Collect global label set
            label_set = set()
            per_sub_counts = []
            for sub in sub_keys:
                if 'Y' not in f[sub]:
                    continue
                y = f[sub]['Y'][()]
                uniq, cnt = np.unique(y, return_counts=True)
                counts = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
                label_set.update(counts.keys())
                per_sub_counts.append((sub, len(y), counts))

            if not per_sub_counts:
                print("  ⚠️ No available label data")
                return

            labels_sorted = sorted(label_set)
            # Print table header
            header = ["Subject", "Trials"] + [f"Label{lb}" for lb in labels_sorted]
            widths = [max(8, 20), 8] + [8] * len(labels_sorted)
            row_fmt = f"{{:<{widths[0]}}} {{:<{widths[1]}}} " + " ".join([f"{{:<{w}}}" for w in widths[2:]])
            print("\n" + "-" * (sum(widths) + len(widths) - 1))
            print(row_fmt.format(*header))
            print("-" * (sum(widths) + len(widths) - 1))

            # Print each subject
            total_trials = 0
            total_per_label = {lb: 0 for lb in labels_sorted}
            for sub, ntr, counts in per_sub_counts:
                total_trials += ntr
                row = [sub, str(ntr)]
                for lb in labels_sorted:
                    c = counts.get(lb, 0)
                    total_per_label[lb] += c
                    row.append(str(c))
                print(row_fmt.format(*row))

            # Print totals
            total_row = ["Total", str(total_trials)] + [str(total_per_label[lb]) for lb in labels_sorted]
            print("-" * (sum(widths) + len(widths) - 1))
            print(row_fmt.format(*total_row))
            print("-" * (sum(widths) + len(widths) - 1) + "\n")
    except Exception as e:
        print(f"  ⚠️ Error reading or calculating statistics: {str(e)}")

def analyze_all_datasets():
    """Analyze all datasets"""
    print("Starting analysis of the first sample length information for all datasets...")
    print("Sampling rate uniformly assumed to be 250Hz")
    print(f"Dataset root directory: {DATASET_ROOT}")
    print(f"Total {len(DATASET_NAMES)} datasets to analyze")
    print("\n" + "="*80)

    results = []
    errors = []

    for i, dataset_name in enumerate(DATASET_NAMES, 1):
        print(f"\n[{i:2d}/{len(DATASET_NAMES)}] 正在分析: {dataset_name}")

        result, error = analyze_single_dataset(dataset_name)

        if result:
            results.append(result)
            print(f"  ✅ 成功 - 受试者:{result['受试者数']}, 通道:{result['通道数']}, 时长:{result['时间长度(秒)']}秒")
        else:
            errors.append((dataset_name, error))
            print(f"  ❌ 失败 - {error}")

    return results, errors

def create_summary_table(results, errors):
    """Create summary table"""
    print("\n" + "="*120)
    print("Dataset Length Information Summary Table (Sampling Rate: 250Hz)")
    print("="*120)

    if not results:
        print("No datasets successfully analyzed")
        return

    # Table header
    print(f"{'Dataset Name':<25} {'Subjects':<8} {'Channels':<6} {'Timepoints':<8} {'Duration(s)':<10} {'Labels':<8} {'File(MB)':<10}")
    print("-" * 120)

    # Data rows
    for result in results:
        print(f"{result['数据集名称']:<25} {result['受试者数']:<8} {result['通道数']:<6} "
              f"{result['时间点数']:<8} {result['时间长度(秒)']:<10} {result['标签数']:<8} {result['文件大小(MB)']:<10}")

    # Statistics
    if results:
        sample_times = [r['Time Length (seconds)'] for r in results]
        channels = [r['Number of Channels'] for r in results]
        subjects = [r['Number of Subjects'] for r in results]
        file_sizes = [r['File Size (MB)'] for r in results]

        print("\n" + "="*80)
        print("Statistics:")
        print(f"- Successfully analyzed datasets: {len(results)} / {len(DATASET_NAMES)}")
        print(f"- Time length range: {min(sample_times):.2f} - {max(sample_times):.2f} seconds")
        print(f"- Channel count range: {min(channels)} - {max(channels)} ")
        print(f"- Total subjects: {sum(subjects)} ")
        print(f"- Total file size: {sum(file_sizes):.1f} MB")

        # Group by time length
        print(f"\nGrouped by time length:")
        time_groups = {}
        for result in results:
            time_len = result['Time Length (seconds)']
            if time_len not in time_groups:
                time_groups[time_len] = []
            time_groups[time_len].append(result['Dataset Name'])

        for time_len in sorted(time_groups.keys()):
            datasets = time_groups[time_len]
            print(f"  {time_len}s ({len(datasets)}): {', '.join(datasets)}")

    # Error messages
    if errors:
        print(f"\nFailed datasets ({len(errors)}):")
        for dataset_name, error in errors:
            print(f"  ❌ {dataset_name}: {error}")

    # Label distribution table for each subject (only output for single dataset analysis)
    if results and len(DATASET_NAMES) == 1:
        print("\n" + "="*120)
        print("Label distribution of trials for each subject")
        print("="*120)
        ds = results[0]['Dataset Name']
        print(f"\n▶ Dataset: {ds}")
        print_subject_label_distribution(ds)

    # Save to CSV
    if results:
        csv_filename = 'dataset_sample_lengths.csv'
        with open(csv_filename, 'w', encoding='utf-8-sig') as f:
            # Write header
            headers = list(results[0].keys())
            f.write(','.join(headers) + '\n')

            # Write data
            for result in results:
                csv_row = []
                for header in headers:
                    value = str(result[header])
                    # Handle values containing commas, quotes, or newlines
                    if any(char in value for char in [',', '"', '\n', '\r']):
                        value = '"' + value.replace('"', '""') + '"'
                    csv_row.append(value)
                f.write(','.join(csv_row) + '\n')

        print(f"\nDetailed results saved to: {csv_filename}")
        print(f"Encoding: UTF-8 with BOM (supports Chinese display in Excel)")

if __name__ == '__main__':
    results, errors = analyze_all_datasets()
    create_summary_table(results, errors)
    print("\nAnalysis complete!")
