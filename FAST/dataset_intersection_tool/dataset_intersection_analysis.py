import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from collections import defaultdict
import seaborn as sns

def load_fast_data_with_fallback(base_path, dataset_names=None):
    """从npz或csv文件加载FAST数据，优先npz，回退到csv"""
    npz_path = os.path.join(base_path, 'pre.npz')
    
    if os.path.exists(npz_path):
        print(f"找到npz文件: {npz_path}")
        data = np.load(npz_path)['epoacc']
        print(f"从npz读取数据，shape: {data.shape}")
        return data
    else:
        print(f"未找到npz文件，尝试从csv文件读取...")
        if dataset_names is None:
            print("Warning: 需要提供dataset_names才能从csv读取数据")
            return None
        
        all_data = []
        for fold_idx in range(9):
            fold_data = []
            for dataset in dataset_names:
                csv_pattern = os.path.join(base_path, f'pre-dataset-{dataset}-fold-{fold_idx:02d}.csv')
                csv_files = glob.glob(csv_pattern)
                
                if csv_files:
                    csv_file = csv_files[0]
                    with open(csv_file, 'r') as f:
                        values = [float(line.strip()) for line in f if line.strip()]
                    fold_data.append(values)
                else:
                    print(f"Warning: 未找到数据集 {dataset} fold {fold_idx} 的csv文件")
                    fold_data.append([0.0] * 90)
            
            all_data.extend(fold_data)
        
        data = np.array(all_data)
        print(f"从csv读取数据完成，shape: {data.shape}")
        return data

def extract_dataset_intersection(model_configs):
    """提取不同模型配置中数据集的交集
    
    Args:
        model_configs (dict): 模型配置字典，格式为 {
            'config_name': {
                'path': str,
                'datasets': list,  # 该配置使用的数据集列表
                'fold_info': int/list/-1,
                'n_fold': int
            }
        }
    
    Returns:
        dict: 包含交集信息的字典
    """
    # 收集每个配置的数据集信息
    config_datasets = {}
    config_performance = {}
    
    for config_name, config_info in model_configs.items():
        config_path = config_info['path']
        dataset_list = config_info['datasets']
        fold_info = config_info.get('fold_info', 0)
        n_fold = config_info.get('n_fold', 9)
        
        print(f"\n处理配置: {config_name}")
        print(f"数据集: {dataset_list}")
        
        # 加载数据
        data = load_fast_data_with_fallback(config_path, dataset_list)
        if data is None:
            print(f"Error: 无法加载配置 {config_name} 的数据")
            continue
        
        # 计算实际的数据集数量
        dataset_length = len(dataset_list)
        actual_n_folds = data.shape[0] // dataset_length
        
        # 检测有效的fold
        valid_folds = []
        for fold_idx in range(actual_n_folds):
            start_idx = fold_idx * dataset_length
            end_idx = (fold_idx + 1) * dataset_length
            if start_idx < data.shape[0] and end_idx <= data.shape[0]:
                fold_data = data[start_idx:end_idx]
                if np.any(fold_data != 0):
                    valid_folds.append(fold_idx)
        
        # 确定要处理的fold
        if fold_info == -1:
            fold_list = valid_folds
        elif isinstance(fold_info, list):
            fold_list = [f for f in fold_info if f in valid_folds]
        else:
            if fold_info in valid_folds:
                fold_list = [fold_info]
            else:
                print(f"Warning: fold {fold_info} 不存在于配置 {config_name} 中")
                continue
        
        if not fold_list:
            print(f"Error: 配置 {config_name} 没有有效的fold数据")
            continue
        
        # 计算每个数据集在最佳epoch的性能
        dataset_performance = {}
        for dataset_idx, dataset_name in enumerate(dataset_list):
            dataset_accs = []
            for fold_idx in fold_list:
                fold_data = data[fold_idx*dataset_length:(fold_idx+1)*dataset_length]
                if dataset_idx < fold_data.shape[0]:
                    # 计算该数据集在所有epoch的平均准确率
                    mean_acc = np.mean(fold_data, axis=0)
                    best_epoch = np.argmax(mean_acc)
                    best_acc = fold_data[dataset_idx, best_epoch]
                    dataset_accs.append(best_acc)
            
            if dataset_accs:
                dataset_performance[dataset_name] = {
                    'mean_acc': np.mean(dataset_accs),
                    'std_acc': np.std(dataset_accs),
                    'best_acc': np.max(dataset_accs),
                    'fold_count': len(dataset_accs)
                }
        
        config_datasets[config_name] = list(dataset_performance.keys())
        config_performance[config_name] = dataset_performance
    
    # 计算数据集交集
    all_datasets = set()
    for datasets in config_datasets.values():
        all_datasets.update(datasets)
    
    # 找出所有配置都包含的数据集（交集）
    intersection_datasets = set(config_datasets[list(config_datasets.keys())[0]])
    for datasets in config_datasets.values():
        intersection_datasets = intersection_datasets.intersection(set(datasets))
    
    # 找出只在部分配置中存在的数据集（差集）
    union_datasets = set()
    for datasets in config_datasets.values():
        union_datasets.update(datasets)
    
    only_some_configs = union_datasets - intersection_datasets
    
    return {
        'intersection_datasets': sorted(list(intersection_datasets)),
        'union_datasets': sorted(list(union_datasets)),
        'only_some_configs': sorted(list(only_some_configs)),
        'config_datasets': config_datasets,
        'config_performance': config_performance,
        'all_datasets': sorted(list(all_datasets))
    }

def plot_intersection_comparison(intersection_info, save_path="images"):
    """绘制数据集交集的性能对比图
    
    Args:
        intersection_info (dict): 交集信息字典
        save_path (str): 图片保存路径
    """
    intersection_datasets = intersection_info['intersection_datasets']
    config_performance = intersection_info['config_performance']
    
    if not intersection_datasets:
        print("Warning: 没有找到数据集交集")
        return
    
    # 创建交集数据集的性能对比图
    n_datasets = len(intersection_datasets)
    n_configs = len(config_performance)
    
    if n_configs == 0:
        print("Error: 没有有效的配置数据")
        return
    
    # 设置图形
    fig, axes = plt.subplots(2, 1, figsize=(max(15, n_datasets * 2), 12), dpi=150)
    
    # 第一个子图：柱状图对比
    ax1 = axes[0]
    
    bar_width = 0.8 / n_configs
    x = np.arange(n_datasets)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_configs))
    
    for i, (config_name, performance) in enumerate(config_performance.items()):
        accs = [performance[dataset]['mean_acc'] for dataset in intersection_datasets]
        offset = (i - n_configs/2 + 0.5) * bar_width
        
        bars = ax1.bar(x + offset, accs, bar_width, 
                      label=config_name, 
                      color=colors[i], alpha=0.8)
        
        # 添加数值标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax1.set_title('Dataset Intersection Performance Comparison', pad=20, fontsize=14)
    ax1.set_xlabel('Datasets', fontsize=12)
    ax1.set_ylabel('Mean Accuracy', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(intersection_datasets, rotation=45, ha='right')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # 计算整体平均值线
    all_accs = []
    for performance in config_performance.values():
        accs = [performance[dataset]['mean_acc'] for dataset in intersection_datasets]
        all_accs.extend(accs)
    
    overall_mean = np.mean(all_accs)
    ax1.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax1.text(n_datasets-1, overall_mean + 0.01, f'Overall Mean: {overall_mean:.3f}', 
             ha='right', va='bottom', color='red', fontweight='bold')
    
    # 第二个子图：热力图
    ax2 = axes[1]
    
    # 准备热力图数据
    heatmap_data = []
    for config_name, performance in config_performance.items():
        row = [performance[dataset]['mean_acc'] for dataset in intersection_datasets]
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    # 创建热力图
    im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    # 设置标签
    ax2.set_xticks(range(len(intersection_datasets)))
    ax2.set_xticklabels(intersection_datasets, rotation=45, ha='right')
    ax2.set_yticks(range(len(config_performance)))
    ax2.set_yticklabels(list(config_performance.keys()))
    
    # 添加数值标签
    for i in range(len(config_performance)):
        for j in range(len(intersection_datasets)):
            text = ax2.text(j, i, f'{heatmap_data[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    ax2.set_title('Performance Heatmap', pad=20, fontsize=14)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Accuracy', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(save_path, exist_ok=True)
    config_names = '_'.join(list(config_performance.keys())[:3])  # 限制文件名长度
    plt.savefig(f"{save_path}/intersection_comparison_{config_names}.png", 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    return fig

def plot_dataset_coverage(intersection_info, save_path="images"):
    """绘制数据集覆盖情况图
    
    Args:
        intersection_info (dict): 交集信息字典
        save_path (str): 图片保存路径
    """
    config_datasets = intersection_info['config_datasets']
    intersection_datasets = intersection_info['intersection_datasets']
    only_some_configs = intersection_info['only_some_configs']
    
    # 创建数据集覆盖情况图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    
    # 第一个子图：数据集覆盖饼图
    total_datasets = len(intersection_info['all_datasets'])
    intersection_count = len(intersection_datasets)
    only_some_count = len(only_some_configs)
    
    labels = ['Intersection', 'Only Some Configs']
    sizes = [intersection_count, only_some_count]
    colors = ['lightblue', 'lightcoral']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Dataset Coverage Distribution', fontsize=14, pad=20)
    
    # 第二个子图：各配置的数据集数量对比
    config_names = list(config_datasets.keys())
    dataset_counts = [len(datasets) for datasets in config_datasets.values()]
    
    bars = ax2.bar(config_names, dataset_counts, color='skyblue', alpha=0.8)
    
    # 添加数值标签
    for bar, count in zip(bars, dataset_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title('Dataset Count per Configuration', fontsize=14, pad=20)
    ax2.set_xlabel('Configurations', fontsize=12)
    ax2.set_ylabel('Number of Datasets', fontsize=12)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(save_path, exist_ok=True)
    config_names_str = '_'.join(list(config_datasets.keys())[:3])
    plt.savefig(f"{save_path}/dataset_coverage_{config_names_str}.png", 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    return fig

def generate_intersection_report(intersection_info, save_path="reports"):
    """生成交集分析报告
    
    Args:
        intersection_info (dict): 交集信息字典
        save_path (str): 报告保存路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("数据集交集分析报告")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # 基本信息
    report_lines.append("1. 基本信息")
    report_lines.append("-" * 30)
    report_lines.append(f"总数据集数量: {len(intersection_info['all_datasets'])}")
    report_lines.append(f"交集数据集数量: {len(intersection_info['intersection_datasets'])}")
    report_lines.append(f"仅部分配置包含的数据集数量: {len(intersection_info['only_some_configs'])}")
    report_lines.append("")
    
    # 交集数据集
    report_lines.append("2. 交集数据集")
    report_lines.append("-" * 30)
    for i, dataset in enumerate(intersection_info['intersection_datasets'], 1):
        report_lines.append(f"{i}. {dataset}")
    report_lines.append("")
    
    # 各配置的数据集
    report_lines.append("3. 各配置的数据集")
    report_lines.append("-" * 30)
    for config_name, datasets in intersection_info['config_datasets'].items():
        report_lines.append(f"{config_name}: {len(datasets)} 个数据集")
        for dataset in datasets:
            report_lines.append(f"  - {dataset}")
        report_lines.append("")
    
    # 性能统计
    report_lines.append("4. 交集数据集性能统计")
    report_lines.append("-" * 30)
    config_performance = intersection_info['config_performance']
    intersection_datasets = intersection_info['intersection_datasets']
    
    for dataset in intersection_datasets:
        report_lines.append(f"\n数据集: {dataset}")
        for config_name, performance in config_performance.items():
            if dataset in performance:
                perf = performance[dataset]
                report_lines.append(f"  {config_name}:")
                report_lines.append(f"    平均准确率: {perf['mean_acc']:.4f}")
                report_lines.append(f"    标准差: {perf['std_acc']:.4f}")
                report_lines.append(f"    最佳准确率: {perf['best_acc']:.4f}")
                report_lines.append(f"    Fold数量: {perf['fold_count']}")
    
    # 保存报告
    config_names_str = '_'.join(list(config_performance.keys())[:3])
    report_file = f"{save_path}/intersection_report_{config_names_str}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"报告已保存到: {report_file}")
    return report_file

def analyze_model_dataset_intersection(model_configs, save_path="analysis_results"):
    """主函数：分析模型数据集交集
    
    Args:
        model_configs (dict): 模型配置字典，每个配置包含自己的数据集集合
        save_path (str): 结果保存路径
    """
    print("开始数据集交集分析...")
    
    # 提取交集信息
    intersection_info = extract_dataset_intersection(model_configs)
    
    if intersection_info is None:
        print("Error: 无法提取交集信息")
        return
    
    # 打印基本信息
    print(f"\n=== 交集分析结果 ===")
    print(f"总数据集数量: {len(intersection_info['all_datasets'])}")
    print(f"交集数据集数量: {len(intersection_info['intersection_datasets'])}")
    print(f"交集数据集: {intersection_info['intersection_datasets']}")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 生成图表
    print("\n生成交集对比图...")
    plot_intersection_comparison(intersection_info, f"{save_path}/images")
    
    print("生成数据集覆盖图...")
    plot_dataset_coverage(intersection_info, f"{save_path}/images")
    
    # 生成报告
    print("生成分析报告...")
    generate_intersection_report(intersection_info, f"{save_path}/reports")
    
    print(f"\n分析完成！结果保存在: {save_path}")
    
    return intersection_info

# 使用示例
if __name__ == "__main__":
    # 示例配置 - 每个配置有自己的数据集集合
    model_configs = {
        'FAST-V0': {
            'path': "LazyLoad/FAST-30-16-10/V0-96-96-100-90-4-5e-05--cs=True",
            'datasets': ["EMO_03_SEED_V", "EMO_02_SEED_IV", "MI_01_KoreaU", "MI_02_ShanghaiU"],
            'fold_info': -1,  # 使用所有fold
            'n_fold': 9
        },
        'FAST-V3': {
            'path': "LazyLoad/FAST-30-16-10/V3-96-96-100-90-4-5e-05--cs=True",
            'datasets': ["EMO_03_SEED_V", "EMO_02_SEED_IV", "MI_01_KoreaU", "MI_03_Shin2017A"],
            'fold_info': -1,  # 使用所有fold
            'n_fold': 9
        }
    }
    
    # 执行分析
    intersection_info = analyze_model_dataset_intersection(model_configs) 
