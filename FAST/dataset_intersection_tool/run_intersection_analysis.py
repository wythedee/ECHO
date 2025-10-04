#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集交集分析运行脚本
用于比较同一个模型在不同数据集下训练的结果差异
"""

from dataset_intersection_analysis import analyze_model_dataset_intersection

def main():
    """主函数：配置并运行数据集交集分析"""
    
    # ===== 配置区域 =====
    # 每个模型配置都有自己的数据集集合
    # 格式: {
    #   '配置名称': {
    #       'path': str,           # 模型路径
    #       'datasets': list,      # 该配置使用的数据集列表
    #       'fold_info': int/list/-1,  # fold信息
    #       'n_fold': int          # fold数量
    #   }
    # }
    weak_datasaets = "MI_03_Shin2017A,MI_04_BCI_IV_2a,MI_09_Track4_Upper_limb,MI_10_HeBin2021_LR,MI_10_HeBin2021_UD,MI_11_HeBin2024_LR,MI_11_HeBin2024_UD,EMO_03_SEED_V,EMO_04_SEED".split(',')
    all_datasets = "MI_01_KoreaU,MI_03_Shin2017A,MI_04_BCI_IV_2a,MI_05_Weibo2014,MI_06_Schirrmeister2017,MI_07_Cho2017,MI_09_Track4_Upper_limb,MI_10_HeBin2021_LR,MI_10_HeBin2021_UD,MI_11_HeBin2024_LR,MI_11_HeBin2024_UD,MI_12_PhysioNet,EMO_02_SEED_IV,EMO_03_SEED_V,EMO_04_SEED,EMO_05_THU-EP".split(',')
    
    model_configs = {
        'Original': {
            'path': "LazyLoad/FAST-30-16-10/V3-96-96-100-90-4-5e-05--cs=True",
            'datasets': all_datasets,
            'fold_info': -1,  # 使用所有fold
            'n_fold': 9
        },
        'Weak': {
            'path': "LazyLoad/FAST-30-9-10/V3-96-96-100-90-4-5e-05--cs=True",
            'datasets': weak_datasaets,
            'fold_info': -1,  # 使用所有fold
            'n_fold': 9
        },
        # 可以添加更多配置
        # 'FAST-V1': {
        #     'path': "LazyLoad/FAST-30-16-10/V1-96-96-100-90-4-5e-05--cs=True",
        #     'datasets': ["EMO_03_SEED_V", "MI_01_KoreaU", "MI_04_BCI_IV_2a"],
        #     'fold_info': [0, 1, 2],  # 使用fold 0,1,2
        #     'n_fold': 9
        # },
    }
    
    # model_configs = {
    #     'Test1': {
    #         'path': 'LazyLoad/FAST-10-EMO_03_SEED_V,EMO_02_SEED_IV,MI_02_ShanghaiU-30/V3-96-96-100-90-4-5e-05--cs=True',
    #         'datasets': ['EMO_03_SEED_V', 'EMO_02_SEED_IV', 'MI_02_ShanghaiU'],
    #         'fold_info': -1,
    #         'n_fold': 9
    #     },
    #     'Test2': {
    #         'path': 'LazyLoad/FAST-10-EMO_03_SEED_V,EMO_02_SEED_IV,MI_01_KoreaU-30/V3-96-96-100-90-4-5e-05--cs=True',
    #         'datasets': ['EMO_03_SEED_V', 'EMO_02_SEED_IV', 'MI_01_KoreaU'],
    #         'fold_info': -1,
    #         'n_fold': 9
    #     }
    # }
    # 设置结果保存路径
    save_path = "dataset_intersection_tool"
    
    # ===== 运行分析 =====
    print("=" * 60)
    print("数据集交集分析工具")
    print("=" * 60)
    print(f"配置数量: {len(model_configs)}")
    
    # 显示每个配置的数据集
    for config_name, config_info in model_configs.items():
        print(f"{config_name}: {len(config_info['datasets'])} 个数据集")
        print(f"  - {', '.join(config_info['datasets'])}")
    
    print(f"结果保存路径: {save_path}")
    print("=" * 60)
    
    # 执行分析
    intersection_info = analyze_model_dataset_intersection(
        model_configs=model_configs,
        save_path=save_path
    )
    
    if intersection_info:
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
        print(f"交集数据集数量: {len(intersection_info['intersection_datasets'])}")
        print(f"交集数据集: {intersection_info['intersection_datasets']}")
        print(f"\n生成的文件:")
        print(f"  - 图片: {save_path}/images/")
        print(f"  - 报告: {save_path}/reports/")
    else:
        print("\n分析失败，请检查配置和路径")

if __name__ == "__main__":
    main() 
