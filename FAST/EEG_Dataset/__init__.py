from EEG_Dataset.EMO_02_SEED_IV import EMO_SEED_IV
from EEG_Dataset.EMO_03_SEED_V import EMO_SEED_V
from EEG_Dataset.EMO_04_SEED import EMO_SEED
from EEG_Dataset.EMO_05_THU_EP import EMO_THU_EP
from EEG_Dataset.MI_01_SSVEP_KoreaU import MI_KoreaU
from EEG_Dataset.MI_03_Shin2017A import MI_Shin2017A
from EEG_Dataset.MI_04_BCI_IV_2a import MI_BCI_IV_2a
from EEG_Dataset.MI_05_Weibo2014 import MI_Weibo2014
from EEG_Dataset.MI_06_Schirrmeister2017 import MI_Schirrmeister2017
from EEG_Dataset.MI_07_Cho2017 import MI_Cho2017
from EEG_Dataset.MI_09_Track4_Upper_limb import MI_Track4_Upper_limb
from EEG_Dataset.MI_10_HeBin2021 import MI_HeBin2021_LR, MI_HeBin2021_UD
from EEG_Dataset.MI_11_HeBin2024 import MI_HeBin2024_LR, MI_HeBin2024_UD
from EEG_Dataset.MI_12_PhysioNet import MI_PhysioNet
from EEG_Dataset.SLEEP_05_isruc import SLEEP_ISRUC_S1, SLEEP_ISRUC_S3

DATASET_BASIC_TASKS = {
    'MI_KoreaU': MI_KoreaU,
    'MI_Shin2017A': MI_Shin2017A,
    'MI_BCI_IV_2a': MI_BCI_IV_2a,
    'MI_Weibo2014': MI_Weibo2014,
    'MI_Schirrmeister2017': MI_Schirrmeister2017,
    'MI_Cho2017': MI_Cho2017,
    'MI_Track4_Upper_limb': MI_Track4_Upper_limb,
    'MI_HeBin2021_LR': MI_HeBin2021_LR,
    'MI_HeBin2021_UD': MI_HeBin2021_UD,
    'MI_HeBin2024_LR': MI_HeBin2024_LR,
    'MI_HeBin2024_UD': MI_HeBin2024_UD,
    'MI_PhysioNet': MI_PhysioNet,
    'EMO_SEED_IV': EMO_SEED_IV,
    'EMO_SEED_V': EMO_SEED_V,
    'EMO_SEED': EMO_SEED,
    'EMO_THU-EP': EMO_THU_EP,
    'SLEEP_ISRUC_S1': SLEEP_ISRUC_S1,
    'SLEEP_ISRUC_S3': SLEEP_ISRUC_S3,
}

DATASET = DATASET_BASIC_TASKS.copy()

DATASET_MI_LR = {
    'MI_KoreaU': MI_KoreaU,
    'MI_Shin2017A': MI_Shin2017A,
    'MI_BCI_IV_2a': MI_BCI_IV_2a,
    'MI_Weibo2014': MI_Weibo2014,
    'MI_Schirrmeister2017': MI_Schirrmeister2017,
    'MI_Cho2017': MI_Cho2017,
    'MI_HeBin2021_LR': MI_HeBin2021_LR,
    'MI_HeBin2024_LR': MI_HeBin2024_LR,
}
