from EEG_Dataset.ADHD_01_AliMotie import ADHD_AliMotie
from EEG_Dataset.CS_01_OS_Muyun2024 import CS_Muyun2024, OS_Muyun2024
from EEG_Dataset.CS_02_Track3_Imagined_Speech import CS_Track3_Imagined_Speech
from EEG_Dataset.EMO_01_DEAP import EMO_DEAP
from EEG_Dataset.EMO_02_SEED_IV import EMO_SEED_IV
from EEG_Dataset.EMO_03_SEED_V import EMO_SEED_V
from EEG_Dataset.EMO_04_SEED import EMO_SEED
from EEG_Dataset.EMO_06_FACED import EMO_FACED
from EEG_Dataset.EP_01_CHBMIT import EP_CHBMIT
from EEG_Dataset.MI_01_SSVEP_KoreaU import MI_KoreaU, SSVEP_KoreaU
from EEG_Dataset.MI_02_ShanghaiU import MI_ShanghaiU
from EEG_Dataset.MI_03_Shin2017A import MI_Shin2017A
from EEG_Dataset.MI_04_BCI_IV_2a import MI_BCI_IV_2a
from EEG_Dataset.MI_05_Weibo2014 import MI_Weibo2014
from EEG_Dataset.MI_06_Schirrmeister2017 import MI_Schirrmeister2017
from EEG_Dataset.MI_07_Cho2017 import MI_Cho2017
from EEG_Dataset.MI_08_Track1_Few_shot import MI_Track1_Few_shot
from EEG_Dataset.MI_09_Track4_Upper_limb import MI_Track4_Upper_limb
from EEG_Dataset.MI_10_HeBin2021 import MI_HeBin2021_LR, MI_HeBin2021_UD
from EEG_Dataset.MI_11_HeBin2024 import MI_HeBin2024_LR, MI_HeBin2024_UD
from EEG_Dataset.MI_12_PhysioNet import MI_PhysioNet
from EEG_Dataset.MI_13_SHU import MI_SHU
from EEG_Dataset.SSVEP_02_TSinghuaU_Benchmark import SSVEP_TSinghuaU_Benchmark
from EEG_Dataset.SSVEP_03_TSinghuaU_eldBETA import SSVEP_TsinghuaU_eldBETA
from EEG_Dataset.SSVEP_04_TSinghuaU_BETA import SSVEP_TSinghuaU_BETA
from EEG_Dataset.STR_01_MentalArithmetic import STR_MentalArithmetic
from EEG_Dataset.EMO_05_THU_EP import EMO_THU_EP
from EEG_Dataset.MDD_01_Mumtaz import MDD_Mumtaz
from EEG_Dataset.CS_04_BCIC_Track3 import CS_BCIC_Track3
from EEG_Dataset.SLEEP_05_isruc import SLEEP_ISRUC_S1, SLEEP_ISRUC_S3
from EEG_Dataset.TUH_04_TUEV_Events import TUH_TUEV_Events
from EEG_Dataset.ATTEN_01_GoNogo import ATTEN_GoNogo

DATASET = {
    'MI_KoreaU': MI_KoreaU,
    'ADHD_AliMotie': ADHD_AliMotie,
    'CS_Muyun2024': CS_Muyun2024,
    'OS_Muyun2024': OS_Muyun2024,
    'CS_Track3_Imagined_Speech': CS_Track3_Imagined_Speech,
    'EMO_DEAP': EMO_DEAP,
    'EMO_SEED_IV': EMO_SEED_IV,
    'EMO_SEED_V': EMO_SEED_V,
    'EMO_THU-EP': EMO_THU_EP,
    'EMO_SEED': EMO_SEED,
    'MI_ShanghaiU': MI_ShanghaiU,
    'MI_Shin2017A': MI_Shin2017A,
    'MI_BCI_IV_2a': MI_BCI_IV_2a,
    'MI_Weibo2014': MI_Weibo2014,
    'MI_Schirrmeister2017': MI_Schirrmeister2017,
    'MI_Cho2017': MI_Cho2017,
    'MI_Track1_Few_shot': MI_Track1_Few_shot,
    'MI_Track4_Upper_limb': MI_Track4_Upper_limb,
    'MI_HeBin2021_LR': MI_HeBin2021_LR,
    'MI_HeBin2021_UD': MI_HeBin2021_UD,
    'MI_HeBin2024_LR': MI_HeBin2024_LR,
    'MI_HeBin2024_UD': MI_HeBin2024_UD,
    'MI_PhysioNet': MI_PhysioNet,
    'MI_SHU': MI_SHU,
    'SSVEP_KoreaU': SSVEP_KoreaU,
    'SSVEP_TSinghuaU_Benchmark': SSVEP_TSinghuaU_Benchmark,
    'SSVEP_TsinghuaU_eldBETA': SSVEP_TsinghuaU_eldBETA,
    'SSVEP_TSinghuaU_BETA': SSVEP_TSinghuaU_BETA,
    'CS_04_BCIC_Track3': CS_BCIC_Track3,
    'ATTEN_GoNogo': ATTEN_GoNogo,
}

DATASET_BASIC_TASKS = {
    'MI_KoreaU': MI_KoreaU,
    'SSVEP_KoreaU': SSVEP_KoreaU,
    'EMO_DEAP': EMO_DEAP,
    'EMO_SEED_IV': EMO_SEED_IV,
    'EMO_SEED_V': EMO_SEED_V,
    'EMO_THU-EP': EMO_THU_EP,
    'EMO_SEED': EMO_SEED,
    'EMO_FACED': EMO_FACED,
    'EP_CHBMIT': EP_CHBMIT,
    'MI_ShanghaiU': MI_ShanghaiU,
    'MI_Shin2017A': MI_Shin2017A,
    'MI_BCI_IV_2a': MI_BCI_IV_2a,
    'MI_Weibo2014': MI_Weibo2014,
    'MI_Schirrmeister2017': MI_Schirrmeister2017,
    'MI_Cho2017': MI_Cho2017,
    'MI_Track1_Few_shot': MI_Track1_Few_shot,
    'MI_Track4_Upper_limb': MI_Track4_Upper_limb,
    'MI_HeBin2021_LR': MI_HeBin2021_LR,
    'MI_HeBin2021_UD': MI_HeBin2021_UD,
    'MI_HeBin2024_LR': MI_HeBin2024_LR,
    'MI_HeBin2024_UD': MI_HeBin2024_UD,
    'MI_PhysioNet': MI_PhysioNet,
    'MI_SHU': MI_SHU,
    'SSVEP_TSinghuaU_Benchmark': SSVEP_TSinghuaU_Benchmark,
    'SSVEP_TsinghuaU_eldBETA': SSVEP_TsinghuaU_eldBETA,
    'SSVEP_TSinghuaU_BETA': SSVEP_TSinghuaU_BETA,
    'SLEEP_ISRUC_S1': SLEEP_ISRUC_S1,
    'SLEEP_ISRUC_S3': SLEEP_ISRUC_S3,
    'MDD_Mumtaz': MDD_Mumtaz,
    'STR_MentalArithmetic': STR_MentalArithmetic,
    'CS_BCIC_Track3': CS_BCIC_Track3,
    'TUH_TUEV_Events': TUH_TUEV_Events,
    'ATTEN_GoNogo': ATTEN_GoNogo,
}

# Only Left Right hand MI datasets
DATASET_MI_LR = {
    'MI_KoreaU': MI_KoreaU,
    'MI_ShanghaiU': MI_ShanghaiU,
    'MI_Shin2017A': MI_Shin2017A,
    'MI_BCI_IV_2a': MI_BCI_IV_2a,
    'MI_Weibo2014': MI_Weibo2014,
    'MI_Schirrmeister2017': MI_Schirrmeister2017,
    'MI_Cho2017': MI_Cho2017,
    'MI_Track1_Few_shot': MI_Track1_Few_shot,
    'MI_HeBin2021_LR': MI_HeBin2021_LR,
    'MI_HeBin2024_LR': MI_HeBin2024_LR,
}
