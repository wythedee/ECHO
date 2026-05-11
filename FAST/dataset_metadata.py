from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    h5_name: str
    classes: list[str]


CHANNEL_NAMES = [f"CH{i:03d}" for i in range(1, 76)]
CHANNEL_ZONES = {"All": CHANNEL_NAMES}


DATASET_BASIC_TASKS = {
    "CS_BCIC_Track3": DatasetInfo("CS_04_BCIC_Track3", ["CS/Hello", "CS/Help-me", "CS/Stop", "CS/Thank-you", "CS/Yes"]),
    "MI_Default": DatasetInfo("MI_Default", ["MI/Left", "MI/Right"]),
    "EMO_DEAP": DatasetInfo("EMO_01_DEAP", ["EMO/Neutral", "EMO/Sad", "EMO/Fear", "EMO/Happy"]),
    "EMO_SEED_V": DatasetInfo("EMO_03_SEED_V", ["EMO/Disgust", "EMO/Fear", "EMO/Sad", "EMO/Neutral", "EMO/Happy"]),
    "EMO_SEED_IV": DatasetInfo("EMO_02_SEED_IV", ["EMO/Neutral", "EMO/Sad", "EMO/Fear", "EMO/Happy"]),
    "EMO_SEED": DatasetInfo("EMO_04_SEED", ["EMO/Neg", "EMO/Pos"]),
    "EMO_THU_EP": DatasetInfo("EMO_05_THU-EP", ["EMO/Neg", "EMO/Pos"]),
    "EMO_FACED": DatasetInfo("EMO_06_FACED", ["EMO/Amusement", "EMO/Inspiration", "EMO/Joy", "EMO/Tenderness", "EMO/Anger", "EMO/Fear", "EMO/Disgust", "EMO/Sadness", "EMO/Neutral"]),
    "EP_CHBMIT": DatasetInfo("EP_01_CHBMIT", ["EP/NonSeizure", "EP/Seizure"]),
    "MDD_Mumtaz": DatasetInfo("MDD_01_Mumtaz", ["MDD/H", "MDD/MDD"]),
    "MI_KoreaU": DatasetInfo("MI_01_KoreaU", ["MI/Left", "MI/Right"]),
    "MI_ShanghaiU": DatasetInfo("MI_02_ShanghaiU", ["MI/Left", "MI/Right"]),
    "MI_Shin2017A": DatasetInfo("MI_03_Shin2017A", ["MI/Left", "MI/Right"]),
    "MI_BCI_IV_2a": DatasetInfo("MI_04_BCI_IV_2a", ["MI/Left", "MI/Right", "MI/BothFeet", "MI/Tongue"]),
    "MI_Weibo2014": DatasetInfo("MI_05_Weibo2014", ["MI/Left", "MI/Right"]),
    "MI_Schirrmeister2017": DatasetInfo("MI_06_Schirrmeister2017", ["MI/Left", "MI/Right"]),
    "MI_Cho2017": DatasetInfo("MI_07_Cho2017", ["MI/Left", "MI/Right"]),
    "MI_Track1_Few_shot": DatasetInfo("MI_08_Track1_Few_shot", ["MI/Left", "MI/Right"]),
    "MI_Track4_Upper_limb": DatasetInfo("MI_09_Track4_Upper_limb", ["MI/Cylin", "MI/Sphe", "MI/Lumbrical"]),
    "MI_HeBin2021_LR": DatasetInfo("MI_10_HeBin2021_LR", ["MI/Left", "MI/Right"]),
    "MI_HeBin2021_UD": DatasetInfo("MI_10_HeBin2021_UD", ["MI/Up", "MI/Down"]),
    "MI_HeBin2024_LR": DatasetInfo("MI_11_HeBin2024_LR", ["MI/Left", "MI/Right"]),
    "MI_HeBin2024_UD": DatasetInfo("MI_11_HeBin2024_UD", ["MI/Up", "MI/Down"]),
    "MI_PhysioNet": DatasetInfo("MI_12_PhysioNet", ["MI/Left", "MI/Right", "MI/BothFists", "MI/BothFeet"]),
    "MI_SHU": DatasetInfo("MI_13_SHU", ["MI/Left", "MI/Right"]),
    "SLEEP_ISRUC_S1": DatasetInfo("SLEEP_05_isruc_S1", ["SLEEP/W", "SLEEP/N1", "SLEEP/N2", "SLEEP/N3", "SLEEP/REM"]),
    "SLEEP_ISRUC_S3": DatasetInfo("SLEEP_05_isruc_S3", ["SLEEP/W", "SLEEP/N1", "SLEEP/N2", "SLEEP/N3", "SLEEP/REM"]),
    "STR_MentalArithmetic": DatasetInfo("STR_01_MentalArithmetic", ["STR/Control", "STR/MentalArithmetic"]),
    "TUH_TUEV_Events": DatasetInfo("TUH_04_TUEV_Events", ["TUEV/SPSW", "TUEV/GPED", "TUEV/PLED", "TUEV/EYEM", "TUEV/ARTF", "TUEV/BCKG"]),
    "ATTEN_GoNogo": DatasetInfo("ATTEN_01_GoNogo", ["ATTEN/Rest", "ATTEN/Attention"]),
}
