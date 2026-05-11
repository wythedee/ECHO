CS_04_BCIC_Track3_split = {
    'train': range(0, 15),
    'val': range(15, 30),
    'test': range(30, 45),
}

MI_02_ShanghaiU_split = {
    'train': range(0, 15),          # files[:15]
    'val'  : range(15, 20),        # files[15:20]
    'test' : range(20, 25)         # files[20:25]
}

MI_04_BCI_IV_2a_split = {
    'train': range(0, 5),           # files[:6]
    'val'  : range(5, 7),           # files[6:8]
    'test' : range(7, 9)           # files[8:
}

MI_06_Schirrmeister2017_split = {
    'train': range(8),
    'val': range(8, 11),
    'test': range(11, 14),
}

MI_13_SHU_split = {
    'train': range(0, 15),
    'val': range(15, 20),
    'test': range(20, 25),
}

# EMO_02_SEED_IV_split = {
#     'train': range(0, 10),
#     'val': range(10, 12),
#     'test': range(12, 15),
# }

# # 2) SEED-V
# EMO_03_SEED_V_split = {
#     'train': range(0, 5),   # trials 0-4
#     'val'  : range(5, 10),  # trials 5-9
#     'test' : range(10, 15)  # trials 10-14
# }

# EMO_04_SEED_split = {
#     'train': range(0, 10),
#     'val': range(10, 12),
#     'test': range(12, 15),
# }

# EMO_04_SEED_trial_split = {
#     'train': range(0, 9),
#     'val'  : range(9, 12),
#     'test' : range(12, 15),
# }

# EMO_02_SEED_IV_trial_split = {
#     'train': range(0, 16),
#     'val'  : range(16, 20),
#     'test' : range(20, 24),
# }

# EMO_03_SEED_V_trial_split = {
#     'train': range(0, 5),
#     'val'  : range(5, 10),
#     'test' : range(10, 15),
# }

# SEED（15 subj → 45 subj）
# EMO_04_SEED_split = {
#     'train': range(0, 15),
#     'val'  : range(15, 30),
#     'test' : range(30, 45),
# }
# # SEED-IV（15 subj → 45 subj）
# EMO_02_SEED_IV_split = {
#     'train': range(0, 15),
#     'val'  : range(15, 30),
#     'test' : range(30, 45),
# }
# # SEED-V（15 subj → 45 subj）
# EMO_03_SEED_V_split = {
#     'train': range(0, 15),
#     'val'  : range(15, 30),
#     'test' : range(30, 45),
# }

EMO_02_SEED_IV_split = {
    'train': range(0, 10),
    'val': range(10, 12),
    'test': range(12, 15),
}

# 2) SEED-V
EMO_03_SEED_V_split = {
    'train': range(0, 5),
    'val'  : range(5, 10),
    'test' : range(10, 15)
}

EMO_04_SEED_split = {
    'train': range(0, 10),
    'val': range(10, 12),
    'test': range(12, 15),
}

EMO_06_FACED_split = {
    'train': range(0, 80),
    'val': range(80, 100),
    'test': range(100, 123),
}

# 3) MI-PhysioNet
MI_12_PhysioNet_split = {
    'train': range(0, 70),   # files[:70]
    'val'  : range(70, 89),  # files[70:89]
    'test' : range(89, 109)  # files[89:109]
}

MI_10_HeBin2021_LR_split = {
    'train': range(0, 40),
    'val': range(40, 49),
    'test': range(49, 59),
}

MI_10_HeBin2021_UD_split = {
    'train': range(0, 40),
    'val': range(40, 49),
    'test': range(49, 59),
}

MDD_01_Mumtaz_split = {
    # Based on lexicographic order of HDF5 keys:
    # H keys first (57 subjects):
    #   - train: 0..39, val: 40..47, test: 48..56
    # MDD keys next (62 subjects, offset=57):
    #   - train: 57..98, val: 99..108, test: 109..118
    'train': list(range(0, 40)) + list(range(57, 99)),
    'val':   list(range(40, 48)) + list(range(99, 109)),
    'test':  list(range(48, 57)) + list(range(109, 119)),
}

SLEEP_05_isruc_S1_split = {
    'train': range(0, 80),
    'val': range(80, 90),
    'test': range(90, 100),
}

SLEEP_05_isruc_S3_split = {
    'train': range(0, 8),
    'val': range(8, 9),
    'test': range(9, 10),
}

# CHB-MIT (EP_01_CHBMIT): subject-wise split following CBraMod
# train: chb01-chb20, val: chb21-chb22, test: chb23-chb24
EP_01_CHBMIT_split = {
    'train': range(0, 17),
    'val': range(17, 19),
    'test': range(19, 21),
}

STR_01_MentalArithmetic_split = {
    'train': range(0, 26),
    'val': range(26, 31),
    'test': range(31, 36),
}

# TUH_04_TUEV_Events (subject-wise):
# train subjects: 290 → 80% train (0..231), 20% val (232..289)
# test subjects: 80 eval subjects appended after train block (290..369)
TUH_04_TUEV_Events_split = {
    'train': range(0, 232),
    'val': range(232, 290),
    'test': range(290, 370),
}

# ATTEN_01_GoNogo: simple subject-wise split (supports cross-subject and cross-trials)
# After preprocessing, subjects are numbered '001'..'N'.
# Default: 60% train, 20% val, 20% test
ATTEN_01_GoNogo_split = {
    'train': range(0, 16),   # 26 subjects: 16/5/5 (approx 60/20/20)
    'val': range(16, 21),
    'test': range(21, 26),
}
