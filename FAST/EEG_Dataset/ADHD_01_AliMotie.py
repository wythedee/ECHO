import torch
import einops
import mne
import numpy as np
import os
import scipy
import h5py
import multiprocessing as mp
import multiprocessing.dummy as dmp
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from share import THREADS, META, SRC_FOLDER, DATA_FOLDER, pipeline

NAME = 'ADHD_01_AliMotie'
ADHD_SUBJECTS = [
    'v177', 'v215', 'v179', 'v22p', 'v227', 'v40p', 'v284', 'v238', 'v244', 'v36p', 'v231', 'v236', 'v32p', 'v204', 'v3p', 
    'v254', 'v37p', 'v6p', 'v213', 'v33p', 'v270', 'v27p', 'v279', 'v12p', 'v183', 'v274', 'v35p', 'v206', 'v31p', 'v234', 
    'v246', 'v29p', 'v18p', 'v14p', 'v250', 'v196', 'v25p', 'v286', 'v288', 'v198', 'v21p', 'v10p', 'v8p', 'v39p', 'v265', 
    'v219', 'v15p', 'v181', 'v24p', 'v20p', 'v200', 'v38p', 'v209', 'v173', 'v34p', 'v263', 'v30p', 'v28p', 'v190', 'v19p', 'v1p']
Control_SUBJECTS = [
    'v304', 'v58p', 'v303', 'v44p', 'v50p', 'v48p', 'v113', 'v114', 'v121', 'v54p', 'v51p', 'v60p', 'v49p', 'v131', 'v302', 
    'v305', 'v55p', 'v138', 'v143', 'v41p', 'v129', 'v59p', 'v127', 'v120', 'v115', 'v45p', 'v112', 'v53p', 'v297', 'v151', 
    'v123', 'v310', 'v299', 'v111', 'v57p', 'v116', 'v118', 'v43p', 'v109', 'v107', 'v149', 'v306', 'v47p', 'v140', 'v147', 
    'v308', 'v42p', 'v117', 'v110', 'v125', 'v46p', 'v298', 'v52p', 'v309', 'v133', 'v300', 'v307', 'v134', 'v56p', 'v108']

SUBJECTS = ADHD_SUBJECTS + Control_SUBJECTS
CH_NAMES = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']

ADHD_AliMotie = META(NAME, CH_NAMES, SUBJECTS, ['ADHD/Control', 'ADHD/Diagnosed'])

def proc_one(sub):
    src_sfreq = 128
    if sub in ADHD_SUBJECTS:
        prefix = 'ADHD'
        y = 1
    else:
        prefix = 'Control'
        y = 0
    fn = f'{SRC_FOLDER}/{NAME}/{prefix}/{sub}.mat'
    data = (scipy.io.loadmat(fn)[sub]/1000).astype(np.float32).T
    raw = mne.io.RawArray(data, mne.create_info(ch_names=CH_NAMES, sfreq=src_sfreq, ch_types='eeg')).resample(250, npad='auto')
    raw = raw.filter(l_freq=1, h_freq=40, verbose=False)
    x = raw.get_data().astype(np.float32)
    sfreq = 250
    x = torch.from_numpy(x).unfold(-1, sfreq*10, sfreq*10)
    x = einops.rearrange(x, 'C N T -> N C T').numpy()
    y = np.array([y] * x.shape[0])
    print(sub, x.shape, y, x.dtype)
    x = pipeline(x, CH_NAMES)
    return sub, x, y

def proc_all():
    with mp.Pool(min(len(SUBJECTS), THREADS)) as pool:
        res = pool.map(proc_one, SUBJECTS)
    with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
        for sub, X, Y in res:
            f.create_dataset(f'{sub}/X', data=X)
            f.create_dataset(f'{sub}/Y', data=Y)
            print(sub, X.shape, Y.shape, np.unique(Y, return_counts=True))

if __name__ == '__main__':
    proc_all()