import h5py
import numpy as np
import os
import shutil
import fcntl
from numpy.lib.format import open_memmap
from typing import Optional

DATASET_ROOT = "/path/to/your/datasets_root"  # Modify as needed

SHM_ROOT = "/dev/shm"

__all__ = ["h5Dataset", "pad_trial"]


def pad_trial(x: np.ndarray, target_len: int):
    """Zero-pad / truncate to target_len (last dimension)."""
    if x.shape[-1] >= target_len:
        return x[..., :target_len]
    pad_w = target_len - x.shape[-1]
    return np.pad(x, ((0, 0), (0, 0), (0, pad_w)), mode="constant")


class h5Dataset:
    """Lazy-loading h5 dataset, supports automatic copying to `/dev/shm` for faster IO."""

    def __init__(self, name: str, classes, root: str = DATASET_ROOT, use_shm: bool = True, use_memmap: bool = True, time_len: Optional[int] = 10, force_rebuild_memmap: bool = False):
        self.name = name
        self.classes = classes
        self.force_rebuild_memmap = force_rebuild_memmap

        orig_path = os.path.join(root, name.split("_")[0], f"{name}.h5")

        # -------- copy file to shared memory (tmpfs) --------
        memmap_flag = use_memmap and os.path.isdir(SHM_ROOT)

        if use_shm and os.path.isdir(SHM_ROOT):
            # if shared memory has not enough space, fallback to disk read
            try:
                shm_usage = shutil.disk_usage(SHM_ROOT)
                file_size = os.path.getsize(orig_path)
                # reserve 128 MB buffer, skip copy if space is not enough
                if file_size + 128 * 2**20 > shm_usage.free:
                    print(f"[h5Dataset] Skip copy: not enough free space in /dev/shm ({shm_usage.free/1e9:.2f} GB left, need {file_size/1e9:.2f} GB). Read from disk.")
                    self.file_path = orig_path
                    memmap_flag = False
                elif memmap_flag and os.path.exists(os.path.join(SHM_ROOT, f"{name}_X.npy")) and os.path.exists(os.path.join(SHM_ROOT, f"{name}_Y.npy")) and not self.force_rebuild_memmap:
                    self.file_path = orig_path
                    print(f"[h5Dataset] Skip .h5 copy: memmap already exists in /dev/shm")
                else:
                    self.file_path = self._copy_to_shm(orig_path)
            except Exception as e:
                print(f"[h5Dataset] Error checking /dev/shm space: {e}. Fallback to disk.")
                self.file_path = orig_path
                memmap_flag = False
        else:
            # fallback to disk read
            self.file_path = orig_path

        # ----------------------------------------------

        # open HDF5 for first time conversion or metadata supplement
        self.h5file = h5py.File(self.file_path, "r")
        # only keep groups that contain both 'X' and 'Y'
        all_keys = list(self.h5file.keys())
        self.sub_keys = [
            k for k in all_keys
            if isinstance(self.h5file[k], h5py.Group) and ("X" in self.h5file[k]) and ("Y" in self.h5file[k])
        ]
        if len(self.sub_keys) != len(all_keys):
            print(f"[h5Dataset] Filtered subjects: using {len(self.sub_keys)}/{len(all_keys)} valid groups (with X & Y)")
        self.n_subjects = len(self.sub_keys)
        if self.n_subjects == 0:
            raise ValueError(f"[h5Dataset] No valid subject groups found in {self.file_path}. Expect groups with 'X' and 'Y'.")
        self.sfreq = 250
        self.time_len = time_len
        self._Y_all = [self.h5file[k]["Y"][()] for k in self.sub_keys]
        self._n_trials = [int(len(y)) for y in self._Y_all]

        # ---------------- memmap acceleration ----------------
        self.use_memmap = memmap_flag
        if self.use_memmap:
            self._prepare_memmap()

    def _prepare_memmap(self):
        """Converts to memmap once if it doesn't exist; otherwise, opens directly."""
        mm_x_path = os.path.join(SHM_ROOT, f"{self.name}_X.npy")
        mm_y_path = os.path.join(SHM_ROOT, f"{self.name}_Y.npy")

        # record start global trial idx for each subject
        self.sub_offsets = np.cumsum([0] + self._n_trials[:-1]).astype(np.int64)

        # lock file to prevent concurrent access
        lock_path = mm_x_path + ".lock"
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)

            # force rebuild: delete existing npy files if requested
            if self.force_rebuild_memmap:
                for file_path in [mm_x_path, mm_y_path]:
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            print(f"[h5Dataset] Removed existing memmap file: {file_path}")
                        except Exception as e:
                            print(f"[h5Dataset] Failed to remove {file_path}: {e}")

            if not (os.path.exists(mm_x_path) and os.path.exists(mm_y_path)):
                total_trials = int(sum(self._n_trials))
                # get channel number & target length
                sample = self.h5file[self.sub_keys[0]]["X"][0]
                C = sample.shape[0]
                target_len = self.sfreq * self.time_len

                print(f"[h5Dataset] Building memmap for {self.name}: {total_trials} trials…")
                X_mm = open_memmap(mm_x_path, mode="w+", dtype=np.float32, shape=(total_trials, C, target_len))
                Y_mm = open_memmap(mm_y_path, mode="w+", dtype=np.int16, shape=(total_trials,))

                gidx = 0
                for sid, key in enumerate(self.sub_keys):
                    X_ds = self.h5file[key]["X"]
                    Y_arr = self._Y_all[sid]
                    for tid in range(self._n_trials[sid]):
                        x = X_ds[tid]
                        X_mm[gidx] = pad_trial(x[None, ...], target_len)[0]
                        Y_mm[gidx] = int(Y_arr[tid])
                        gidx += 1
                X_mm.flush(); Y_mm.flush()
                print(f"[h5Dataset] Memmap saved to /dev/shm, size≈{os.path.getsize(mm_x_path)/1e9:.2f} GB")

            # open as read-only memmap
            self.X_mm = open_memmap(mm_x_path, mode="r", dtype=np.float32)
            self.Y_mm = open_memmap(mm_y_path, mode="r", dtype=np.int16)
            print(f"[h5Dataset] Memmap opened from /dev/shm, size≈{os.path.getsize(mm_x_path)/1e9:.2f} GB")

        # release lock
        try:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
        except Exception:
            pass

        # ----------  Clean up SHM copy ----------
        if isinstance(self.file_path, str) and self.file_path.startswith(SHM_ROOT):
            try:
                self.h5file.close()
            except Exception:
                pass

            try:
                os.remove(self.file_path)
                print(f"[h5Dataset] Removed SHM copy {self.file_path} to free space.")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[h5Dataset] Failed to remove SHM copy: {e}")

            # remove lock file
            lock_copy = self.file_path + ".lock"
            try:
                os.remove(lock_copy)
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def get_trial(self, sub_idx: int, trial_idx: int):
        """Returns (X, y); X: (C, T) float32. If memmap is available, slices directly."""
        if self.use_memmap and hasattr(self, "X_mm"):
            global_idx = int(self.sub_offsets[sub_idx] + trial_idx)
            x = self.X_mm[global_idx]
            y = int(self.Y_mm[global_idx])
            return x, y

        # fallback to h5 direct read
        x = self.h5file[self.sub_keys[sub_idx]]["X"][trial_idx]
        y = self.get_label(sub_idx, trial_idx)
        return x.astype(np.float32), y

    # ------------------------- util -------------------------
    def n_trials(self, sub_idx: int):
        return self._n_trials[sub_idx]

    def get_label(self, sub_idx: int, trial_idx: int):
        return int(self._Y_all[sub_idx][trial_idx])

    # Compatible with old interface
    def get_sub(self, idx: int):
        x = self.h5file[self.sub_keys[idx]]["X"][()]
        y = self.h5file[self.sub_keys[idx]]["Y"][()]
        return x.astype(np.float32), y

    # --------------------------------------------------------
    def close(self):
        try:
            self.h5file.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def _copy_to_shm(self, orig_path):
        """Safe copy file to /dev/shm; fallback to original path if failed."""
        shm_path = os.path.join(SHM_ROOT, f"{self.name}.h5")

        lock_path = shm_path + ".lock"
        try:
            with open(lock_path, "w") as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)

                if not os.path.exists(shm_path) or os.path.getsize(shm_path) != os.path.getsize(orig_path):
                    tmp_path = shm_path + f".{os.getpid()}.tmp"
                    try:
                        print(f"[h5Dataset] Copying {orig_path} → {shm_path} …")
                        shutil.copy2(orig_path, tmp_path)
                        os.replace(tmp_path, shm_path)
                        print("[h5Dataset] Copy finished.")
                    except OSError as e:
                        # Copy failed (possibly due to insufficient space), fallback and prompt
                        print(f"[h5Dataset] Copy failed: {e}. Using original file.")
                        if os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass
                        return orig_path
        finally:
            try:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
            except Exception:
                pass

        return shm_path
