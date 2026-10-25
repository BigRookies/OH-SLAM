import os

import numpy as np
import random
import torch


def seed_everything(seed=42):
    """
        Set the `seed` value for torch and numpy seeds. Also turns on
        deterministic execution for cudnn.
        
        Parameters:
        - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")


def params2cpu(params):
    res = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu().contiguous().numpy()
        else:
            res[k] = v
    return res


def save_params(output_params, output_dir):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")

    # 🔥 Fix: Use compressed savez and handle large files with progressive fallback
    try:
        # Calculate total size
        total_size_mb = sum(v.nbytes for v in to_save.values() if isinstance(v, np.ndarray)) / (1024**2)
        print(f"  Total parameter size: {total_size_mb:.2f} MB")

        # Use compressed format for large files
        if total_size_mb > 500:  # If larger than 500MB, use compression
            print(f"  Using compressed format (size > 500 MB)")
            save_path_compressed = os.path.join(output_dir, "params.npz")
            np.savez_compressed(save_path_compressed, **to_save)
        else:
            np.savez(save_path, **to_save)

        print(f"  ✓ Parameters saved successfully to {save_path}")
        return True

    except OSError as e:
        if "No space left" in str(e):
            print(f"  ❌ ERROR: Disk space full during full save!")
            print(f"  Attempting progressive fallback strategy...")

            # Strategy 1: Save without semantic features
            try:
                print(f"  [Fallback 1/3] Saving without semantic features...")
                essential_keys = ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales']
                essential_params = {k: v for k, v in to_save.items() if k in essential_keys}

                save_path_essential = os.path.join(output_dir, "params_essential.npz")
                np.savez_compressed(save_path_essential, **essential_params)
                print(f"  ✓ Essential parameters saved to {save_path_essential}")
                print(f"  Note: Semantic features were skipped")
                return True

            except OSError as e2:
                if "No space left" in str(e2):
                    print(f"  ❌ ERROR: Still no space!")

                    # Strategy 2: Save only geometry (minimal reconstruction)
                    try:
                        print(f"  [Fallback 2/3] Saving geometry only...")
                        minimal_keys = ['means3D', 'unnorm_rotations', 'logit_opacities', 'log_scales']
                        minimal_params = {k: v for k, v in to_save.items() if k in minimal_keys}

                        save_path_minimal = os.path.join(output_dir, "params_minimal.npz")
                        np.savez_compressed(save_path_minimal, **minimal_params)
                        print(f"  ✓ Minimal parameters saved to {save_path_minimal}")
                        print(f"  Note: RGB and semantic features were skipped")
                        return True

                    except OSError as e3:
                        if "No space left" in str(e3):
                            print(f"  ❌ ERROR: Critical disk space issue!")

                            # Strategy 3: Save individual files (most reliable)
                            try:
                                print(f"  [Fallback 3/3] Saving individual parameter files...")
                                saved_count = 0
                                for key in ['means3D', 'unnorm_rotations', 'logit_opacities']:
                                    try:
                                        param_path = os.path.join(output_dir, f"{key}.npy")
                                        np.save(param_path, to_save[key])
                                        print(f"    ✓ Saved {key}")
                                        saved_count += 1
                                    except:
                                        print(f"    ✗ Failed to save {key}")
                                        continue

                                if saved_count > 0:
                                    print(f"  ✓ Saved {saved_count}/3 critical parameters individually")
                                    return True
                                else:
                                    print(f"  ✗ Failed to save any parameters")
                                    return False

                            except Exception as e4:
                                print(f"  ✗ All fallback strategies failed: {e4}")
                                return False
                        else:
                            raise
                else:
                    raise
        else:
            raise


def save_params_ckpt(output_params, output_dir, time_idx):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving checkpoint parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")

    # 🔥 Fix: Use compressed savez for checkpoints
    try:
        total_size_mb = sum(v.nbytes for v in to_save.values() if isinstance(v, np.ndarray)) / (1024**2)
        print(f"  Checkpoint size: {total_size_mb:.2f} MB")

        # Always use compression for checkpoints to save disk space
        np.savez_compressed(save_path, **to_save)
        print(f"  ✓ Checkpoint saved to {save_path}")

    except OSError as e:
        if "No space left" in str(e):
            print(f"  ❌ ERROR: Disk space full! Skipping checkpoint {time_idx}")
        else:
            raise


def save_seq_params(all_params, output_dir):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **params_to_save)


def save_seq_params_ckpt(all_params, output_dir,time_idx):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **params_to_save)