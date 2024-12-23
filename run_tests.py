import subprocess
import os
import argparse

def main(args):
    # Define tests
    tests = ["N9", "N10", "N11", "N12", "D9", "D10", "D11", "D12", "V9", "V10", "V11", "V12"]

    dataset_root = args.dataset_root
    save_root = args.save_root

    # Create dataset and save directories
    dataset_dirs = [os.path.join(dataset_root, test) for test in tests]
    save_dirs = [os.path.join(save_root, test) for test in tests]

    # Ensure save directories exist
    for save_dir in save_dirs:
        os.makedirs(save_dir, exist_ok=True)

    # Model and script configuration
    arch = args.arch
    checkpoint_epoch = args.epoch
    model_path = os.path.join(
        args.model_root, f"checkpoint_ep{checkpoint_epoch}.pt"
    )
    script_path = args.script_path

    # Iterate over dataset and save directory pairs and call the script
    for dataset_dir, save_dir in zip(dataset_dirs, save_dirs):
        cmd = (
            f"CUDA_VISIBLE_DEVICES=0 python3 {script_path} "
            f"--dataset-dir \"{dataset_dir}\" "
            f"--arch \"{arch}\" "
            f"--bs 1 "
            f"--mode test_wo_reset "
            f"--save-dir \"{save_dir}\" "
            f"--model-path \"{model_path}\""
            f"--model-options \'num_res_blocks\':2"
        )
        try:
            print(f"Running: {cmd}")
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                text=True,
                capture_output=True
            )
            print("Output:", result.stdout)
            print("Errors:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the command:\n{cmd}")
            print("Error message:", e.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test scripts for datasets.")
    parser.add_argument("--dataset-root", default="/root/test_savedirs/", help="Root directory of the datasets.")
    parser.add_argument("--save-root", default="/root/test_results/", help="Root directory to save outputs.")
    parser.add_argument("--arch", default="AdaptiveFlowNet", help="Model architecture.")
    parser.add_argument("--epoch", type=int, default=20, help="Checkpoint epoch number.")
    parser.add_argument("--model-root", required=True, help="Root directory of the model checkpoints.")
    parser.add_argument("--script-path", default="main_dsec.py", help="Path to the script to execute.")

    args = parser.parse_args()
    main(args)
