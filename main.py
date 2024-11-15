import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import soundfile as sf
from scipy.signal import resample
import torch
from teacher_code_modified import compute_stoi  # Import your STOI computation function
import os
import json
from pystoi.stoi import stoi

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(path_data: DictConfig) -> None:
    try:
        print("Starting the process...")

        # Open the reference JSON file
        with open(path_data.test_path.ref_file, 'r') as ref_file:
            ref_json = json.load(ref_file)
            print("Loaded reference JSON file.")

            # Build file paths
            spin_file_path = Path(path_data.test_path.spin_folder) / f"{ref_json[0]['signal']}.wav"
            target_file_path = Path(path_data.test_path.scenes_folder) / f"{ref_json[0]['scene']}_target_anechoic.wav"

            # Print paths for debugging
            print(f"Spin file path: {spin_file_path}")
            print(f"Target file path: {target_file_path}")

            # Read audio files
            try:
                spin, spin_sr = sf.read(spin_file_path)
                target, target_sr = sf.read(target_file_path)
                print("Audio files loaded successfully.")
            except FileNotFoundError as e:
                print(f"Error loading audio files: {e}")
                return

            # Resample both signals to the specified sample rate in config.yml
            new_sr = path_data.sample_rate
            try:
                spin_resampled = resample(spin, int(len(spin) * new_sr / spin_sr))
                target_resampled = resample(target, int(len(target) * new_sr / target_sr))
                print(f"Audio files resampled to {new_sr} Hz.")
            except Exception as e:
                print(f"Error during resampling: {e}")
                return

            # Separate left and right channels for STOI computation (if stereo)
            try:
                le_spin, re_spin = spin_resampled[:, 0], spin_resampled[:, 1]
                le_target, re_target = target_resampled[:, 0], target_resampled[:, 1]
                print("Separated left and right channels for STOI calculation.")
            except IndexError as e:
                print(f"Error separating channels: {e}")
                return

            # Check if the target and spin signals are valid
            print(f"Left Target Length: {len(le_target)}")
            print(f"Left Spin Length: {len(le_spin)}")

            # Calculate STOI for each ear
            try:
                # print("Before STOI calculation for Left Ear")
                stoi_score_le = compute_stoi(le_target, le_spin)
                # print("After STOI calculation for Left Ear")

                # print("Before STOI calculation for Right Ear")
                stoi_score_re = compute_stoi(re_target, re_spin)
                # print("After STOI calculation for Right Ear")

                # Print STOI scores
                print(f"STOI Score (Left Ear): {stoi_score_le}")
                print(f"STOI Score (Right Ear): {stoi_score_re}")
                
                # # Compute STOI using pystoi for both ears
                pystoi_left = stoi(le_target, le_spin, spin_sr, extended=False)
                pystoi_right = stoi(re_target, re_spin, spin_sr, extended=False)

                print(f"PySTOI Score (Left Ear): {pystoi_left}")
                print(f"PySTOI Score (Right Ear): {pystoi_right}")

                # Add a check to see if the code continues past the STOI calculation
                print("Continuing to result preparation...")
                
                
                # Save the STOI results to a JSON file
                result = {
                    'left_ear_stoi': stoi_score_le,
                    'right_ear_stoi': stoi_score_re
                }

                result_path = Path(path_data.test_path.stoi_output_file)
                with open(result_path, 'w') as result_file:
                    json.dump(result, result_file, indent=4)

                print(f"STOI results saved to {result_path}")

            except Exception as e:
                print(f"Error during STOI computation: {e}")
                return

    except FileNotFoundError:
        print(f'JSON file not found: {path_data.test_path.ref_file}')
    except Exception as e:
        print(f"Error: {e}")

# Run the script
if __name__ == "__main__":
    main()
