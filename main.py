# Import libraries
import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import soundfile as sf
from scipy.signal import resample
import sounddevice as sd
import pystoi


# SECTION Main code here

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(path_data: DictConfig) -> None:
    # print(path_data.test_path.ref_file) # REMOVE_LATER
    # Open reference JSON file
    try:
        with open(path_data.test_path.ref_file, 'r') as ref_file:
            ref_json = json.load(ref_file)
            
            # SECTION - Load and resample SPIN and clean audio 
            # TODO - Make a automated way to open each spin and target anechoic scene using loop. Or maybe this script just needs to work on one prompt on each run.
            scene_index = 4
            
            print("Info about Current SPIN File:\n", ref_json[scene_index], "\n")    # REMOVE_LATER
            # Path of audio files to open, HA_Output and target_anechoic
            spin_file_path = Path(path_data.test_path.spin_folder) / f"{ref_json[scene_index]['signal']}.wav"
            target_file_path = Path(path_data.test_path.scenes_folder) / f"{ref_json[scene_index]['scene']}_target_anechoic.wav"
            print("Paths to current audio files:\n", spin_file_path, "\n", target_file_path, "\n")  # REMOVE_LATER
            # Opening audio files using soundfile
            spin, spin_sr = sf.read(spin_file_path)
            target, target_sr = sf.read(target_file_path)
            # Resampling
            new_sr = path_data.sample_rate
            # REVIEW - Can use scipy (resample or decimate functions) or librosa library. Some issue with scipy using only frequency domain or something like that.
            spin_resampled = resample(spin, int(len(spin) * new_sr / spin_sr))   # NOTE - current_no_of_samples / current_sampling_rate is the duration of audio signal
            target_resampled = resample(target, int(len(target) * new_sr / target_sr))
            print(f"Number of samples of spin audio is {len(spin)} and sample rate is {ref_json[scene_index]['signal']} is {spin_sr} and shape is {spin.shape}.")   # REMOVE_LATER
            print(f"Number of samples of target audio is {len(target)} and sample rate is {ref_json[scene_index]['scene']} is {target_sr} and shape is {target.shape}.")   # REMOVE_LATER
            print(f"Number of samples of new spin audio is {len(spin_resampled)} and sample rate is {new_sr} and shape is {spin_resampled.shape}.")  # REMOVE_LATER
            print(f"Number of samples of new target audio is {len(target_resampled)} and sample rate is {new_sr} and shape is {target_resampled.shape}.")  # REMOVE_LATER
            
            # NOTE - Play audio files (before and after resampling)
            # Before
            print("Playing SPIN and target audio files before resampling...")
            sd.play(spin, spin_sr)
            sd.wait()
            sd.play(target, target_sr)
            sd.wait()
            # After
            print("Playing SPIN and target audio files after resampling...")
            sd.play(spin_resampled, new_sr)
            sd.wait()   
            sd.play(target_resampled, new_sr)
            sd.wait()

            # SECTION - Implement STOI Metric

            # NOTE - Convert stereo to mono
            if len(spin_resampled.shape) == 2:
                spin_resampled = spin_resampled.mean(axis=1)
            else:
                raise Exception("SPIN audio is not stereo.")
            if len(target_resampled.shape) == 2:
                target_resampled = target_resampled.mean(axis=1)
            else:
                raise Exception("Target audio is not stereo.")

            print("Playing SPIN and target audio files after converting to mono...")
            sd.play(spin_resampled, new_sr)
            sd.wait()
            sd.play(target_resampled, new_sr)
            sd.wait()

            # NOTE - Directly using pystoi library to see how stoi monotonic output looks like.
            stoi_val = pystoi.stoi(target_resampled, spin_resampled, new_sr)
            print(f"STOI value between target and spin audio is {stoi_val}.")


        ref_file.close()
    except FileNotFoundError:
        raise Exception(f'JSON file not found: {path_data.test_path.ref_file}')
    finally:
        print(f'Finished processing JSON file.')



# !SECTION

if __name__ == '__main__':
    main()