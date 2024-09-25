# Import libraries
import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import soundfile as sf
from scipy.signal import resample


# SECTION Main code here

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(path_data: DictConfig) -> None:
    # print(path_data.test_path.ref_file) # REMOVE_LATER
    # NOTE - Open reference JSON file
    try:
        with open(path_data.test_path.ref_file, 'r') as ref_file:
            ref_json = json.load(ref_file)
            
            # TODO - Make a automated way to open each spin and target anechoic scene using loop. Or maybe this script just needs to work on one prompt on each run.
            print(ref_json[0], "\n")    # REMOVE_LATER
            # NOTE - Path of audio files to open, HA_Output and target_anechoic
            spin_file_path = Path(path_data.test_path.spin_folder) / f"{ref_json[0]['signal']}.wav"
            print(spin_file_path, "\n")  # REMOVE_LATER
            target_file_path = Path(path_data.test_path.scenes_folder) / f"{ref_json[0]['scene']}_target_anechoic.wav"
            print(target_file_path, "\n")  # REMOVE_LATER
            # TODO - NOTE - Opening audio files using soundfile
            spin, spin_sr = sf.read(spin_file_path)
            print(f"Sample rate of spin audio {ref_json[0]['signal']} is {spin_sr}.")
            # NOTE - Resampling 
            # REVIEW - Can use scipy (resample or decimate functions) or librosa library. Some issue with scipy using only frequency domain or something like that.
            spin_resampled = resample(spin, int(len(spin) * path_data.sample_rate / spin_sr))   # NOTE - current_no_of_samples / current_sampling_rate is the duration of audio signal
            print(f"Number of samples of new spin audio is {len(spin_resampled)} and sample rate is {path_data.sample_rate}.")
            # TODO - Plot the original and resampled audio using matplotlib

        ref_file.close()
    except FileNotFoundError:
        print(f'File not found: {path_data.test_path.ref_file}')
        return None
    finally:
        print(f'Finished processing JSON file.')



# !SECTION

if __name__ == '__main__':
    main()