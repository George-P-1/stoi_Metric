# Import libraries
import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import soundfile as sf
from scipy.signal import resample
import sounddevice as sd
import pystoi
import matplotlib.pyplot as plt

import plotter
import mystoi


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
            
            # Path of audio files to open, HA_Output and target_anechoic
            spin_file_path = Path(path_data.test_path.spin_folder) / f"{ref_json[scene_index]['signal']}.wav"
            target_file_path = Path(path_data.test_path.scenes_folder) / f"{ref_json[scene_index]['scene']}_target_anechoic.wav"
            # Opening audio files using soundfile
            spin, spin_sr = sf.read(spin_file_path)
            target, target_sr = sf.read(target_file_path)
            # Resampling
            new_sr = path_data.sample_rate
            # REVIEW - Can use scipy (resample or decimate functions) or librosa library. Some issue with scipy using only frequency domain or something like that.
            spin_resampled = resample(spin, int(len(spin) * new_sr / spin_sr))   # NOTE - current_no_of_samples / current_sampling_rate is the duration of audio signal
            target_resampled = resample(target, int(len(target) * new_sr / target_sr))
            
            if True:    # REMOVE_LATER
                print("Total number of scenes in JSON file:", len(ref_json))    # REMOVE_LATER
                print("Info about Current SPIN File:\n", ref_json[scene_index], "\n")    # REMOVE_LATER
                print("Paths to current audio files:\n", spin_file_path, "\n", target_file_path, "\n")  # REMOVE_LATER
                print(f"Number of samples of spin audio is {len(spin)} and sample rate is {ref_json[scene_index]['signal']} is {spin_sr} and shape is {spin.shape}.")   # REMOVE_LATER
                print(f"Number of samples of target audio is {len(target)} and sample rate is {ref_json[scene_index]['scene']} is {target_sr} and shape is {target.shape}.")   # REMOVE_LATER
                print(f"Number of samples of new spin audio is {len(spin_resampled)} and sample rate is {new_sr} and shape is {spin_resampled.shape}.")  # REMOVE_LATER
                print(f"Number of samples of new target audio is {len(target_resampled)} and sample rate is {new_sr} and shape is {target_resampled.shape}.")  # REMOVE_LATER
            

            # NOTE - Convert stereo to mono [Instead of computing stoi of each ear separately]
            if len(spin.shape) == 2:
                spin_mono = spin.mean(axis=1)
                spin_resampled_mono = spin_resampled.mean(axis=1)
            else:
                raise Exception("SPIN audio is not stereo.")
            if len(target.shape) == 2:
                target_mono = target.mean(axis=1)
                target_resampled_mono = target_resampled.mean(axis=1)
            else:
                raise Exception("Target audio is not stereo.")

            # NOTE - Plots
            if False:    # REMOVE_LATER
                # Spectrograms
                plt.figure("Spectrograms (Mono Signals)")
                # Plot before resampling
                plt.subplot(2,2,1)
                plotter.plot_spectrogram(spin_mono, spin_sr, 'Spectrogram of SPIN (before resampling)')
                plt.subplot(2,2,2)
                plotter.plot_spectrogram(target_mono, target_sr, 'Spectrogram of Target (before resampling)')
                # Plot after resampling
                plt.subplot(2,2,3)
                plotter.plot_spectrogram(spin_resampled_mono, new_sr, 'Spectrogram of SPIN (after resampling)')
                plt.subplot(2,2,4)
                plotter.plot_spectrogram(target_resampled_mono, new_sr, 'Spectrogram of Target (after resampling)')
                plt.tight_layout()  # Adjust spacing

                # Signals in amplitude-time
                plt.figure("Amplitude-Time (Mono Signals)")
                # Plot before resampling
                plt.subplot(2,2,1)
                plotter.plot_regular(spin_mono, len(spin_mono), spin_sr, 'SPIN (before resampling)')
                plt.subplot(2,2,2)
                plotter.plot_regular(target_mono, len(target_mono), target_sr, 'Target (before resampling)')
                # Plot after resampling
                plt.subplot(2,2,3)
                plotter.plot_regular(spin_resampled_mono, len(spin_resampled_mono), new_sr, 'SPIN (after resampling)')
                plt.subplot(2,2,4)
                plotter.plot_regular(target_resampled_mono, len(target_resampled_mono), new_sr, 'Target (after resampling)')
                plt.tight_layout()  # Adjust spacing

                plt.show()

            # NOTE - Play audio files (before and after resampling)
            if False:     # REMOVE_LATER - Change Boolean to turn on/off playing files. 
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
                print("Playing SPIN and target audio files after converting to mono...\n")
                sd.play(spin_resampled_mono, new_sr)
                sd.wait()
                sd.play(target_resampled_mono, new_sr)
                sd.wait()

            # TODO - Remove first 2 seconds from both signals cuz of silence in the clean signal


            # SECTION - Compute STOI Metric using pystoi
            # Directly using pystoi library to see how stoi monotonic output looks like.
            # REVIEW - Not sure which channel is right ear and which one is left ear
            stoi_val = pystoi.stoi(target_resampled[:,0], spin_resampled[:,0], new_sr)
            print(f"STOI value for intelligibility of Right Ear SPIN signal {ref_json[scene_index]['signal']} is {stoi_val}.")
            stoi_val = pystoi.stoi(target_resampled[:,1], spin_resampled[:,1], new_sr)
            print(f"STOI value for intelligibility of Left Ear SPIN signal {ref_json[scene_index]['signal']} is {stoi_val}.")
            stoi_val = pystoi.stoi(target_resampled_mono, spin_resampled_mono, new_sr)
            print(f"STOI value for intelligibility of Mono SPIN signal {ref_json[scene_index]['signal']} is {stoi_val}.")

            # SECTION - Implement STOI Metric
            print("----------mystoi values below----------\n")
            
            stoi_val = mystoi.compute_stoi(target_resampled[:,0], spin_resampled[:,0], new_sr)
            print(f"STOI value for intelligibility of Right Ear SPIN signal {ref_json[scene_index]['signal']} is {stoi_val}.")
            stoi_val = mystoi.compute_stoi(target_resampled[:,1], spin_resampled[:,1], new_sr)
            print(f"STOI value for intelligibility of Left Ear SPIN signal {ref_json[scene_index]['signal']} is {stoi_val}.")
            stoi_val = mystoi.compute_stoi(target_resampled_mono, spin_resampled_mono, new_sr)
            print(f"STOI value for intelligibility of Mono SPIN signal {ref_json[scene_index]['signal']} is {stoi_val}.")


        ref_file.close()
    except FileNotFoundError:
        raise Exception(f'JSON file not found: {path_data.test_path.ref_file}')
    finally:
        print(f'Finished processing JSON file.')


if __name__ == '__main__':
    main()