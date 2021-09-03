import argparse
import os

from Loaders import AudioLoader, DictionaryLoader
from MidiGenerator import MidiGenerator
from NeuralNetwork import NeuralNetwork
from PathHandler import PathHandler
from Polyphonics import PolyphonicAudiofiles, PolyphonicDictionary
from SoundGenerator import SoundGenerator

"""
Directories and files to create in the working directory
"""

midi_dir = "midi/"
samples_dir = "samples/"
audiofiles_dir = "audiofiles/"
quantized_dict_dir = "dictionary.data"
network_weights_dir = "network_weights.data"
guessing_dir = "guess_these_notes2/"
sound_font = "FluidR3_GM.sf2"

"""
CLI setup with argparse
"""

parser = argparse.ArgumentParser(description="Program preparing data and training the "
                                             "neural network recognizing notes. It is "
                                             "necessary to specify 10 parameters listed "
                                             "below. After specifying the variables, "
                                             "you can type instructions to execute. "
                                             "It is "
                                             "possible to generate the data first and "
                                             "train or test the network in several "
                                             "consequent usages, for example use "
                                             "--initial and --generate-dictionary in "
                                             "first usage, then --train-with-dictionary "
                                             "in consequent iteration, "
                                             "then --generate-audiofiles and "
                                             "--test-with-audiofiles in the last usage. "
                                             "If the data is already prepared, "
                                             "it is advised to only change the "
                                             "parameters regarding the neural network, "
                                             "such as number of iterations, learning "
                                             "rate and network sensitivity.")

parser.add_argument("-wd", "--working-directory",
                    type=str,
                    help="Directory to store all generated files: MIDI, samples, "
                         "dictionaries and polyphonic audiofiles")
parser.add_argument("-ss", "--spectrum-start",
                    type=int,
                    help="Start of range for which spectrum frequencies will be "
                         "analyzed, calculated from frequency of given MIDI number")
parser.add_argument("-se", "--spectrum-end",
                    type=int,
                    help="End of range for which spectrum frequencies will be analyzed, "
                         "calculated from frequency of given MIDI number")
parser.add_argument("-ms", "--midi-start",
                    type=int,
                    help="Start of range for which MIDI files will be made, and then "
                         "used for polyphonic sound generation")
parser.add_argument("-me", "--midi-end",
                    type=int,
                    help="End of range for which MIDI files will be made, and then used "
                         "for polyphonic sound generation")
parser.add_argument("-mnn", "--max-notes-num",
                    type=int,
                    help="Maximum number of notes in polyphonic combinations")
parser.add_argument("-al", "--audio-length",
                    type=int,
                    help="Length of generated samples and polyphonic sounds")
parser.add_argument("-in", "--iterations-number",
                    type=int,
                    help="Number of neural network training iterations")
parser.add_argument("-ns", "--network-sensitivity",
                    type=float,
                    help="How sensible the network is ranging from 0.0 to 1.0, "
                         "too small value will result in many false-positive errors; "
                         "too big value will result in many false-negative errors")
parser.add_argument("-lr", "--learning-rate",
                    type=float,
                    help="Learning rate of neural network")
parser.add_argument("-cpu", "--use-cpu",
                    help="Use CPU instead of GPU during neural network computations",
                    action="store_true")

parser.add_argument("-i", "--initialize",
                    help="Generate MIDI files and samples with monophonic sounds",
                    action="store_true")
parser.add_argument("-ga", "--generate-audiofiles",
                    type=str,
                    default=False,
                    nargs='?',
                    help="Generate polyphonic audiofiles using monophonic samples",
                    const=audiofiles_dir)
parser.add_argument("-gd", "--generate-dictionary",
                    action="store_true",
                    help="Generate dictionary containing polyphonic data using monophonic samples, "
                         "skipping the process of saving audio files to ROM (faster method)")
parser.add_argument("-train-wd", "--train-with-dictionary",
                    action="store_true",
                    help="Train the neural network model using data in the dictionary")
parser.add_argument("-train-wa", "--train-with-audiofiles",
                    action="store_true",
                    help="Train the neural network model using data directly from "
                         "generated polyphonic audiofiles")
parser.add_argument("-test-wd", "--test-with-dictionary",
                    action="store_true",
                    help="Test the neural network model using data in the dictionary")
parser.add_argument("-test-wa", "--test-with-audiofiles",
                    type=str,
                    default=False,
                    nargs='?',
                    help="Test the neural network model using data directly from "
                         "generated audiofiles; optionally specify custom directory to "
                         "choose audiofiles from - filenames in directory have to "
                         "contain MIDI numbers of monophonic sounds in the file joined "
                         "with \"_\" sign (path relative to working directory)",
                    const=audiofiles_dir)
parser.add_argument("--guess",
                    type=str,
                    help="Use trained model to guess MIDI numbers of notes in audiofiles in a specific "
                         "directory (path relative to working directory)")

args = parser.parse_args()
spectrum_range = args.spectrum_end - args.spectrum_start
midi_range = args.midi_end - args.midi_start
NeuralNetwork.load_variables(args.learning_rate, args.iterations_number,
                             args.network_sensitivity, args.use_cpu)

midi_path = os.path.join(args.working_directory, midi_dir)
samples_path = os.path.join(args.working_directory, samples_dir)
audiofiles_path = os.path.join(args.working_directory, audiofiles_dir)
quantized_dict_path = os.path.join(args.working_directory, quantized_dict_dir)
network_weights_path = os.path.join(args.working_directory, network_weights_dir)

"""
Executing commands given as parameters to argument parser
"""

if args.initialize:
    PathHandler.create_dir(midi_path)
    MidiGenerator.generate_midi(midi_path, args.midi_start, args.midi_end,
                                args.audio_length)
    PathHandler.create_dir(samples_path)
    SoundGenerator.generate_samples(midi_path, samples_path, sound_font)
    print("MIDI files and samples have been created")

if args.generate_audiofiles:
    print("Creating and saving wav files with polyphonic sound...")
    PathHandler.create_dir(audiofiles_path)
    PolyphonicAudiofiles.generate_audiofiles(samples_path,
                                             audiofiles_path,
                                             args.max_notes_num,
                                             args.audio_length)
    print("Files have been saved")
if args.generate_dictionary:
    print("Creating and saving quantized data...")
    PathHandler.prepare_for_file(quantized_dict_path)
    PolyphonicDictionary.generate_dictionary(samples_path,
                                             quantized_dict_path,
                                             args.max_notes_num,
                                             args.audio_length,
                                             args.spectrum_start,
                                             args.spectrum_end)
    print("File has been saved")
if args.train_with_audiofiles or args.train_with_dictionary:
    print("Training network...")
    if args.train_with_dictionary:
        x, y = DictionaryLoader.load_dictionary(
            quantized_dict_path, args.midi_start, midi_range)
    elif args.train_with_audiofiles:
        x, y = AudioLoader.load_audiofiles(
            audiofiles_path, args.midi_start, midi_range, args.spectrum_start,
            args.spectrum_end)

    PathHandler.prepare_for_file(network_weights_path)

    NeuralNetwork.train_and_save_weights(
        x, y, midi_range, spectrum_range, network_weights_path)
    print("Network has been trained, weights are saved")

if args.test_with_audiofiles or args.test_with_dictionary:
    print("Testing network...")
    if args.test_with_dictionary:
        x, y = DictionaryLoader.load_dictionary(
            quantized_dict_path, args.midi_start, midi_range)
    elif args.test_with_audiofiles:
        test_path = os.path.join(args.working_directory, args.test_with_audiofiles)
        x, y = AudioLoader.load_audiofiles(
            test_path, args.midi_start, midi_range, args.spectrum_start,
            args.spectrum_end)
    matrix, false_neg, false_pos = NeuralNetwork.load_weights_and_test(x, y,
                                                                       network_weights_path)
    print(f"Number of false positives: {false_pos}, number of false negatives: {false_neg}")

if args.guess:
    print(f"Guessing from directory {args.guess}")
    guessing_path = os.path.join(args.working_directory, args.guess)
    x = AudioLoader.load_audiofiles_to_guess(
        guessing_path, args.spectrum_start, args.spectrum_end)
    notes_in_files = NeuralNetwork.load_weights_and_guess(x,
                                                          network_weights_path,
                                                          guessing_path,
                                                          args.midi_start)
    print("\n".join(notes_in_files))
