import os
import pickle

from scipy.fftpack import fft
from scipy.io import wavfile

from Quantizator import Quantizator


class Loader:
    """
    Method generating array of expected results from neural network
    basing on MIDI numbers of notes
    """

    @staticmethod
    def generate_results(notes, midi_start, midi_range):
        results = [0.0] * midi_range
        for note in notes:
            index = int(note) - midi_start
            results[index] = 1.0
        return results

    """
    Method loading and generating input and output data for neural network
    """

    @classmethod
    def load(cls, quantized_dict, midi_start, midi_range):

        x, y = list(), list()
        for notes, freqs in quantized_dict.items():
            x.append(freqs)
            results = cls.generate_results(notes, midi_start, midi_range)
            y.append(results)

        return x, y


class AudioLoader(Quantizator, Loader):
    """
    Method calculating arguments needed for "Quantizator" class to provide the data needed
    to generate input and output data for the neural network from an audio file
    """

    @classmethod
    def get_audiofile_fft_ranges_sums(cls, notes, dir, spectrum_start, spectrum_end):
        sample_rate, samples = wavfile.read(os.path.join(dir, notes))
        fourier_transform = fft(samples)
        frequency_coefficient = len(samples) / sample_rate
        note_frequencies = cls.get_fft_ranges_sums(
            fourier_transform, spectrum_start, spectrum_end, frequency_coefficient)
        return note_frequencies

    """
    Method creating dictionary from audio files and using it to create 
    the neural network input and output data
    """

    @classmethod
    def load_audiofiles(cls, dir, midi_start, midi_range, spectrum_start, spectrum_end):
        notes_list = sorted(os.listdir(dir))
        quantized_dict = \
            {
                frozenset(note for note in notes[:-4].split("_")):
                    cls.get_audiofile_fft_ranges_sums(
                        notes, dir, spectrum_start, spectrum_end)
                for notes in notes_list
            }
        x, y = cls.load(quantized_dict, midi_start, midi_range)
        return x, y

    """
    Method creating dictionary from audio files and using it to create 
    the neural network input data and skipping the output data
    """

    @classmethod
    def load_audiofiles_to_guess(cls, dir, spectrum_start, spectrum_end):
        files_list = sorted(os.listdir(dir))
        x = list()
        for notes in files_list:
            note_freqs = cls.get_audiofile_fft_ranges_sums(
                notes, dir, spectrum_start, spectrum_end)
            x.append(note_freqs)
        return x


class DictionaryLoader(Loader):
    """
    Method loading dictionary containing processed data from a file using the "pickle"
    library and generating the neural network input and output data
    """

    @classmethod
    def load_dictionary(cls, dict_path, midi_start, midi_range):
        with open(dict_path, "rb") as input_file:
            quantized_dict = pickle.load(input_file)
        x, y = cls.load(quantized_dict, midi_start, midi_range)
        return x, y
