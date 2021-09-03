import os
import pickle
from itertools import combinations

import numpy as np
from pydub import AudioSegment
from scipy.fftpack import fft

from Quantizator import Quantizator


class PolyphonicData:
    """
    Method responsible for generating polyphonic combinations of notes
    with certain number of elements defined by "max_notes_num"
    from specified directory containing monophonic sounds
    """

    @staticmethod
    def generate_audio_segments(input_dir, max_notes_num, audio_length):
        audiofiles = os.listdir(input_dir)
        sounds_dict = {
            f[:-4]: AudioSegment.from_wav(f"{input_dir}{f}").set_channels(1) for f in
            audiofiles}
        for length in range(1, max_notes_num + 1):
            for i, combination in enumerate(combinations(sounds_dict.items(), length)):
                combination_sound = AudioSegment.silent(
                    duration=audio_length * 1000)
                for _, audio_segment in combination:
                    combination_sound = combination_sound.overlay(
                        audio_segment)

                yield (frozenset(str(pitch) for pitch, _ in combination),
                       combination_sound)


class PolyphonicAudiofiles(PolyphonicData):
    """
    Method generating and saving polyphonic audio files using functionalities of
    "PolyphonicData" class
    """

    @classmethod
    def generate_audiofiles(cls, input_dir, audiofiles_dir, max_notes_num, audio_length):
        for notes, combination_sound in cls.generate_audio_segments(input_dir,
                                                                    max_notes_num,
                                                                    audio_length):
            combination_sound.export(
                os.path.join(audiofiles_dir + "_".join(
                    str(note) for note in sorted(notes)) + ".wav"), format="wav")


class PolyphonicDictionary(PolyphonicData, Quantizator):
    """
    Method calculating arguments needed for "Quantizator" class to provide the data saved
    in the dictionary
    """

    @classmethod
    def get_audiosegment_fft_ranges_sums(cls, audiosegment, spectrum_start, spectrum_end,
                                         audio_length):
        samples = np.array(audiosegment.get_array_of_samples())
        fourier_transform = fft(samples)
        frequency_coefficient = audio_length
        note_frequencies = cls.get_fft_ranges_sums(
            fourier_transform, spectrum_start, spectrum_end, frequency_coefficient)
        return note_frequencies

    """
    Method generating dictionary containing polyphonic sounds data
    and dumping it into one file with the "pickle" library,
    """

    @classmethod
    def generate_dictionary(cls, input_dir, quantized_dict_path, max_notes_num,
                            audio_length, spectrum_start, spectrum_end):
        quantized_dict = dict()
        for notes, combination_sound in cls.generate_audio_segments(input_dir,
                                                                    max_notes_num,
                                                                    audio_length):
            quantized_dict[notes] = \
                cls.get_audiosegment_fft_ranges_sums(
                    combination_sound, spectrum_start, spectrum_end, audio_length)

        with open(f"{quantized_dict_path}", "wb") as output_file:
            pickle.dump(quantized_dict, output_file)
