import os

from midi2audio import FluidSynth

"""
Class responsible for generating sound files from MIDI files using specified Sound Font
"""


class SoundGenerator:

    @classmethod
    def generate_samples(cls, midi_dir, output_dir, sound_font):
        files = os.listdir(midi_dir)
        fs = FluidSynth(sound_font=sound_font)
        for f in files:
            fs.midi_to_audio(os.path.join(midi_dir, f),
                             os.path.join(output_dir, f"{f[:-4]}.wav"))
