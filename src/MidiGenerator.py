import os

from midiutil import MIDIFile

"""
Class responsible for generating MIDI files containing one note each
to specified directory
"""


class MidiGenerator:

    @classmethod
    def generate_midi(cls, directory, midi_start, midi_end, audio_length):
        track = 0
        channel = 0
        time = 0  # In beats
        duration = audio_length  # In beats
        tempo = 60  # In BPM
        volume = 100  # 0-127, as per the MIDI standard

        notes = [x for x in range(midi_start, midi_end)]

        for note in notes:
            midiFile = MIDIFile(numTracks=1)
            midiFile.addTempo(track, time, tempo)
            midiFile.addNote(track, channel, note, time, duration, volume)

            with open(os.path.join(directory, f"{note}.mid"), "wb") as output_file:
                midiFile.writeFile(output_file)
