import os
import pickle
from time import time

import numpy as np
import torch


class NeuralNetwork:
    device = None
    network_sensitivity = None
    CONST_DIVISOR = 1000000000000000000.0
    dtype = torch.float

    """
    Method classifying occurrence of notes by converting floating point data
    into zeros and ones basing on the "network_sensitivity" variable
    """

    @classmethod
    def classify_results(cls, y_pred):
        return np.where(y_pred.cpu().detach().numpy() > cls.network_sensitivity, 1, 0)

    """
    Method generating input data as a Tensor object and scaling it properly
    """

    @classmethod
    def generate_x_tensor(cls, x):
        x = torch.tensor(x, device=cls.device, dtype=cls.dtype)
        divide_coeff = cls.CONST_DIVISOR * np.max(np.array(x.cpu()))
        x = x / divide_coeff
        return x

    """
    Method generating output data as a Tensor object
    """

    @classmethod
    def generate_y_tensor(cls, y):
        y = torch.tensor(y, device=cls.device, dtype=cls.dtype)
        return y

    @classmethod
    def generate_tensors(cls, x, y):
        x = cls.generate_x_tensor(x)
        y = cls.generate_y_tensor(y)
        return x, y

    """
    Method calculating the output layer elements using input layer and hidden layer Tensors
    """

    @classmethod
    def calculate_predicted_y(cls, x, w1, w2):
        return x.mm(w1).clamp(min=0).mm(w2)

    """
    Method computating Mean Square Error between expected and calculated output
    """

    @classmethod
    def calculate_loss(cls, y_pred, y):
        return (y_pred - y).pow(2).sum()

    """
    Method training the neural network using backpropagation
    in a specified number of iterations
    """

    @classmethod
    def train_network(cls, x, y, midi_range, spectrum_range):

        input_layer_size, hidden_layer_size, output_layer_size = \
            spectrum_range, spectrum_range, midi_range
        x, y = cls.generate_tensors(x, y)
        # Randomly initialize weights
        w1 = torch.randn(input_layer_size, hidden_layer_size,
                         device=cls.device, dtype=cls.dtype, requires_grad=True)
        w2 = torch.randn(hidden_layer_size, output_layer_size,
                         device=cls.device, dtype=cls.dtype, requires_grad=True)

        start_time = time()
        for iteration in range(cls.iterations_number):
            y_pred = cls.calculate_predicted_y(x, w1, w2)
            loss = cls.calculate_loss(y_pred, y)
            if iteration % 100 == 99:
                print(f"Iteration {iteration + 1}:     "
                      f"MSE: {loss.item() / len(x)}, Time in training: {time() - start_time}")
            loss.backward()
            with torch.no_grad():
                w1 -= cls.learning_rate * w1.grad
                w2 -= cls.learning_rate * w2.grad

                w1.grad.zero_()
                w2.grad.zero_()

        return w1, w2

    """
    Method calculating the output layer elements and returning the data,
    including false-positive and false-negative occurrences
    """

    @classmethod
    def test_network(cls, x, y, w1, w2):

        x, y = cls.generate_tensors(x, y)
        y_pred = cls.calculate_predicted_y(x, w1, w2)
        final_y = cls.classify_results(y_pred)
        ideal_y = y.cpu().detach().numpy()
        false_negative = np.sum(np.where(final_y < ideal_y, 1, 0))
        false_positive = np.sum(np.where(final_y > ideal_y, 1, 0))

        return final_y, false_negative, false_positive

    """
    Method calculating the output layer elements and formatting the data
    to present occurred sounds in their MIDI numbers
    """

    @classmethod
    def guess_notes(cls, x, w1, w2, midi_start):

        x = cls.generate_x_tensor(x)
        y_pred = cls.calculate_predicted_y(x, w1, w2)
        results = list()
        for notes in y_pred:
            guessed_notes, = np.where(
                notes.cpu().detach().numpy() > cls.network_sensitivity)
            guessed_notes += midi_start
            results.append(guessed_notes)
        return results

    """
    Method wrapping the "train_network" method to dump the network weights data to a file
    """

    @classmethod
    def train_and_save_weights(cls, x, y, midi_range, spectrum_range, weights_path):
        w1, w2 = cls.train_network(x, y, midi_range, spectrum_range)
        with open(weights_path, "wb") as weights_file:
            pickle.dump((w1, w2), weights_file)

    """
    Method wrapping the "test_network" method with loading network weights from a file
    """

    @classmethod
    def load_weights_and_test(cls, x, y, weights_path):
        with open(weights_path, "rb") as weights_file:
            w1, w2 = pickle.load(weights_file)
        return cls.test_network(x, y, w1, w2)

    """
    Method wrapping the "guess_notes" method with loading network weights from a file
    """

    @classmethod
    def load_weights_and_guess(cls, x, weights_path, guessing_path, midi_start):
        with open(weights_path, "rb") as weights_file:
            w1, w2 = pickle.load(weights_file)
        midi_notes_list = cls.guess_notes(x, w1, w2, midi_start)

        files = sorted(os.listdir(guessing_path))
        notes_in_files = list()
        for file, notes in zip(files, midi_notes_list):
            notes_str = " ".join(str(note) for note in notes)
            formatted_str = f"Notes in {file}: {notes_str}"
            notes_in_files.append(formatted_str)
        return notes_in_files

    """
    Method loading needed variables from CLI input
    """

    @classmethod
    def load_variables(cls, learning_rate, iterations_number, network_sensitivity,
                       use_cpu):
        cls.learning_rate = learning_rate
        cls.iterations_number = iterations_number
        cls.network_sensitivity = network_sensitivity
        if use_cpu:
            cls.device = torch.device("cpu:0")
        else:
            cls.device = torch.device("cuda:0")
