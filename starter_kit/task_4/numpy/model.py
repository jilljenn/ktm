import numpy as np 
import pandas as pd
import os
from pathlib import Path


PATH = Path('../../../data/neurips')

class MyModel:
    def __init__(self):
        """
        Simple example baseline model, which will always query the most answered question available during question 
        selection, and will always predict the most common answer for each question when a student hasn't answered it.
        """
        self.most_popular_answers = None
        self.num_answers = None

    def train_model(self):
        """
        Train a model.
        """
        # # For local evaluation
        # data_path = os.path.normpath('../../data/test_input/test_train_task_4.csv')

        # For full training
        data_path = PATH / 'train_data/train_task_3_4.csv' # for training

        df = pd.read_csv(data_path)
        data_array = df.pivot(index='UserId', columns='QuestionId', values='IsCorrect').to_numpy()
        observation_mask = 1 - np.isnan(data_array).astype(int)
        self.num_answers = observation_mask.sum(axis=0)

        self.most_popular_answers = []

        # Get most popular answer (right or wrong) for each question, to use for predictions.
        for column in data_array.T:
            col_answers = column[np.isnan(column)==False].astype(int)
            most_popular = np.argmax(np.bincount(col_answers))
            self.most_popular_answers.append(most_popular)

        np.save('model_task_4_most_popular.npy', self.most_popular_answers)
        np.save('model_task_4_num_answers.npy', self.num_answers)

    def load(self, most_popular_path, num_answers_path):
        """
        Load a model's state, by loading arrays containing the most popular answers.
        Args:
            most_popular_path (string or pathlike): Path to array containing the most popular answer to each question.
            num_answers_path (string or pathlike): Path to array containing the number of recorded answers to each 
                question.
        """
        self.most_popular_answers = np.load(most_popular_path)
        self.num_answers = np.load(num_answers_path)

    def select_feature(self, masked_data, can_query):
        """
        Select most popular feature from those available.
        """
        masked_num_answers = can_query * self.num_answers
        selections = np.argmax(masked_num_answers, axis=1)
        return selections

    def predict(self, masked_data, masked_binary_data):
        """
        Produce binary predictions.
        """
        predictions = np.repeat(self.most_popular_answers.reshape(1,-1), repeats=masked_data.shape[0], axis=0)
        return predictions


