import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from time import time
from pathlib import Path
import pandas as pd
import os


PATH = Path('../../../data/neurips')


def df_to_sparse_inputs(df_data, user_to_idx):
    """ Generates inputs to build X as a sparse matrix """
    
    nb_rows = len(df_data)
    nb_users = len(user_to_idx)
    len_df_data = len(df_data)
    indexes = np.arange(len_df_data).tolist()
    col_users = df_data["UserId"].apply(lambda x: user_to_idx[x]).to_list()
    col_questions = [q + nb_users for q in df_data["QuestionId"].to_list()]

    row = 2 * indexes 
    col = col_users + col_questions
    data = 2 * nb_rows * [1]

    return row, col, data 


    """ Transform data into into/output """
def transform_df_data(df_data):

    user_ids = df_data["UserId"].unique()
    user_to_idx = {u_id: i for i, u_id in enumerate(user_ids)}
    row, col, data = df_to_sparse_inputs(df_data, user_to_idx)
    X = csr_matrix((data, (row, col)), dtype=np.int8)
    y = np.array(df_data["IsCorrect"])

    return X, y, user_to_idx


class Submission:
    """
    API Wrapper class which loads a saved model upon construction, and uses this to implement an API for feature 
    selection and missing value prediction. This API will be used to perform active learning evaluation in private.

    Note that the model's final predictions must be binary, but that both categorical and binary data is available to
    the model for feature selection and making predictions.
    """
    def __init__(self, data_path="train_task_3_4.csv"):
        # For local evaluation
        data_path = PATH / 'test_input/train_task_4.csv'
        # self.model = pickle.load(open("lr.pkl", "rb"))
        self.df_data = pd.read_csv(data_path)[["QuestionId", "UserId", "IsCorrect"]]
        self.offset = self.df_data["UserId"].max() + 1

        X, y, self.user_to_idx = transform_df_data(self.df_data)
        self.lr = LogisticRegression(solver="liblinear")
        self.lr.fit(X, y, sample_weight=None)
        self.is_first_round = True
        self.selections = None

    def select_feature(self, masked_data, masked_binary_data, can_query):
        """
        Use your model to select a new feature to observe from a list of candidate features for each student in the
            input data, with the goal of selecting features with maximise performance on a held-out set of answers for
            each student.
        Args:
            masked_data (np.array): Array of shape (num_students, num_questions) containing data revealed to the model
                at the current step. Unobserved values are denoted by -1.
            masked_binary_data (np.array): Array of shape (num_students, num_questions) containing binary data revealed 
                to the model at the current step. Unobserved values are denoted by -1.
            can_query (np.array): Binary array of shape (num_students, num_questions), indicating which features can be 
                queried by the model in the current step.
        Returns:
            selections (list): List of ints, length num_students, containing the index of the feature selected to query 
            for each student (row) in the dataset.
        """
        # Use the loaded model to perform feature selection.
        nb_users = len(self.user_to_idx)
        nb_unseen_users = masked_data.shape[0]

        if self.is_first_round is True:
            coeffs_users = self.lr.coef_[:, :nb_users].squeeze()
            coeffs_questions = self.lr.coef_[:, nb_users:].squeeze()
            nb_questions = len(coeffs_questions)
            idx_user_median = np.argsort(coeffs_users)[len(coeffs_users)//2]
            selections = []

            inputs = []
            for i in range(nb_questions):
                row = [0] * self.lr.coef_.shape[1]
                row[idx_user_median] = 1
                row[nb_users + i] = 1
                inputs.append(row)
            inputs = np.array(inputs)
            probas = self.lr.predict_proba(inputs)[:, 1]
            
            for idx_user in range(nb_unseen_users):
                probas_ = probas.copy()
                idx_mask = np.where(can_query[idx_user, :] == 0)[0]
                probas_[idx_mask] -= 1
                idx_question = np.argsort(np.abs(probas_ - 0.5))[0]
                selections.append(idx_question)

        else:
            selections = []
            for idx_user in range(nb_unseen_users):
                mask = (can_query[idx_user, :] == 1)
                subset_idx = np.argmin(np.abs(self.probas[idx_user, :][mask] - 0.5))
                parent_idx = np.arange(self.probas[idx_user, :].shape[0])[mask][subset_idx]
                selections.append(parent_idx)


        self.is_first_round = False
        self.selections = selections

        return selections

    def update_model(self, masked_data, masked_binary_data, can_query):
        """
        Update the model to incorporate newly revealed data if desired (e.g. training or updating model state).
        Args:
            masked_data (np.array): Array of shape (num_students, num_questions) containing categorical data revealed to 
                the model at the current step. Unobserved values are denoted by -1.
            masked_binary_data (np.array): Array of shape (num_students, num_questions) containing binary data revealed 
                to the model at the current step. Unobserved values are denoted by -1.
            can_query (np.array): Binary array of shape (num_students, num_questions), indicating which features can be 
                queried by the model in the current step.
        Returns:
            selections (list): List of ints, length num_students, containing the index of the feature selected to query 
            for each student (row) in the dataset.
        """

        new_user = {}
        data = []
        for i, question_id in enumerate(self.selections):
            is_correct = masked_binary_data[i, question_id]
            new_user[i] = i + self.offset
            data.append([question_id, i + self.offset, is_correct])
        df = pd.DataFrame(data, columns=["QuestionId", "UserId", "IsCorrect"])
        self.df_data = pd.concat([self.df_data, df])

        nb_students, nb_questions = masked_data.shape
        X, y, self.user_to_idx = transform_df_data(self.df_data)
        self.lr = LogisticRegression(solver="liblinear")
        self.lr.fit(X, y, sample_weight=None)

        # compute probabilities for each unseen_user/item
        nb_total_users = len(self.user_to_idx)
        self.probas = np.zeros(masked_data.shape, dtype=np.float16)
        nb_unseen_users = masked_data.shape[0]
        questions_ = np.eye(nb_questions)
        for i in range(nb_unseen_users):
            user_ = np.zeros((nb_questions, nb_total_users))
            user_[:, self.user_to_idx[new_user[i]]] = 1
            inputs = np.c_[user_, questions_]
            self.probas[i, :] = self.lr.predict_proba(inputs)[:, 1]


    def predict(self, masked_data, masked_binary_data):
        """
        Use your model to predict missing binary values in the input data. Both categorical and binary versions of the
        observed input data are available for making predictions with.
        Args:
            masked_data (np.array): Array of shape (num_students, num_questions) containing categorical data revealed to 
                the model at the current step. Unobserved values are denoted by -1.
            masked_binary_data (np.array): Array of shape (num_students, num_questions) containing binary data revealed 
                to the model at the current step. Unobserved values are denoted by -1.
        Returns:
            predictions (np.array): Array of shape (num_students, num_questions) containing predictions for the
                unobserved values in `masked_binary_data`. The values given to the observed data in this array will be 
                ignored.
        """

        return np.round(self.probas)
