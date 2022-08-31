"""
Collaborative filtering with TF 2.9 Keras.
"""
import logging
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


BATCH_SIZE = 100
N_DIM = 20
LAMBDA_REG = 0.1
LEARNING_RATE = 0.01


df = pd.read_csv('../vae/data/movie100k/data.csv')
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)
n_samples = len(df_train)
logging.warning('Movie100k data loaded: %s', df_train.shape)
n_users = 1 + df['user'].nunique()
n_items = 1 + df['item'].nunique()


train_dataset = tf.data.Dataset.from_tensor_slices((
    df_train[['user', 'item']],
    df_train['rating'])).batch(BATCH_SIZE).shuffle(1000)
test_data = (df_test[['user', 'item']], df_test['rating'])


class CF(tf.keras.Model):
    '''
    Latent factor model with L2 regularization.
    R_ij = bias^user_i + bias^item_j + u_i^T v_j
    '''
    def __init__(self):
        super().__init__()
        l2_regularizer = tf.keras.regularizers.l2(
            LAMBDA_REG / (n_samples / BATCH_SIZE))
        self.user_bias = tf.keras.layers.Embedding(n_users, 1,
            embeddings_regularizer=l2_regularizer)
        self.item_bias = tf.keras.layers.Embedding(n_items, 1,
            embeddings_regularizer=l2_regularizer)
        self.user_emb = tf.keras.layers.Embedding(n_users, N_DIM,
            embeddings_regularizer=l2_regularizer)
        self.item_emb = tf.keras.layers.Embedding(n_items, N_DIM,
            embeddings_regularizer=l2_regularizer)

    def call(self, users_items):
        user_batch = users_items[:, 0]
        item_batch = users_items[:, 1]
        user_bias = self.user_bias(user_batch)
        item_bias = self.item_bias(item_batch)
        user_embed = self.user_emb(user_batch)
        item_embed = self.item_emb(item_batch)
        dot_prod = tf.keras.layers.dot([user_embed, item_embed], axes=-1)
        pred = tf.keras.layers.add([user_bias, item_bias, dot_prod])
        return pred

    def display(self):
        '''
        Ça c'est juste pour que summary et plot_model soient jolis, sinon ça ne sert à rien
        Merci https://github.com/tensorflow/tensorflow/issues/31647#issuecomment-692586409
        '''
        input_ = tf.keras.Input(shape=(2,))
        built_model = tf.keras.Model(inputs=[input_], outputs=self.call(x))
        built_model.summary()
        tf.keras.utils.plot_model(
            built_model, to_file='model-cf.png', show_shapes=True)


model = CF()
model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
model.fit(train_dataset, validation_data=test_data, epochs=50, batch_size=1000,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])

model.display()
