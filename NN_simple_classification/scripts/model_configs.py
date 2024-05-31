from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D, LSTM, Bidirectional
from transformers import BertTokenizer, TFBertModel
#from scripts.params import f1
import yaml


model_configs = {
    'Simple_Word2Vec': {
        'embedding_dim': 100,
        'vocab_size': 10000,
        'trainable': False,
        'model_layers': [
            Embedding(input_dim=10000, output_dim=100, trainable=False),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(6, activation='softmax')
        ],
        'compile_params': {
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy']
        }
    }
}
