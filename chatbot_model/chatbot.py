import os
import re

import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.models import load_model

CURRENT_DIR = os.path.dirname(__file__)


def process_tokens(input_file):
    with open(f'{CURRENT_DIR}/{input_file}') as f:
        lines = f.readlines()
        input_docs =  list(lines[0].replace('\n', '').split(','))
        target_docs = list(lines[1].replace('\n', '').split(','))
        input_tokens = eval(lines[2])
        target_tokens = eval(lines[3])

    input_tokens = sorted(list(input_tokens))
    target_tokens = sorted(list(target_tokens))

    input_features_dict = dict(
        [(token, i) for i, token in enumerate(input_tokens)])
    target_features_dict = dict(
        [(token, i) for i, token in enumerate(target_tokens)])

    max_encoder_seq_length = max(
        [len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
    max_decoder_seq_length = max(
        [len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])

    return max_decoder_seq_length, max_encoder_seq_length, input_features_dict, target_features_dict


def generate_encoder_model(model):
    training_model = load_model(f'{CURRENT_DIR}/{model}')
    encoder_inputs = training_model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)
    return encoder_model


def generate_decoder_model(num_decoder_tokens):
    dimensionality = 256
    decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    latent_dim = 256
    decoder_state_input_hidden = Input(shape=(latent_dim,))
    decoder_state_input_cell = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
    decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs,
                                                             initial_state=decoder_states_inputs)
    decoder_states = [state_hidden, state_cell]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return decoder_model


def decode_response(test_input, max_decoder_seq_length, target_features_dict):
    # Getting the output states to pass into the decoder
    num_decoder_tokens = len(target_features_dict)
    encoder_model = generate_encoder_model('training_model_V2.h5')
    states_value = encoder_model.predict(test_input)
    # Generating empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Setting the first token of target sequence with the start token
    target_seq[0, 0, target_features_dict['<START>']] = 1.

    # A variable to store our response word by word
    decoded_sentence = ''

    stop_condition = False

    reverse_target_features_dict = dict(
        (i, token) for token, i in target_features_dict.items())

    decoder_model = generate_decoder_model(num_decoder_tokens)

    while not stop_condition:
        # Predicting output tokens with probabilities and states
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
        # Choosing the one with highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        decoded_sentence += " " + sampled_token
        # Stop if hit max length or found the stop token
        if sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        # Update the target sequence
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        # Update states
        states_value = [hidden_state, cell_state]
    return decoded_sentence


class ChatBot:
    exit_commands = ("salir", "pausa", "exit", "salida", "chao", "bye", "stop", "parar")

    # Method to start the conversation
    def start_chat(self, user_response):
        if user_response in self.exit_commands:
            return "Ok, ¡que tengas un buen día!"
        return self.chat(user_response)

    # Method to handle the conversation
    def chat(self, reply):
        return self.generate_response(reply)

    # Method to convert user input into a matrix
    def string_to_matrix(self, user_input, max_encoder_seq_length, input_features_dict):
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
        num_encoder_tokens = len(input_features_dict)
        user_input_matrix = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        for timestep, token in enumerate(tokens):
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.
        return user_input_matrix

    # Method that will create a response using seq2seq model we built
    def generate_response(self, user_input):
        max_decoder_seq_length, max_encoder_seq_length, input_features_dict, target_features_dict = process_tokens('data_processed.txt')
        input_matrix = self.string_to_matrix(user_input, max_encoder_seq_length, input_features_dict)
        chatbot_response = decode_response(input_matrix, max_decoder_seq_length, target_features_dict)
        # Remove <START> and <END> tokens from chatbot_response
        chatbot_response = chatbot_response.replace("<START>", '')
        chatbot_response = chatbot_response.replace("<END>", '')
        return chatbot_response
