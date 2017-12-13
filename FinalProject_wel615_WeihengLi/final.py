# keras_utilities
# ekphrasis
# scikit-learn
# frozendict
# cachetools
# tqdm
# ftfy


from data_loader import Task4Loader
from kutilities.helpers.data_preparation import get_labels_to_categories_map, get_class_weights2, onehot_to_categories
from kutilities.callbacks import MetricsCallback
from keras.layers.recurrent import LSTM
from WordVectorsManager import WordVectorsManager
from collections import Counter
from ignore_warnings import set_ignores
from keras.constraints import maxnorm
from keras.engine import Input
from keras.engine import Model
#from keras.engine import merge
from keras.layers import merge, concatenate
from keras.layers import Dropout, Dense, Bidirectional, \
    Embedding, GaussianNoise, Activation, Flatten, \
    TimeDistributed, RepeatVector, Permute, MaxoutDense, GlobalMaxPooling1D, \
    Convolution1D, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from kutilities.layers import AttentionWithContext, Attention, MeanOverTime
from sklearn import preprocessing
from keras.layers import LSTM,Conv1D,MaxPooling1D, Bidirectional, Dropout
import numpy
from ignore_warnings import set_ignores
from keras import backend as K
from keras import models 
from keras.utils import plot_model



set_ignores()

def f1(y_true, y_pred):
    '''
    from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
    def recall(y_true, y_pred):
        """
        Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))


def target_RNN(embeddings, classes, max_length, unit=LSTM, cells=64,
                        layers=1, **kwargs):
    # parameters
    bi = kwargs.get("bidirectional", False)
    noise = kwargs.get("noise", 0.)
    dropout_words = kwargs.get("dropout_words", 0)
    dropout_rnn = kwargs.get("dropout_rnn", 0)
    dropout_rnn_U = kwargs.get("dropout_rnn_U", 0)
    dropout_attention = kwargs.get("dropout_attention", 0)
    dropout_final = kwargs.get("dropout_final", 0)
    attention = kwargs.get("attention", None)
    final_layer = kwargs.get("final_layer", False)
    clipnorm = kwargs.get("clipnorm", 1)
    loss_l2 = kwargs.get("loss_l2", 0.)
    lr = kwargs.get("lr", 0.001)


    input_text = Input(shape=(max_length,), dtype='int32')

    emb_text = embeddings_layer(max_length=max_length, embeddings=embeddings,
                               trainable=False, masking=True, scale=False,
                               normalize=False)(input_text)

    if noise > 0:
        emb_text = GaussianNoise(noise)(emb_text)
    if dropout_words > 0:
        emb_text = Dropout(dropout_words)(emb_text)

    merge_text = []
    # #one lstm

    # # cov_text = Conv1D(activation="relu", padding="same", filters=300, kernel_size=5)(emb_text)
    # # pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    # # pooling_text = GlobalMaxPooling1D()(cov_text)

    # layer_output = get_RNN(unit, cells, bi=True, return_sequences=False, dropout_U=dropout_rnn_U)(emb_text)
    # if dropout_rnn > 0:
    #     layer_output = Dropout(dropout_rnn)(layer_output)

    # #one conv

    # cov_text = Conv1D(activation="relu", padding="same", filters=300, kernel_size=5)(emb_text)
    # #pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    # pooling_text = GlobalMaxPooling1D()(cov_text)




    # #bilstm+bilstm+attention
    # # cov_text = Conv1D(activation="relu", padding="same", filters=300, kernel_size=5)(emb_text)
    # # pooling_text = MaxPooling1D(pool_size=4)(cov_text)

    # layer_output = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(emb_text)
    # if dropout_rnn > 0:
    #     layer_output = Dropout(dropout_rnn)(layer_output)
    # layer_output2 = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(layer_output)
    # if dropout_rnn > 0:
    #     layer_output2 = Dropout(dropout_rnn)(layer_output2)

    # attention_text = Attention()(layer_output2)
    

    # if dropout_attention > 0:
    #     attention_text = Dropout(dropout_attention)(attention_text)


    # #bilstm+bilstm+bilstm+attention
    # # cov_text = Conv1D(activation="relu", padding="same", filters=300, kernel_size=5)(emb_text)
    # # pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    # # pooling_text = GlobalMaxPooling1D()(cov_text)

    # layer_output = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(emb_text)
    # if dropout_rnn > 0:
    #     layer_output = Dropout(dropout_rnn)(layer_output)
    # layer_output2 = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(layer_output)
    # if dropout_rnn > 0:
    #     layer_output2 = Dropout(dropout_rnn)(layer_output2)
    # layer_output3 = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(layer_output2)
    # if dropout_rnn > 0:
    #     layer_output3 = Dropout(dropout_rnn)(layer_output3)

    # attention_text = Attention()(layer_output3)
    

    # if dropout_attention > 0:
    #     attention_text = Dropout(dropout_attention)(attention_text)


    # #origin one lstm
    # # cov_text = Conv1D(activation="relu", padding="same", filters=300, kernel_size=5)(emb_text)
    # # pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    # # pooling_text = GlobalMaxPooling1D()(cov_text)

    # layer_output = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(emb_text)
    # if dropout_rnn > 0:
    #     layer_output = Dropout(dropout_rnn)(layer_output)

    # attention_text = Attention()(layer_output)
    

    # if dropout_attention > 0:
    #     attention_text = Dropout(dropout_attention)(attention_text)


    # #origin conv one lstm
    # cov_text = Conv1D(activation="relu", padding="same", filters=300, kernel_size=5)(emb_text)
    # pooling_text = MaxPooling1D(pool_size=4)(cov_text)

    # layer_output = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(pooling_text)
    # if dropout_rnn > 0:
    #     layer_output = Dropout(dropout_rnn)(layer_output)

    # attention_text = Attention()(layer_output)
    

    # if dropout_attention > 0:
    #     attention_text = Dropout(dropout_attention)(attention_text)


    #bilstm+bilstm+attention merge bilstm+attention
    # cov_text = Conv1D(activation="relu", padding="same", filters=300, kernel_size=5)(emb_text)
    # pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    # pooling_text = GlobalMaxPooling1D()(cov_text)

    layer_output = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(emb_text)
    if dropout_rnn > 0:
        layer_output = Dropout(dropout_rnn)(layer_output)

    attention_text = Attention()(layer_output)
    if dropout_attention > 0:
        attention_text = Dropout(dropout_attention)(attention_text)

    layer_output2 = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(layer_output)
    if dropout_rnn > 0:
        layer_output2 = Dropout(dropout_rnn)(layer_output2)

    attention_text2 = Attention()(layer_output2)
    

    if dropout_attention > 0:
        attention_text2 = Dropout(dropout_attention)(attention_text2)

    merge_text.append(attention_text)
    merge_text.append(attention_text2)
    attention_mul = concatenate(merge_text)





    # # merge conv lstm lstm+lstm
    # cov_text = Conv1D(activation="relu", padding="same", filters=300, kernel_size=5)(emb_text)
    # pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    # globalpooling = GlobalMaxPooling1D()(cov_text)
    # merge_text.append(globalpooling)

    # layer_output = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(pooling_text)
    # if dropout_rnn > 0:
    #     layer_output = Dropout(dropout_rnn)(layer_output)

    # attention_text2 = Attention()(layer_output)
    # if dropout_attention > 0:
    #     attention_text2 = Dropout(dropout_attention)(attention_text2)
    # merge_text.append(attention_text2)
    # layer_output2 = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(layer_output)
    # if dropout_rnn > 0:
    #     layer_output2 = Dropout(dropout_rnn)(layer_output2)

    # # layer_input = {}
    # # layer_output = {-1:pooling_text}
    # # for i in range(layers):
    # #     j=i
    # #     layer_input[i] = layer_output[i-1]
    # #     rs = (layers > 1 and i < layers - 1) or attention

    # #     layer_output[i] = get_RNN(unit, cells, bi, return_sequences=rs,
    # #                       dropout_U=dropout_rnn_U)(layer_input[i])
    # #     if dropout_rnn > 0:
    # #         layer_output[i] = Dropout(dropout_rnn)(layer_output[i])
    # # if layers==0:
    # #     j = -1

    # attention_text = Attention()(layer_output2)
    

    # if dropout_attention > 0:
    #     attention_text = Dropout(dropout_attention)(attention_text)

    # merge_text.append(attention_text)
    # #attention_mul = concatenate(merge_text)
    # attention_mul = merge(merge_text)



    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(attention_mul)
    model = Model(input=input_text, output=probabilities)
    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr),
                  loss='categorical_crossentropy', metrics=['accuracy',f1])#,f1_score,f1_score2,f12
    return model


def target_RNN2(embeddings, classes, max_length, unit=LSTM, cells=64,
                        **kwargs):
    # parameters
    bi = kwargs.get("bidirectional", False)
    noise = kwargs.get("noise", 0.)
    dropout_words = kwargs.get("dropout_words", 0)
    dropout_rnn = kwargs.get("dropout_rnn", 0)
    dropout_rnn_U = kwargs.get("dropout_rnn_U", 0)
    dropout_attention = kwargs.get("dropout_attention", 0)
    dropout_final = kwargs.get("dropout_final", 0)
    attention = kwargs.get("attention", None)
    final_layer = kwargs.get("final_layer", False)
    clipnorm = kwargs.get("clipnorm", 1)
    loss_l2 = kwargs.get("loss_l2", 0.)
    lr = kwargs.get("lr", 0.001)
    bi = kwargs.get("bi",False)

    attention_times = kwargs.get("attention_times",1)


    input_text = Input(shape=(max_length,), dtype='int32')

    emb_text = embeddings_layer(max_length=max_length, embeddings=embeddings,
                               trainable=False, masking=True, scale=False,
                               normalize=False)(input_text)

    if noise > 0:
        emb_text = GaussianNoise(noise)(emb_text)
    if dropout_words > 0:
        emb_text = Dropout(dropout_words)(emb_text)

    # cnn
    cov_text = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(emb_text)
    pooling_text = GlobalMaxPooling1D()(cov_text)
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(pooling_text)

    #cnn+lstm
    cov_text = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(emb_text)
    pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    lstm_text = get_RNN(unit, cells, bi, return_sequences=False, dropout_U=dropout_rnn_U)(pooling_text)
    lstm_text = Dropout(0.5)(lstm_text)
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(lstm_text)

    #lstm
    lstm_text = get_RNN(unit, cells, bi, return_sequences=False, dropout_U=dropout_rnn_U)(emb_text)
    lstm_text = Dropout(0.5)(lstm_text)
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(lstm_text)


    #bilstm
    lstm_text = get_RNN(unit, cells, bi=True, return_sequences=False, dropout_U=dropout_rnn_U)(emb_text)
    lstm_text = Dropout(0.5)(lstm_text)
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(lstm_text)

    #cnn+lstm+dense+lstm
    cov_text = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(emb_text)
    pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    lstm_text = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(pooling_text)
    lstm_text = Dropout(0.5)(lstm_text)
    attention_probs = Dense(32, activation='softmax', name='attention_probs')(lstm_text)
    attention_mul = merge([lstm_text, attention_probs], output_shape=32, name='attention_mul', mode='mul')
    lstm_text2 = get_RNN(unit, cells, bi, return_sequences=False, dropout_U=dropout_rnn_U)(attention_mul)
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(lstm_text2)

    #cnn+lstm+dense+lstm+denselstm
    cov_text = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(emb_text)
    pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    lstm_text = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(pooling_text)
    lstm_text = Dropout(0.5)(lstm_text)
    attention_probs = Dense(32, activation='softmax', name='attention_probs')(lstm_text)
    attention_mul = merge([lstm_text, attention_probs], output_shape=32, name='attention_mul', mode='mul')
    lstm_text2 = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(attention_mul)
    lstm_text2 = Dropout(0.5)(lstm_text2)
    attention_probs2 = Dense(32, activation='softmax', name='attention_probs2')(lstm_text2)
    attention_mul2 = merge([lstm_text2, attention_probs2], output_shape=32, name='attention_mul2', mode='mul')
    lstm_text3 = get_RNN(unit, cells, bi, return_sequences=False, dropout_U=dropout_rnn_U)(attention_mul2)
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(lstm_text3)

    #cnn+lstm+dense+conv+lstm+dense+lstm
    cov_text = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(emb_text)
    pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    lstm_text = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(pooling_text)
    lstm_text = Dropout(0.5)(lstm_text)
    attention_probs = Dense(32, activation='softmax', name='attention_probs')(lstm_text)
    attention_mul = merge([lstm_text, attention_probs], output_shape=32, name='attention_mul', mode='mul')
    cov_text2 = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(attention_mul)
    pooling_text2 = MaxPooling1D(pool_size=4)(cov_text2)
    lstm_text2 = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(pooling_text2)
    lstm_text2 = Dropout(0.5)(lstm_text2)
    attention_probs2 = Dense(32, activation='softmax', name='attention_probs2')(lstm_text2)
    attention_mul2 = merge([lstm_text2, attention_probs2], output_shape=32, name='attention_mul2', mode='mul')
    lstm_text3 = get_RNN(unit, cells, bi, return_sequences=False, dropout_U=dropout_rnn_U)(attention_mul2)
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(lstm_text3)

    # cnn+lstm+dense+conv
    cov_text = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(emb_text)
    pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    lstm_text = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(pooling_text)
    lstm_text = Dropout(0.5)(lstm_text)
    attention_probs = Dense(32, activation='softmax', name='attention_probs')(lstm_text)
    attention_mul = merge([lstm_text, attention_probs], output_shape=32, name='attention_mul', mode='mul')
    conv_text2 = Convolution1D(nb_filter=80, filter_length=4, border_mode='valid', activation='relu')(attention_mul)
    globalpooling = GlobalMaxPooling1D()(conv_text2)
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(globalpooling)

    #lstm + conv + lstm
    lstm_text = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(emb_text)
    lstm_text = Dropout(0.5)(lstm_text)
    cov_text = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(lstm_text)
    pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    lstm_text2 = get_RNN(unit, cells, bi, return_sequences=False, dropout_U=dropout_rnn_U)(pooling_text)
    lstm_text2 = Dropout(0.5)(lstm_text2)
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(lstm_text2)

    #conv+lstm merge lstm
    cov_text = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(emb_text)
    pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    lstm_text = get_RNN(unit, cells, bi, return_sequences=False, dropout_U=dropout_rnn_U)(pooling_text)
    lstm_text = Dropout(0.5)(lstm_text)
    lstm_text2 = get_RNN(unit, cells, bi, return_sequences=False, dropout_U=dropout_rnn_U)(emb_text)
    lstm_text2 = Dropout(0.5)(lstm_text2)
    merge_text = merge([lstm_text,lstm_text2])
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(merge_text)

    #cov+lstm+cont+conv+lstm+cont+conv
    cov_text = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(emb_text)
    pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    lstm_text = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(pooling_text)
    lstm_text = Dropout(0.5)(lstm_text)
    attention_probs = Dense(32, activation='softmax', name='attention_probs')(lstm_text)
    attention_mul = merge([lstm_text, attention_probs], output_shape=32, name='attention_mul', mode='mul')
    cov_text2 = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(attention_mul)
    pooling_text2 = MaxPooling1D(pool_size=4)(cov_text2)
    lstm_text2 = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(pooling_text2)
    lstm_text2 = Dropout(0.5)(lstm_text2)
    attention_probs2 = Dense(32, activation='softmax', name='attention_probs2')(lstm_text2)
    attention_mul2 = merge([lstm_text2, attention_probs2], output_shape=32, name='attention_mul2', mode='mul')
    conv_text3 = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(attention_mul2)
    globalpooling = GlobalMaxPooling1D()(conv_text3)
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(globalpooling)


    #cov+lstm+cont+conv+lstm+cont+conv merge cov+lstm+cont+conv merge cov
    conv_text = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(emb_text)
    globalpooling = GlobalMaxPooling1D()(conv_text)

    cov_text = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(emb_text)
    pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    lstm_text = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(pooling_text)
    lstm_text = Dropout(0.5)(lstm_text)
    attention_probs = Dense(32, activation='softmax', name='attention_probs')(lstm_text)
    attention_mul = merge([lstm_text, attention_probs], output_shape=32, name='attention_mul', mode='mul')
    conv_text2 = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(attention_mul)
    globalpooling2 = GlobalMaxPooling1D()(conv_text2)

    cov_text2 = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(attention_mul)
    pooling_text2 = MaxPooling1D(pool_size=4)(cov_text2)
    lstm_text2 = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(pooling_text2)
    lstm_text2 = Dropout(0.5)(lstm_text2)
    attention_probs2 = Dense(32, activation='softmax', name='attention_probs')(lstm_text2)
    attention_mul2 = merge([lstm_text2, attention_probs2], output_shape=32, name='attention_mul', mode='mul')
    conv_text3 = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(attention_mul2)
    globalpooling3 = GlobalMaxPooling1D()(conv_text3)
    merge_text = merge([globalpooling,globalpooling2,globalpooling])
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(merge_text)

    #lstm merge conv+lstm merge conv+lstm+conv+lstm
    cov_text = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(emb_text)
    pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    lstm_text = get_RNN(unit, cells, bi, return_sequences=False, dropout_U=dropout_rnn_U)(pooling_text)
    lstm_text = Dropout(0.5)(lstm_text)

    lstm_text2_1 = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(pooling_text)
    lstm_text2_1 = Dropout(0.5)(lstm_text2_1)
    cov_text2 = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(lstm_text2_1)
    pooling_text2 = MaxPooling1D(pool_size=4)(cov_text2)
    lstm_text3 = get_RNN(unit, cells, bi, return_sequences=False, dropout_U=dropout_rnn_U)(pooling_text2)
    lstm_text3 = Dropout(0.5)(lstm_text3)


    lstm_text2 = get_RNN(unit, cells, bi, return_sequences=False, dropout_U=dropout_rnn_U)(emb_text)
    lstm_text2 = Dropout(0.5)(lstm_text2)


    merge_text = merge([lstm_text,lstm_text2,lstm_text3])
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(merge_text)

    #cnn+lstm+dense+lstm+dense+conv
    cov_text = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(emb_text)
    pooling_text = MaxPooling1D(pool_size=4)(cov_text)
    lstm_text = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(pooling_text)
    lstm_text = Dropout(0.5)(lstm_text)
    attention_probs = Dense(32, activation='softmax', name='attention_probs')(lstm_text)
    attention_mul = merge([lstm_text, attention_probs], output_shape=32, name='attention_mul', mode='mul')
    lstm_text2 = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(attention_mul)
    lstm_text2 = Dropout(0.5)(lstm_text2)
    attention_probs2 = Dense(32, activation='softmax', name='attention_probs2')(lstm_text2)
    attention_mul2 = merge([lstm_text2, attention_probs2], output_shape=32, name='attention_mul2', mode='mul')
    cov_text2 = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(attention_mul2)
    globalpooling2 = GlobalMaxPooling1D()(cov_text2)
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(globalpooling2)


    #lstm+dense+lstm+dense+conv
    lstm_text = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(emb_text)
    lstm_text = Dropout(0.5)(lstm_text)
    attention_probs = Dense(32, activation='softmax', name='attention_probs')(lstm_text)
    attention_mul = merge([lstm_text, attention_probs], output_shape=32, name='attention_mul', mode='mul')
    lstm_text2 = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(attention_mul)
    lstm_text2 = Dropout(0.5)(lstm_text2)
    attention_probs2 = Dense(32, activation='softmax', name='attention_probs2')(lstm_text2)
    attention_mul2 = merge([lstm_text2, attention_probs2], output_shape=32, name='attention_mul2', mode='mul')
    cov_text2 = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(attention_mul2)
    globalpooling2 = GlobalMaxPooling1D()(cov_text2)
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(globalpooling2)

    # according to origin
    lstm_text = get_RNN(unit, cells, bi=True, return_sequences=True, dropout_U=dropout_rnn_U)(emb_text)
    lstm_text = Dropout(0.3)(lstm_text)
    print("a",lstm_text.shape)
    atten_text = Attention()(lstm_text)
    print("b",atten_text.shape)
    # cov_text2 = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(atten_text)
    # print("c",cov_text2.shape)
    # globalpooling2 = GlobalMaxPooling1D()(cov_text2)
    # print("d",globalpooling2.shape)
    probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(atten_text)






    #lstm+dense+lstm+dense

    # # cov_text = Convolution1D(nb_filter=80, filter_length=4,
    # #                        border_mode='valid', activation='relu')(emb_text)
    # # pooling_text = GlobalMaxPooling1D()(cov_text)
    # # pooling_text = MaxPooling1D(pool_size=4)(cov_text)

    # cov_text = Conv1D(activation="relu", padding="same", filters=64, kernel_size=5)(emb_text)
    # pooling_text = MaxPooling1D(pool_size=4)(cov_text)

    # # lstm_text = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(emb_text)
    # # if dropout_rnn > 0:
    # #     lstm_text = Dropout(dropout_rnn)(lstm_text)
    # # attention_probs = Dense(32, activation='softmax', name='attention_probs')(lstm_text)
    # # attention_mul = merge([lstm_text, attention_probs], output_shape=32, name='attention_mul', mode='mul')






    # # all_lstm_text={}
    # # all_lstm_text[0]=pooling_text
    # # all_attention_probs={}
    # # all_attention_mul={}

    # # for i in range(attention_times):
    # #     j=i
    # #     all_lstm_text[i+1] = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(all_lstm_text[i])
    # #     if dropout_rnn > 0:
    # #         all_lstm_text[i+1] = Dropout(dropout_rnn)(all_lstm_text[i+1])
    # #     attention_probs[i] = Dense(32, activation='softmax', name='attention_probs')(all_lstm_text[i+1])
    # #     attention_mul[i] = merge([all_lstm_text[i+1], attention_probs[i]], output_shape=32, name='attention_mul', mode='mul')

    # lstm_text = get_RNN(unit, cells, bi, return_sequences=False, dropout_U=dropout_rnn_U)(pooling_text)
    # lstm_text = Dropout(0.5)(lstm_text)
    # # cov_text = Convolution1D(nb_filter=80, filter_length=4,
    # #                         border_mode='valid', activation='relu')(lstm_text)
    # # # we use max pooling:
    # # pooling_text = GlobalMaxPooling1D()(cov_text)
    # # lstm_text2 = get_RNN(unit, cells, bi, return_sequences=True, dropout_U=dropout_rnn_U)(lstm_text)
    # # lstm_text2 = Dropout(0.3)(lstm_text2)
    # # cov_text2 = Convolution1D(nb_filter=80, filter_length=4,
    # #                         border_mode='valid', activation='relu')(lstm_text2)
    # # # we use max pooling:
    # # pooling_text2 = GlobalMaxPooling1D()(cov_text2)
    # # merge_text = merge([pooling_text,pooling_text2])
    # probabilities = Dense(classes, activation='softmax', activity_regularizer=l2(loss_l2))(lstm_text)

    model = Model(input=input_text, output=probabilities)
    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def embeddings_layer(max_length, embeddings, trainable=False, masking=False,
                     scale=False, normalize=False):
    if scale:
        print("Scaling embedding weights...")
        embeddings = preprocessing.scale(embeddings)
    if normalize:
        print("Normalizing embedding weights...")
        embeddings = preprocessing.normalize(embeddings)

    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]
    print("vocab_size",vocab_size, embeddings.shape)
    _embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        input_length=max_length if max_length > 0 else None,
        trainable=trainable,
        mask_zero=masking if max_length > 0 else False,
        weights=[embeddings]
    )

    return _embedding


def get_RNN(unit=LSTM, cells=64, bi=False, return_sequences=False, dropout_U=0.,
            consume_less='cpu', l2_reg=0):
    # rnn = LSTM(cells, return_sequences=return_sequences, dropout=0.5, recurrent_dropout=0.5)
    rnn = unit(cells, return_sequences=return_sequences,
               consume_less=consume_less, dropout=dropout_U,
               kernel_regularizer=l2(l2_reg))
    if bi:
        return Bidirectional(rnn)
    else:
        return rnn

if __name__ == "__main__":
    max_length = 50
    dim = 300
    WV_name = "datastories.twitter."+str(dim)+"d"

    #get embeddings

    vectors = WordVectorsManager(WV_name).read()
    vocab_size = len(vectors)
    print('Loaded %s word vectors.' % vocab_size)
    word_indices = {}
    pos = 0
    # +1 for zero padding token and +1 for unk
    embeddings = numpy.ndarray((vocab_size + 2, dim), dtype='float32')
    for i, (word, vector) in enumerate(vectors.items()):
        if len(vector) > dim-1:
            pos = i + 1
            word_indices[word] = pos
            embeddings[pos] = vector

    # add unknown token
    pos += 1
    word_indices["<unk>"] = pos
    embeddings[pos] = numpy.random.uniform(low=-0.05, high=0.05, size=dim)
    print(embeddings[:2])
    j=0
    for i in word_indices:
        print(i,word_indices[i])
        j+=1
        if j>5:
            break;


    #load data
    loader = Task4Loader(word_indices, text_lengths=max_length)

    training, testing = loader.load_final()
    print(training[0][:2])

    #build NN one simple and one attention
    print("Building NN Model...")

    # nn_model = target_RNN(embeddings, classes=3, max_length=max_length, unit=LSTM,
    #                                cells=32, attention="simple", noise=0.1, loss_l2=0.0001,
    #                                dropout_attention=0.1, dropout_rnn=0.1, dropout_rnn_U=0, 
    #                                attention_times=1,dropout_words=0.5, bi = False)

    nn_model = target_RNN(embeddings, classes=3,
                               max_length=max_length, layers=2, unit=LSTM,
                               cells=150, bidirectional=True,
                               attention="simple",
                               noise=0.3, clipnorm=1, lr=0.001, loss_l2=0.0001,
                               final_layer=False, dropout_final=0.5,
                               dropout_attention=0.5,
                               dropout_words=0.3, dropout_rnn=0.3,
                               dropout_rnn_U=0.3)

    # nn_model = cnn_multi_filters(embeddings, max_length, [3, 4, 5], 100,
    #                          noise=0.1,
    #                          drop_text_input=0.2,
    #                          drop_conv=0.5, )t_attention=0.1, dropout_rnn=0.1, dropout_rnn_U=0)

    # nn_model = cnn_simple(embeddings, max_length)
    plot_model(nn_model, show_layer_names=True, show_shapes=True,
     to_file="taskimage.png")
    print(nn_model.summary())
    #exit()
    

    classes = ['positive', 'negative', 'neutral']
    class_to_cat_mapping = get_labels_to_categories_map(classes)
    cat_to_class_mapping = {v: k for k, v in
                            get_labels_to_categories_map(classes).items()}

    _datasets = {}
    _datasets["1-train"] = training
    _datasets["2-val"] = testing

    class_weights = get_class_weights2(onehot_to_categories(training[1]))
    # print(training[0])
    print("Class weights:", {cat_to_class_mapping[c]: w for c, w in class_weights.items()})
    # print(training[0].shape,training[1].shape)
    history = nn_model.fit(training[0], training[1],
                           validation_data=testing,
                           epochs=20, batch_size=50,
                           class_weight=class_weights)

