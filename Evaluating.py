from Encoder import *
from Decoder import *
from Data_PreProcessing import *
from Decoding_Strategies import *
import os

os.chdir(os.path.dirname(__file__))

embedding_dim = 256
units = 512
vocab_size = 10000 + 1
max_length = max_and_min_lengths_of_tokenized_captions()[0]

features_shape = 2560
attention_features_shape = 324

BATCH_SIZE = 8
BUFFER_SIZE = 1000

efficientNetB7_featureExtractor = load_efficientNetB7_featureExtractor()
tokenizer = load_tokenizer()
encoder = EfficientNetB7_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
optimizer = tf.keras.optimizers.Adam()

checkpoint_path = 'Checkpoints/Train'
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=30)

try:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f'\nLatest checkpoint ({ckpt_manager.latest_checkpoint}) loaded successfully ..\n')
except:
    print('\nError when loading the latest checkpoint ..\n')    

def evaluate_greedy(image):
    attention_plot = np.zeros((max_length, attention_features_shape))  # Store the attention weights for every time
    # step. shape = (74, 324)

    hidden = decoder.reset_state(batch_size=1)  # reset the hidden state for the decoder, shape = (1, 512)

    temp_input = tf.expand_dims(load_image(image)[0], 0)  # shape = (1, 600, 600, 3)
    img_tensor_val = efficientNetB7_featureExtractor(temp_input)  # shape = (1, 18, 18, 2560)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))  # shape = (1,
    # 324, 2560)

    features = encoder(img_tensor_val)  # shape = (1, 324, 256)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)  # shape = (1, 1)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        # predictions shape = (1, 10001)
        # hidden shape = (1, 512)
        # attention_weights shape = (1, 324, 1)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()  # shape = (324,)

        predicted_id = tf.argmax(predictions[0]).numpy()  # apply Greedy Search, prediction_id shape = ()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':  # Stop predicting when the model predicts the <end> token.
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)  # shape = (1, 1)

    attention_plot = attention_plot[:len(result), :]  # take the predicted attention weights for each time step.
    return result, attention_plot


def evaluate_beam(image, beam_width=10):
    start = [tokenizer.word_index['<start>']]

    result = [[start, 0.0]]  # [[a_Sequence_of_Tokens], its probability]

    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = efficientNetB7_featureExtractor(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    while len(result[0][0]) < max_length:
        i = 0
        temp = []
        for s in result:

            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
            i = i + 1
            # Getting the top <beam_width> (n) predictions
            word_preds = np.argsort(predictions[0])[-beam_width:]

            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += predictions[0][w]
                temp.append([next_cap, prob])
        result = temp
        # Sorting according to the probabilities
        result = sorted(result, reverse=False, key=lambda l: l[1])
        # Getting the top words
        result = result[-beam_width:]

        predicted_id = result[-1]  # Feed the decoder with the token which has the Max Probability
        pred_list = predicted_id[0]

        prd_id = pred_list[-1]
        if prd_id != 3:
            dec_input = tf.expand_dims([prd_id], 0)  # Decoder input is the word predicted with highest probability
            # among the top_k words predicted
        else:
            break

    result = result[-1][0]

    intermediate_caption = [tokenizer.index_word[i] for i in result]
    final_caption = []
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)

        else:
            break

    attention_plot = attention_plot[:len(result), :]
    return final_caption, attention_plot


def evaluate_nucleus(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = efficientNetB7_featureExtractor(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = nucleus_sampling(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def evaluate_topK(image):
    attention_plot = np.zeros((max_length, attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = efficientNetB7_featureExtractor(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()  # shape = (324,)

        predicted_id = top_k_sampling_with_temperature(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(7, 7))

    len_result = len(result)
    index = 1
    len_cleaned = len([word for word in result if word not in ['<start>', '<end>', '<unk>']])
    for token in range(len_result):
        if result[token] not in ['<start>', '<end>', '<unk>']:
            temp_att = np.resize(attention_plot[token], (18, 18))
            if len_cleaned <= 6:
                rows, columns = 2, 3
            else:
                rows, columns = 4, 4
            ax = fig.add_subplot(rows, columns, index)
            ax.set_title(result[token])
            index += 1
            plt.xticks([])
            plt.yticks([])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()
