from Encoder import *
from Decoder import *
from Data_PreProcessing import *
from Decoding_Strategies import *
from time import sleep
import os

os.chdir(os.path.dirname(__file__))

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('\nConfig is DONE!\n')
except:
    print('\nInvalid device or cannot modify virtual devices once initialized.\n')

embedding_dim = 256
units = 512
vocab_size = 10000 + 1

BATCH_SIZE = 8
BUFFER_SIZE = 1000
train_num_steps = len(load_train_images_paths_and_train_captions_stage2()[0]) // BATCH_SIZE

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

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

dataset = train_dataset = create_training_dataset_stage2(batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)


def calculate_time(start_time):
    elapsed_time = time.time() - start_time
    days = 0
    if elapsed_time >= 86400:
        days = int(elapsed_time / 86400)
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return days, elapsed


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(img_tensor, target):  # this function represent each training step (each training batch)
    loss = 0

    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        for i in range(1, target.shape[1]):
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


# Initializing a list (if not exists) to sore train losses while training.
if 'train_loss_plot_stage2.pkl' not in os.listdir('Loss Plots'):
    train_loss_plot_stage2_initial = []
    dump(train_loss_plot_stage2_initial, open('Loss Plots/train_loss_plot_stage2.pkl', 'wb'))


def start_training():
    start_epoch = 21
    beginning_time = time.time()

    try:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f'\nLatest checkpoint {ckpt_manager.latest_checkpoint} loaded successfully ..\n')
    except:
        print('\nError when loading the latest checkpoint ..\n')

    for epoch in range(start_epoch, 31):

        start = time.time()

        train_loss_plot_stage2 = load(open('Loss Plots/train_loss_plot_stage2.pkl', 'rb'))
        total_loss_train = 0
        for (batch, (features_batch, captions_batch)) in enumerate(tqdm(dataset, desc=f'Epoch {epoch} - Training', unit='batch')):
            batch_loss, t_loss = train_step(features_batch, captions_batch)
            total_loss_train += t_loss
        train_loss_plot_stage2.append(total_loss_train / train_num_steps)
        dump(train_loss_plot_stage2, open('Loss Plots/train_loss_plot_stage2.pkl', 'wb'))

        sleep(5)

        ckpt_manager.save()  # Creating a checkpoint.

        print('Epoch {}: TrainLoss {:.6f}'.format(epoch, (total_loss_train / train_num_steps)))
        days, elapsed_time = calculate_time(start)
        if days:
            print('Time taken for this epoch: {} Day(s) and {} \n'.format(days, elapsed_time))
        else:
            print('Time taken for this epoch: {} \n'.format(elapsed_time))

        sleep(5)

    days, elapsed_time = calculate_time(beginning_time)
    if days:
        print(f"Stage 2 of the Training Process took: {days} day(s) and {elapsed_time}")
    else:
        print(f"Stage 2 of the Training Process took: {elapsed_time}")


def plot_losses():
    train_loss_plot_stage1 = load(open('Loss Plots/train_loss_plot.pkl', 'rb'))
    train_loss_plot_stage2 = load(open('Loss Plots/train_loss_plot_stage2.pkl', 'rb'))
    train_loss_plot1 = [float(x) for x in train_loss_plot_stage1]
    train_loss_plot2 = [float(x) for x in train_loss_plot_stage2]
    plt.plot(train_loss_plot1)
    plt.plot(range(20, 30), train_loss_plot2)
    plt.title('Training Stages')
    plt.legend(['Train Loss - Stage 1 (126K Samples)', 'Train Loss - Stage 2 (158K Samples)'])
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xticks(range(0, 31, 5))
    plt.show()
