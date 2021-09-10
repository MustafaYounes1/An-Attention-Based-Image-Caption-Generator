import pandas as pd
import string
from collections import Counter
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.utils import shuffle
from pickle import dump, load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import seaborn
from sklearn.model_selection import train_test_split
import numpy as np

os.chdir(os.path.dirname(__file__))


# TOKEN file:
# -----------
# A token is used as a replaceable tag within a topic and is represented using a token element.
# The inner text of the element is a token name. The tokens are defined in a separate token file.
# They are an easy way to represent common items that you use regularly such as a common phrase or external link.
def load_captions_as_dataframe():
    captions = pd.read_table(
        'Flickr30k/results_20130124.token', sep='\t',
        header=None,
        names=['image#index', 'caption'])
    data = []
    for val in captions['image#index'].values:
        data.append(val.split('#'))
    for i in range(len(data)):
        data[i].append(captions['caption'][i])
    data = pd.DataFrame(data, columns=["image_name", "index", 'caption'])
    data = data.reindex(columns=['index', 'image_name', 'caption'])
    return data


def vocabulary_size_before_cleaning():  # 23460
    data = load_captions_as_dataframe()
    vocab = []
    for txt in data.caption.values:
        vocab.extend(txt.split())
    print(f'Vocabulary Size Before Cleaning is : {len(set(vocab))}')


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)  # The maketrans() method itself returns a dictionary
    # describing each replacement, in unicode, string.maketrans(x, y, z) :
    """
        *   x:	Required. If only one parameter is specified, this has to be a dictionary describing how to perform the 
            replace. If two or more parameters are specified, this parameter has to be a string specifying the 
            characters you want to replace.
        *   y:	Optional. A string with the same length as parameter x. Each character in the first parameter will be 
            replaced with the corresponding character in this string.
        *   z:	Optional. A string describing which characters to remove from the original string.
    """
    text_no_punctuation = text.translate(translator)
    return text_no_punctuation


def remove_single_character(text):
    more_than_one = ""
    for word in text.split():
        if len(word) > 1:
            more_than_one += word + " "
    return more_than_one


def remove_numeric(text):
    text_no_numeric = ""
    for word in text.split():
        if word.isalpha():
            text_no_numeric += word + " "
    return text_no_numeric


def lowered_case(text):
    lowered = ""
    for word in text.split():
        lowered += word.lower() + " "
    return lowered


def cleaning_text(text):
    temp1 = remove_punctuation(text)
    temp2 = remove_single_character(temp1)
    temp3 = lowered_case(temp2)
    cleaned = remove_numeric(temp3)
    return cleaned


# pkl file:
# ----------
# A PKL file is a file created by pickle, a Python module that enables objects to be serialized
# to files on disk and deserialized back into the program at runtime. It contains a byte stream that represents the
# objects.
# The pickle module implements binary protocols for serializing and de-serializing a Python object
# structure. “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream,
# and “unpickling” is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is
# converted back into an object hierarchy.
def creating_dataframe_of_cleaned_captions():
    if 'cleaned_captions_as_dataframe.pkl' not in os.listdir('Flickr30k/'):
        data = load_captions_as_dataframe()
        print('Cleaning captions (this may take a while) ..')
        for i, caption in enumerate(data.caption.values):
            cleaned = cleaning_text(caption)
            data["caption"].iloc[i] = cleaned
        data.to_pickle('Flickr30k/cleaned_captions_as_dataframe.pkl')
    else:
        pass


def load_cleaned_captions_as_dataframe():
    return pd.read_pickle('Flickr30k/cleaned_captions_as_dataframe.pkl')


def vocabulary_size_after_cleaning():  # 19735
    data = load_cleaned_captions_as_dataframe()
    vocab = []
    for txt in data.caption.values:
        vocab.extend(txt.split())
    print(f'Vocabulary Size After Cleaning is : {len(set(vocab))}')


def top50_least50_word_frequencies():
    if 'top50_least_50_word_frequencies_2.png' not in os.listdir('Flickr30k/'):
        dataframe = load_cleaned_captions_as_dataframe()
        vocab = []
        for txt in dataframe.caption.values:
            vocab.extend(txt.split())
        ctr = Counter(vocab)  # Counter is a sub-class which is used to count hashable objects. It implicitly creates a
        # hash table of an iterable when invoked.
        keys, values = [], []
        for key in ctr.keys():
            keys.append(key)
        for value in ctr.values():
            values.append(value)
        data = {'word': keys, 'count': values}
        df = pd.DataFrame(data)
        df = df.sort_values(by='count', ascending=False)
        df = df.reset_index()[['word', 'count']]  # Reset the index of the DataFrame, and use the default one instead.
        top50 = df.truncate(before=0, after=49)
        least50 = df.truncate(before=19685, after=19734)
        fig, axes = plt.subplots(nrows=2, ncols=1)
        top50.plot(ax=axes[0], x='word', y='count', kind='bar', fontsize=15, title='50 most frequently appearing '
                                                                                   'words')
        least50.plot(ax=axes[1], x='word', y='count', kind='bar', color='red', fontsize=15, title='50 least frequently '
                                                                                                  'appearing words')
        plt.show()
    else:
        image = Image.open('Flickr30k/top50_least_50_word_frequencies_2.png')
        image.show()



def tagging_the_captions():  # returns a list of the cleaned, tagged captions.
    captions = []
    data = load_cleaned_captions_as_dataframe()
    for caption in data['caption'].astype(str):  # astype() method is used to cast a pandas object to a specified dtype.
        captions.append('<start> ' + caption.strip() + ' <end>')
    return captions


def images_paths():  # important Note: this function returns a list which has the same path -for each image- 5 times 
    # sequentially. (to keep a correspondence between captions list and images_paths list)
    data = load_cleaned_captions_as_dataframe()
    paths = []
    folder_path = 'Flickr30k/images/'
    for file_name in data['image_name'].astype(str):
        paths.append(folder_path + file_name)
    return paths


def num_of_images_and_captions_in_our_dataset():
    print(f'We have {len(set(images_paths()))} image in Flickr30k Dataset.')
    print(f'Each image has 5 captions, Thus we have {len(tagging_the_captions())} captions in Flickr30k.')


# We have 158,915 images and captions (each image has 5 captions, so we will use the same image five times in the
# training phase. We will take only 158,720 of our images and captions so that we can select batch size properly i.e.
# 2,480 batches if batch size = 64.
def data_limiter():
    # The shuffle is used to shuffle your matrices randomly. Programmatically, random sequences are generated using a
    # seed number. You are guaranteed to have the same random sequence if you use the same seed. The random_state
    # parameter allows you to provide this random seed to sklearn methods. This is useful because it allows you to
    # reproduce the randomness for your development and testing purposes.
    all_cleaned_tagged_captions = tagging_the_captions()
    all_images_paths = images_paths()
    random_captions, paths_to_random_images = shuffle(all_cleaned_tagged_captions, all_images_paths, random_state=1)
    limited_captions = random_captions[:158720]
    paths_to_limited_images = paths_to_random_images[:158720]
    # Due to this random process we will end up with 31,783 images from our data set (just by chance, it is the same
    # as the total number of images in Flickr30k data set)

    # pickle.dump(), which takes two arguments: the object you want to pickle and the file to which the object has to
    # be saved, 'wb': The w means that you'll be writing to the file, and b refers to binary mode.
    dump(limited_captions, open('Flickr30k/limited_captions.pkl', 'wb'))
    dump(paths_to_limited_images, open('Flickr30k/paths_to_limited_images.pkl', 'wb'))


def load_limited_images_paths_and_captions():
    # 'rb' : The r stands for read mode and the b stands for binary mode.
    limited_captions = load(open('Flickr30k/limited_captions.pkl', 'rb'))
    paths_to_limited_images = load(open('Flickr30k/paths_to_limited_images.pkl', 'rb'))
    return paths_to_limited_images, limited_captions


def vocabulary_size_after_data_limiting():  # 19732 , <start> and <end> tokens are a part of the vocabulary.
    captions = load_limited_images_paths_and_captions()[1]
    words = []
    for caption in captions:
        for word in caption.split():
            words.append(word)
    print(f'Vocabulary size after cleaning, tagging and limiting the captions: {len(set(words))}')


def tokenize_captions(words_to_keep=10000, captions=load_limited_images_paths_and_captions()[1]):
    """
    * tf.keras.preprocessing.text.Tokenizer
    ---------------------------------------
    This class allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer
    being the index of a token in a dictionary)
    - num_words: the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words
        will be kept.
    - filters: a string where each element is a character that will be filtered from the texts. The default is all
        punctuation, plus tabs and line breaks, minus the ' character.
    - oov_token (Out Of view): if given, it will be added to word_index and used to replace out-of-vocabulary words
        during text_to_sequence calls
    - lower: boolean. Whether to convert the texts to lowercase.
    - char_level: if True, every character will be treated as a token.
    - split: str. Separator for word splitting.

    By default, all punctuation is removed, turning the texts into space-separated sequences of words (words maybe
    include the ' character). These sequences are then split into lists of tokens. They will then be indexed or
    vectorized. (0 is a reserved index that won't be assigned to any word.)

    Note: after the tokenization process lower integer means more frequent.
    Note: Tokenizer will use only (num_words) most common words and at the same time, it will keep the counter of all
          words - even when it's obvious that it will not use it later.
    Note: tokenizer.word_counts returns each word with its count

    Methods:
        - fit_on_texts(texts): Updates internal vocabulary based on a list of texts.
            In the case where texts contains lists, we assume each entry of the lists to be a token.
        - fit_on_sequences(sequences): Updates internal vocabulary based on a list of sequences.
        - get_config(): Returns the tokenizer configuration as Python dictionary.
        - texts_to_sequences(texts): Transforms each text in texts to a sequence of integers.
            Only top num_words-1 most frequent words will be taken into account.(Only words known by the tokenizer
            will be taken into account.)
        - sequences_to_texts(sequences): Transforms each sequence into a list of text.
            Only top num_words-1 most frequent words will be taken into account.(Only words known by the tokenizer will
            be taken into account.)
    """
    tokenizer = Tokenizer(num_words=words_to_keep, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(captions)
    # Mapping '<pad>' to '0' , tokenizer.index_word[0] will be <pad>, and tokenizer.word_index['<pad>'] will be 0
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    tokenized_captions = tokenizer.texts_to_sequences(captions)
    dump(tokenizer, open('Flickr30k/tokenizer.pkl', 'wb'))
    dump(tokenized_captions, open('Flickr30k/tokenized_captions.pkl', 'wb'))
    """
    Results: In total we have 19734 words (<unk> and <pad> are a part of our dictionary now)
        - There are 7398 words appeared once.
        - There are 2403 words appeared twice.
        - There are 1383 words appeared 3 times.
    """


def load_tokenizer():
    return load(open('Flickr30k/tokenizer.pkl', 'rb'))


def load_tokenized_captions():
    return load(open('Flickr30k/tokenized_captions.pkl', 'rb'))


def plot_histogram_of_tokenized_captions_length():
    tokenized_captions = load_tokenized_captions()
    df = pd.DataFrame()
    df["sequence_length"] = list(map(lambda x: len(x), tokenized_captions))
    df = df.sort_values(by='sequence_length', ascending=False)
    df = df.reset_index()[['sequence_length']]
    seaborn.set()
    seaborn.histplot(df["sequence_length"])
    plt.show()


def max_and_min_lengths_of_tokenized_captions():  # max: 74, min: 3
    tokenized_captions = load_tokenized_captions()
    max_length = max(len(caption) for caption in tokenized_captions)
    min_length = min(len(caption) for caption in tokenized_captions)
    return max_length, min_length


def padding_tokenized_captions():
    """
    * tf.keras.preprocessing.sequence.pad_sequences  - returns Numpy array with shape (len(sequences), maxlen)
    This function transforms a list (of length num_samples) of sequences (lists of integers) into a 2D Numpy array of
    shape (num_samples, num_timesteps). num_timesteps is either the maxlen argument if provided, or the length of the
    longest sequence in the list.
    Sequences that are shorter than num_timesteps are padded with value until they are num_timesteps long.
    Sequences longer than num_timesteps are truncated so that they fit the desired length.
    The position where padding or truncation happens is determined by the arguments padding and truncating,
    respectively. Pre-padding or removing values from the beginning of the sequence is the default.
    - sequences:	List of sequences (each sequence is a list of integers).
    - maxlen:	    Optional Int, maximum length of all sequences. If not provided, sequences will be padded to the
                    length of the longest individual sequence.
    - dtype:	    (Optional, defaults to int32). Type of the output sequences. To pad sequences with variable length
                    strings, you can use object.
    - padding:	    String, 'pre' or 'post' (optional, defaults to 'pre'): pad either before or after each sequence.
    - truncating:	String, 'pre' or 'post' (optional, defaults to 'pre'): remove values from sequences larger than
                    maxlen, either at the beginning or at the end of the sequences.
    - value:	    Float or String, padding value. (Optional, defaults to 0.)
    """
    sequences = load_tokenized_captions()
    max_length = max_and_min_lengths_of_tokenized_captions()[0]
    padded_captions = pad_sequences(sequences, maxlen=max_length, padding='post')
    dump(padded_captions, open('Flickr30k/padded_captions.pkl', 'wb'))


def load_padded_tokenized_captions():
    return load(open('Flickr30k/padded_captions.pkl', 'rb'))


def create_dataframe_of_images_paths_and_their_corresponding_padded_tokenized_captions():
    Images_paths = load_limited_images_paths_and_captions()[0]
    padded_tokenized_captions = load_padded_tokenized_captions()
    d = {'Images paths': Images_paths, 'Padded tokenized caption': list(padded_tokenized_captions)}
    df = pd.DataFrame(data=d)
    df.to_pickle('Flickr30k/images_paths_and_their__padded_tokenized_captions.pkl')


def load_dataframe_of_images_paths_and_their_corresponding_padded_tokenized_captions():
    return pd.read_pickle('Flickr30k/images_paths_and_their__padded_tokenized_captions.pkl')


def image_path_by_its_padded_tokenized_caption(padded_tokenized_caption):
    df = load_dataframe_of_images_paths_and_their_corresponding_padded_tokenized_captions()
    for index, element in enumerate(df['Padded tokenized caption']):
        if (element == padded_tokenized_caption).all():
            return df['Images paths'].iloc[index]
    return 'passed features does not match with any features of the Flickr30k images.'


def splitting_paths_and_captions_to_train_and_test():
    all_images_paths = load_limited_images_paths_and_captions()[0]
    all_captions = load_padded_tokenized_captions()
    train_images_paths, test_images_paths, train_captions, test_captions = train_test_split(all_images_paths,
                                                                                            all_captions, test_size=0.2,
                                                                                            random_state=0)
    dump(train_images_paths, open('Flickr30k/train_images_paths.pkl', 'wb'))
    dump(test_images_paths, open('Flickr30k/test_images_paths.pkl', 'wb'))
    dump(train_captions, open('Flickr30k/train_captions.pkl', 'wb'))
    dump(test_captions, open('Flickr30k/test_captions.pkl', 'wb'))


def load_train_images_paths_and_train_captions():
    train_images_paths = load(open('Flickr30k/train_images_paths.pkl', 'rb'))
    train_captions = load(open('Flickr30k/train_captions.pkl', 'rb'))
    return train_images_paths, train_captions


def load_test_images_paths_and_test_captions():
    test_images_paths = load(open('Flickr30k/test_images_paths.pkl', 'rb'))
    test_captions = load(open('Flickr30k/test_captions.pkl', 'rb'))
    return test_images_paths, test_captions


def report_after_training_testing_split():  # 126976 for training, and 31744 for testing.
    train_images_paths = load(open('Flickr30k/train_images_paths.pkl', 'rb'))
    test_images_paths = load(open('Flickr30k/test_images_paths.pkl', 'rb'))
    print(f'Our Training Set has: {len(train_images_paths)} images, with their corresponding captions.')
    print(f'Our Testing Set has: {len(test_images_paths)} images, with their corresponding captions.')


def load_npy(img_path, caption):
    img_features_path = 'Extracted Features/' + img_path.decode('utf-8')
    img_features = np.load(img_features_path + '.npy')
    return img_features, caption


def create_training_dataset(batch_size=64, buffer_size=1000):
    train_images_paths, train_captions = load_train_images_paths_and_train_captions()
    """
    - tf.data.Dataset Represents a potentially large set of elements.
    - The simplest way to create a dataset is to create it from a python list with from_tensor_slices method.
    - tf.data.Dataset.as_numpy_iterator Returns an iterator which converts all elements of the dataset to numpy.
    - to pick an element from a dataset by its index:
        l = list(dataset.as_numpy_iterator) --> l[index]
    - the size of a dataset is computed by: len(list(dataset))
    """
    dataset = tf.data.Dataset.from_tensor_slices((train_images_paths, train_captions))
    # <TensorSliceDataset shapes: ((), (74,)), types: (tf.string, tf.int32)>  :
    # ((image_path) (padded_tokenized_caption))
    """
    * tf.numpy_function(func, inp, Tout, name=None) Wraps a python function and uses it as a TensorFlow op.
    Given a python function func, wrap this function as an operation in a TensorFlow function. func must take numpy 
    arrays as its arguments and return numpy arrays as its outputs.
    - func: 	
    A Python function, which accepts numpy.ndarray objects as arguments and returns a list of numpy.ndarray objects 
    - inp: A list of tf.Tensor objects.
    - Tout: A list or tuple of tensorflow data types or a single tensorflow data type if there is only one, indicating 
            what func returns.
    - name: (Optional) A name for the operation.
    """
    """
    * map(map_func, num_parallel_calls=None, deterministic=None) Maps map_func across the elements of this dataset. 
        - This transformation applies map_func to each element of this dataset, and returns a new dataset 
          containing the transformed elements, in the same order as they appeared in the input. map_func can be used to 
          change both the values and the structure of a dataset's elements. 
        - num_parallel_calls : 
          A tf.int32 scalar tf.Tensor, representing the number elements to process asynchronously in parallel. 
          If not specified, elements will be processed sequentially. If the value tf.data.AUTOTUNE is used, then 
          the number of parallel calls is set dynamically based on available CPU. 
        - deterministic:
          A boolean controlling whether determinism should be traded for performance by allowing elements to be produced 
          out of order. If deterministic is None, the tf.data.Options.experimental_deterministic dataset option 
          (True by default) is used to decide whether to produce elements deterministically
    """
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(load_npy, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # <TensorSliceDataset> ((image_features) (padded_tokenized_caption))
    """
    * shuffle(buffer_size, seed=None, reshuffle_each_iteration=None)
    Randomly shuffles the elements of this dataset.
    This dataset fills a buffer with buffer_size elements, then randomly samples elements from this buffer, replacing 
    the selected elements with new elements. For perfect shuffling, a buffer size greater than or equal to the full 
    size of the dataset is required.
    
    * batch(batch_size, drop_remainder=False) Combines consecutive elements of this dataset into batches.
        - drop_remainder :
          A tf.bool scalar tf.Tensor, representing whether the last batch should be dropped in the case it has fewer 
          than batch_size elements; the default behavior is not to drop the smaller batch.
    
    * prefetch(buffer_size) Creates a Dataset that prefetches elements from this dataset.
    Most dataset input pipelines should end with a call to prefetch. This allows later elements to be prepared while 
    the current element is being processed. This often improves latency and throughput, at the cost of using additional 
    memory to store prefetched elements.
    - buffer_size: 
        A tf.int64 scalar tf.Tensor, representing the maximum number of elements that will be buffered when prefetching.
    """
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def create_testing_dataset(batch_size=64, buffer_size=1000):
    test_images_paths, test_captions = load_test_images_paths_and_test_captions()
    dataset = tf.data.Dataset.from_tensor_slices((test_images_paths, test_captions))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(load_npy, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def splitting_paths_and_captions_to_train_and_test_stage2():
    all_images_paths = load_limited_images_paths_and_captions()[0]
    all_captions = load_padded_tokenized_captions()
    train_images_paths, test_images_paths, train_captions, test_captions = train_test_split(all_images_paths,
                                                                                            all_captions,
                                                                                            test_size=0.00126,
                                                                                            random_state=0)
    dump(train_images_paths, open('Flickr30k/train_images_paths_stage2.pkl', 'wb'))
    dump(test_images_paths, open('Flickr30k/test_images_paths_stage2.pkl', 'wb'))
    dump(train_captions, open('Flickr30k/train_captions_stage2.pkl', 'wb'))
    dump(test_captions, open('Flickr30k/test_captions_stage2.pkl', 'wb'))


def load_train_images_paths_and_train_captions_stage2():
    train_images_paths = load(open('Flickr30k/train_images_paths_stage2.pkl', 'rb'))
    train_captions = load(open('Flickr30k/train_captions_stage2.pkl', 'rb'))
    return train_images_paths, train_captions


def load_test_images_paths_and_test_captions_stage2():
    test_images_paths = load(open('Flickr30k/test_images_paths_stage2.pkl', 'rb'))
    test_captions = load(open('Flickr30k/test_captions_stage2.pkl', 'rb'))
    return test_images_paths, test_captions


def report_after_training_testing_split_stage2():  # 158520 for training, and 200 for testing.
    train_images_paths = load(open('Flickr30k/train_images_paths_stage2.pkl', 'rb'))
    test_images_paths = load(open('Flickr30k/test_images_paths_stage2.pkl', 'rb'))
    print(f'Our Training Set has: {len(train_images_paths)} images, with their corresponding captions.')
    print(f'Our Testing Set has: {len(test_images_paths)} images, with their corresponding captions.')


def create_training_dataset_stage2(batch_size=64, buffer_size=1000):
    train_images_paths, train_captions = load_train_images_paths_and_train_captions_stage2()
    dataset = tf.data.Dataset.from_tensor_slices((train_images_paths, train_captions))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(load_npy, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def create_testing_dataset_stage2(batch_size=64, buffer_size=1000):
    test_images_paths, test_captions = load_test_images_paths_and_test_captions_stage2()
    dataset = tf.data.Dataset.from_tensor_slices((test_images_paths, test_captions))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(load_npy, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def creating_dataframe_of_testImages_and_captions_stage2():
    paths = load_test_images_paths_and_test_captions_stage2()[0]
    data = load_captions_as_dataframe()
    images_names, images_captions = [], []
    for image_name in paths:
        image_name = image_name.split('/')[-1]
        for i in range(5):
            images_names.append(image_name)
        captions = list(data["caption"].loc[data["image_name"] == image_name])
        for caption in captions:
            images_captions.append(caption)
    df = pd.DataFrame({'image_name': images_names, 'caption': images_captions})
    dump(df, open('Flickr30k/dataframe_of_testing_images_stage2.pkl', 'wb'))


def load_dataframe_of_testImages_and_captions_stage2():
    df = load(open('Flickr30k/dataframe_of_testing_images_stage2.pkl', 'rb'))
    return df
