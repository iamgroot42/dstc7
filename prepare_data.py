import os
import re

import ijson
import functools
from tweetmotif.twokenize import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import _pickle as cPickle
import tensorflow as tf

tf.flags.DEFINE_integer(
    "min_word_frequency", 1, "Minimum frequency of words in the vocabulary")

tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")

tf.flags.DEFINE_string("train_in", None, "Path to input data file")
tf.flags.DEFINE_string("validation_in", None, "Path to validation data file")
tf.flags.DEFINE_string("tfidf_in", None, "Path to all utterances file")

tf.flags.DEFINE_string("train_out", None, "Path to output train tfrecords file")
tf.flags.DEFINE_string("validation_out", None, "Path to output validation tfrecords file")
tf.flags.DEFINE_string("tfidf_out", None, "Path to output TFIDF vectorizer")

tf.flags.DEFINE_string("vocab_path", None, "Path to save vocabulary txt file")
tf.flags.DEFINE_string("vocab_processor", None, "Path to save vocabulary processor")
tf.flags.DEFINE_boolean("has_dssm", False, "Does dataset have DSSM features?")
tf.flags.DEFINE_boolean("use_tfidf", False, "Add TF-IDF features while constructing data?")

FLAGS = tf.flags.FLAGS

TRAIN_PATH = os.path.join(FLAGS.train_in)
VALIDATION_PATH = os.path.join(FLAGS.validation_in)

def read_all_lines(path):
    lines = []
    replace_urls = lambda text: re.sub(r'^https?:\/\/.*[\r\n]*', '__URL__', text, flags=re.MULTILINE)
    with open(path, 'r') as f:
        for line in f:
            lines.append(replace_urls(line.rstrip()))
    return lines


def tokenizer_fn(iterator):
    return (x.split(" ") for x in iterator)


def process_dialog(dialog):
    """
    Add EOU and EOT tags between utterances and create a single context string.
    :param dialog:
    :return:
    """

    row = []
    dssm_row = []
    utterances = dialog['messages-so-far']

    # Create the context
    context = ""
    speaker = None
    for msg in utterances:
        if speaker is None:
            context += ' '.join(tokenize(msg['utterance'])) + " __eou__ "
            speaker = msg['speaker']
        elif speaker != msg['speaker']:
            context += "__eot__ " + ' '.join(tokenize(msg['utterance'])) + " __eou__ "
            speaker = msg['speaker']
        else:
            context += ' '.join(tokenize(msg['utterance'])) + " __eou__ "

    context += "__eot__"
    row.append(context)

    if FLAGS.has_dssm:
        dssm_row.append(dialog['dssm'])

    # Create the next utterance options and the target label
    correct_answer = dialog['options-for-correct-answers'][0]
    target_id = correct_answer['candidate-id']
    target_index = None
    for i, utterance in enumerate(dialog['options-for-next']):
        if utterance['candidate-id'] == target_id:
            target_index = i
        row.append(' '.join(tokenize(utterance['utterance'])) + " __eou__ ")
        if FLAGS.has_dssm:
            dssm_row.append(utterance['dssm'])

    if target_index is None:
        print('Correct answer not found in options-for-next - example {}. Setting 0 as the correct index'.format(dialog['example-id']))
    else:
        row.append(target_index)

    return (row, dssm_row)


def create_dialog_iter(filename):
    """
    Returns an iterator over a JSON file.
    :param filename:
    :return:
    """
    with open(filename, 'rb') as f:
        json_data = ijson.items(f, 'item')
        for entry in json_data:
            row = process_dialog(entry)
            yield row

def create_utterance_iter(input_iter):
    """
    Returns an iterator over every utterance (context and candidates) for the VocabularyProcessor.
    :param input_iter:
    :return:
    """
    for row in input_iter:
        all_utterances = []
        replace_urls = lambda text: re.sub(r'^https?:\/\/.*[\r\n]*', '__URL__', text, flags=re.MULTILINE)
        context = replace_urls(row[0][0])
        next_utterances = [replace_urls(x) for x in row[0][1:101]]
        all_utterances.append(context)
        all_utterances.extend(next_utterances)
        for utterance in all_utterances:
            yield utterance

def create_vocab(input_iter, min_frequency):
    """
    Creates and returns a VocabularyProcessor object with the vocabulary
    for the input iterator.
    """
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        FLAGS.max_sentence_len,
        min_frequency=min_frequency,
        tokenizer_fn=tokenizer_fn)
    vocab_processor.fit(input_iter)
    return vocab_processor


def transform_sentence(sequence, vocab_processor):
    """
    Maps a single sentence into the integer vocabulary. Returns a python array.
    """
    return next(vocab_processor.transform([sequence])).tolist()


def create_example_new_format(row, vocab, vectorizer=None):
    """
    Creates an example as a tensorflow.Example Protocol Buffer object.
    :param row:
    :param vocab:
    :return:
    """
    if FLAGS.has_dssm:
        row, row_dssm = row
    else:
        row, _ = row
    context = row[0]
    next_utterances = row[1:101]
    target = row[-1]

    context_transformed = transform_sentence(context, vocab)
    context_len = len(next(vocab._tokenizer([context])))

    # New Example
    example = tf.train.Example()
    example.features.feature["context"].int64_list.value.extend(context_transformed)
    example.features.feature["context_len"].int64_list.value.extend([context_len])
    example.features.feature["target"].int64_list.value.extend([target])
    if FLAGS.has_dssm:
#        example.features.feature["context_dssm"].bytes_list.value.extend([row_dssm[0].encode('utf-8')])
        example.features.feature["context_dssm"].float_list.value.extend([float(x) for x in row_dssm[0].split(',')])
    if vectorizer:
        context_features = vectorizer.transform([context]).todense()
        print(context_features.shape)
        example.features.feature["context_tfidf"].float_list.value.extend(context_features)

    # Distractor sequences
    for i, utterance in enumerate(next_utterances):
        opt_key = "option_{}".format(i)
        opt_len_key = "option_{}_len".format(i)
        opt_dssm_key = "option_{}_dssm".format(i)
        # Utterance Length Feature
        opt_len = len(next(vocab._tokenizer([utterance])))
        example.features.feature[opt_len_key].int64_list.value.extend([opt_len])
        # Distractor Text Feature
        opt_transformed = transform_sentence(utterance, vocab)
        example.features.feature[opt_key].int64_list.value.extend(opt_transformed)
        if FLAGS.has_dssm:
#            example.features.feature[opt_dssm_key].bytes_list.value.extend([row_dssm[i+1].encode('utf-8')])
            example.features.feature[opt_dssm_key].float_list.value.extend([float(x) for x in row_dssm[i+1].split(',')])
    return example


def create_tfrecords_file(input_filename, output_filename, example_fn):
    """
    Creates a TFRecords file for the given input data and
    example transofmration function
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    print("Creating TFRecords file at {}...".format(output_filename))
    for i, row in enumerate(create_dialog_iter(input_filename)):
        x = example_fn(row)
        writer.write(x.SerializeToString())
    writer.close()
    print("Wrote to {}".format(output_filename))


def write_vocabulary(vocab_processor, outfile):
    """
    Writes the vocabulary to a file, one word per line.
    """
    vocab_size = len(vocab_processor.vocabulary_)
    with open(outfile, "w") as vocabfile:
        for id in range(vocab_size):
            word =  vocab_processor.vocabulary_._reverse_mapping[id]
            vocabfile.write(word + "\n")
    print("Saved vocabulary to {}".format(outfile))


if __name__ == "__main__":
    vectorizer = None
    if FLAGS.use_tfidf:
        assert FLAGS.tfidf_in is not None and FLAGS.tfidf_out is not None
        print("Creating TF-IDF vectorizer")
        lines = read_all_lines(FLAGS.tfidf_in)
        vectorizer = TfidfVectorizer(input='content', ngram_range=(1,3), stop_words ='english', sublinear_tf=True, max_features=10000,)
        with open(FLAGS.tfidf_out, 'wb') as tfidfFile:
            cPickle.dump(vectorizer, tfidfFile)
    print("Creating vocabulary...")
    input_iter = create_dialog_iter(TRAIN_PATH)
    input_iter = create_utterance_iter(input_iter)
    vocab = create_vocab(input_iter, min_frequency=FLAGS.min_word_frequency)
    print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))

    # Create vocabulary.txt file
    write_vocabulary(
        vocab, os.path.join(FLAGS.vocab_path))

    # Save vocab processor
    vocab.save(os.path.join(FLAGS.vocab_processor))

    # Create train.tfrecords
    create_tfrecords_file(
        input_filename=TRAIN_PATH,
        output_filename=os.path.join(FLAGS.train_out),
        example_fn=functools.partial(create_example_new_format, vocab=vocab, vectorizer=vectorizer))

    # Create validation.tfrecords
    create_tfrecords_file(
        input_filename=VALIDATION_PATH,
        output_filename=os.path.join(FLAGS.validation_out),
        example_fn=functools.partial(create_example_new_format, vocab=vocab, vectorizer=vectorizer))
