import tensorflow as tf
from collections import namedtuple

# Model Parameters
tf.flags.DEFINE_integer(
    "vocab_size",
    100000,
    "The size of the vocabulary. Only change this if you changed the preprocessing")

# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer("rnn_dim", 256, "Dimensionality of the RNN cell")
tf.flags.DEFINE_integer("max_context_len", 160, "Truncate contexts to this length")
tf.flags.DEFINE_integer("max_utterance_len", 160, "Truncate utterance to this length")

# Pre-trained embeddings
tf.flags.DEFINE_string("glove_path", None, "Path to pre-trained Glove vectors")
tf.flags.DEFINE_string("vocab_path", None, "Path to vocabulary.txt file")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_float("keep_rate", 1.0, "Drop out probability")
tf.flags.DEFINE_float("decay_rate", 0.65, "Exponential decay rate")
tf.flags.DEFINE_integer("factorization", -1, "Dimension for matrix multiplication factorization at last step")
tf.flags.DEFINE_integer("batch_size", 16, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 32, "Batch size during evaluation")
tf.flags.DEFINE_integer("decay_steps", 5000, "Decay steps")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")
tf.flags.DEFINE_string("feature_type", "default", "Use last state(default), mean of states(mean), \
    sum of states(sum), cnn(cnn) or maxpool(max)?")
tf.flags.DEFINE_bool("staircase", False, "Staircase decay")
tf.flags.DEFINE_bool("bidirectional", False, "Use bidirectional LSTM?")
tf.flags.DEFINE_bool("residual", False, "Use residual connections?")
tf.flags.DEFINE_bool("fastgrnn", False, "Use FastGRNN cell instead of LSTM?")
tf.flags.DEFINE_bool("attention", False, "Use attention for alignment?")
tf.flags.DEFINE_bool("dssm", False, "Use DSSM features?")
tf.flags.DEFINE_bool("tfidf", False, "Use TF-IDF features?")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
    "HParams",
    [
        "batch_size",
        "embedding_dim",
        "eval_batch_size",
        "learning_rate",
        "max_context_len",
        "max_utterance_len",
        "optimizer",
        "rnn_dim",
        "vocab_size",
        "glove_path",
        "vocab_path",
        "keep_rate",
        "decay_rate",
        "decay_steps",
        "staircase",
        "bidirectional",
        "attention",
        "feature_type",
        "dssm",
        "tfidf",
        "residual",
        "fastgrnn",
        "factorization",
    ])


def create_hparams():
    return HParams(
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        vocab_size=FLAGS.vocab_size,
        optimizer=FLAGS.optimizer,
        learning_rate=FLAGS.learning_rate,
        embedding_dim=FLAGS.embedding_dim,
        max_context_len=FLAGS.max_context_len,
        max_utterance_len=FLAGS.max_utterance_len,
        glove_path=FLAGS.glove_path,
        vocab_path=FLAGS.vocab_path,
        rnn_dim=FLAGS.rnn_dim,
        keep_rate=FLAGS.keep_rate,
        decay_rate=FLAGS.decay_rate,
        decay_steps=FLAGS.decay_steps,
        staircase=FLAGS.staircase,
        bidirectional=FLAGS.bidirectional,
        attention=FLAGS.attention,
        feature_type=FLAGS.feature_type,
        dssm=FLAGS.dssm,
        tfidf=FLAGS.dssm,
        residual=FLAGS.residual,
        fastgrnn=FLAGS.fastgrnn,
        factorization=FLAGS.factorization,
    )
