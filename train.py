import os
import time

import tensorflow as tf
import model
from hparams import create_hparams
import metrics
import inputs

from models.dual_encoder import dual_encoder_model


tf.flags.DEFINE_string("train_in", None, "Path to input data file")
tf.flags.DEFINE_string("validation_in", None, "Path to validation data file")
tf.flags.DEFINE_string("test_in", None, "Path to test data file")
tf.flags.DEFINE_string("test_out", "./predictions.txt", "Path to test data file")

tf.flags.DEFINE_string("model_dir", None, "Directory to store model checkpoints")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 200, "Evaluate after this many train steps")
tf.flags.DEFINE_boolean("infer_mode", False, "Inference mode while generating data?")

FLAGS = tf.flags.FLAGS

TIMESTAMP = int(time.time())

if FLAGS.model_dir:
    MODEL_DIR = FLAGS.model_dir
else:
    print("No model directory specified!")
    exit()

if FLAGS.infer_mode and not FLAGS.test_in:
    print("Test mode specified, but test data directory not given!")
    exit()

if FLAGS.infer_mode:
    TEST_FILE = os.path.abspath(os.path.join(FLAGS.test_in))
else:
    TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.train_in))
    VALIDATION_FILE = os.path.abspath(os.path.join(FLAGS.validation_in))

tf.logging.set_verbosity(FLAGS.loglevel)


def main(unused_argv):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    hyper_params = create_hparams()
    print("\n\nModel hyperparameters", hyper_params)

    model_fn = model.create_model_fn(
        hyper_params,
        model_impl=dual_encoder_model)

    estimator = tf.contrib.learn.Estimator(
        model_fn=model_fn,
        model_dir=MODEL_DIR,
        config=tf.contrib.learn.RunConfig(session_config=config))

    # Training mode
    if not FLAGS.infer_mode:
        input_fn_train = inputs.create_input_fn(
            mode=tf.contrib.learn.ModeKeys.TRAIN,
            input_files=[TRAIN_FILE],
            batch_size=hyper_params.batch_size,
            num_epochs=FLAGS.num_epochs,
            has_dssm=hyper_params.dssm,
            has_lcs=hyper_params.lcs,)

        input_fn_eval = inputs.create_input_fn(
            mode=tf.contrib.learn.ModeKeys.EVAL,
            input_files=[VALIDATION_FILE],
            batch_size=hyper_params.eval_batch_size,
            num_epochs=1,
            has_dssm=hyper_params.dssm,
            has_lcs=hyper_params.lcs,)

        eval_metrics = metrics.create_evaluation_metrics()

        eval_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=input_fn_eval,
            every_n_steps=FLAGS.eval_every,
            metrics=eval_metrics,)

        estimator.fit(input_fn=input_fn_train, steps=None, monitors=[eval_monitor])
    # Testing mode
    else:
        input_fn_infer = inputs.create_input_fn(
            mode=tf.contrib.learn.ModeKeys.INFER,
            input_files=[TEST_FILE],
            batch_size=hyper_params.eval_batch_size,
            num_epochs=1,
            has_dssm=hyper_params.dssm,
            has_lcs=hyper_params.lcs,
            randomize=False)

        preds = estimator.predict(input_fn=input_fn_infer)
        i = 0
        with open(FLAGS.test_out, 'w') as f:
            for pred in preds:
                i += 1
                output_string = ",".join([("%.15f" % indi) for indi in pred])
                f.write(output_string + "\n")
                print(i)


if __name__ == "__main__":
    tf.app.run()
