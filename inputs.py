import tensorflow as tf

TEXT_FEATURE_SIZE = 160
CDSSM_DIM = 300

def get_feature_columns(mode, has_dssm, has_lcs):
    feature_columns = []

    feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="context", dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
    feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="context_len", dimension=1, dtype=tf.int64))

    feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="target", dimension=1, dtype=tf.int64))

    if has_dssm:
        feature_columns.append(tf.contrib.layers.real_valued_column(
            column_name="context_dssm", dimension=CDSSM_DIM, dtype=tf.float32))
    for i in range(100):
        feature_columns.append(tf.contrib.layers.real_valued_column(
            column_name="option_{}".format(i), dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
        feature_columns.append(tf.contrib.layers.real_valued_column(
            column_name="option_{}_len".format(i), dimension=1, dtype=tf.int64))
        if has_dssm:
            feature_columns.append(tf.contrib.layers.real_valued_column(
                column_name="option_{}_dssm".format(i), dimension=CDSSM_DIM, dtype=tf.float32))
        if has_lcs:
            feature_columns.append(tf.contrib.layers.real_valued_column(
                column_name="option_{}_lcs".format(i), dimension=3, dtype=tf.float32))
    return set(feature_columns)


def create_input_fn(mode, input_files, batch_size, num_epochs, has_dssm, has_lcs, randomize=True):
    def input_fn():
        features = tf.contrib.layers.create_feature_spec_for_parsing(
            get_feature_columns(mode, has_dssm, has_lcs))

        feature_map = tf.contrib.learn.io.read_batch_features(
            file_pattern=input_files,
            batch_size=batch_size,
            features=features,
            reader=tf.TFRecordReader,
            randomize_input=randomize,
            num_epochs=num_epochs,
            queue_capacity=200000 + batch_size * 10,
            name="read_batch_features_{}".format(mode))

        # This is an ugly hack because of a current bug in tf.learn
        # During evaluation TF tries to restore the epoch variable which isn't defined during training
        # So we define the variable manually here
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            tf.get_variable(
                "read_batch_features_eval/file_name_queue/limit_epochs/epochs",
                initializer=tf.constant(0, dtype=tf.int64))

        target = feature_map.pop("target")

        return feature_map, target

    return input_fn
