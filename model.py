import tensorflow as tf


def get_id_feature(features, key, len_key, max_len, dssm_key, lcs_key=None):
    ids = features[key]
    ids_len = tf.squeeze(features[len_key], [1])
    ids_len = tf.minimum(ids_len, tf.constant(max_len, dtype=tf.int64))
    return ids, ids_len, features.get(dssm_key, None), features.get(lcs_key, None)


def create_train_op(loss, hparams):
    def exp_decay(learning_rate, global_step):
        return tf.train.exponential_decay(learning_rate, global_step, decay_steps=hparams.decay_steps, decay_rate=hparams.decay_rate,
                                          staircase=hparams.staircase, name="lr_decay")
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=hparams.learning_rate,
        clip_gradients=3.0,
        optimizer=hparams.optimizer,
        learning_rate_decay_fn=exp_decay
    )
    return train_op


def create_model_fn(hparams, model_impl):
    def model_fn(features, targets, mode):
        context, context_len, context_dssm, _ = get_id_feature(
            features, "context", "context_len", hparams.max_context_len, "context_dssm")

        all_utterances = []
        all_utterances_lens = []
        all_utterances_dssm = []
        all_utterances_lcs = []

        for i in range(100):
            option, option_len, option_dssm, option_lcs = get_id_feature(features,
                                                "option_{}".format(i),
                                                "option_{}_len".format(i),
                                                hparams.max_utterance_len,
                                                "option_{}_dssm".format(i),
                                                "option_{}_lcs".format(i),)
            all_utterances.append(option)
            all_utterances_lens.append(option_len)
            all_utterances_dssm.append(option_dssm)
            all_utterances_lcs.append(option_lcs)

        dssm_object = None
        if hparams.dssm:
            dssm_object = (context_dssm, all_utterances_dssm)
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            probs, loss = model_impl(
                hparams,
                mode,
                context,
                context_len,
                all_utterances,
                tf.transpose(tf.stack(all_utterances_lens, axis=0)),
                targets,
                hparams.batch_size,
                hparams.bidirectional,
                hparams.attention,
                hparams.feature_type,
                dssm_object,
                all_utterances_lcs,)
            train_op = create_train_op(loss, hparams)
            return probs, loss, train_op

        if mode == tf.contrib.learn.ModeKeys.INFER:
            probs, loss = model_impl(
                hparams,
                mode,
                context,
                context_len,
                all_utterances,
                tf.transpose(tf.stack(all_utterances_lens, axis=0)),
                None,
                hparams.eval_batch_size,
                hparams.bidirectional,
                hparams.attention,
                hparams.feature_type,
                dssm_object,
                all_utterances_lcs,)

            return probs, 0.0, None

        if mode == tf.contrib.learn.ModeKeys.EVAL:
            probs, loss = model_impl(
                hparams,
                mode,
                context,
                context_len,
                all_utterances,
                tf.transpose(tf.stack(all_utterances_lens, axis=0)),
                targets,
                hparams.eval_batch_size,
                hparams.bidirectional,
                hparams.attention,
                hparams.feature_type,
                dssm_object,
                all_utterances_lcs,)

            shaped_probs = probs

            return shaped_probs, loss, None

    return model_fn
