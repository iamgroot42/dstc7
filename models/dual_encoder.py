import tensorflow as tf
from models import helpers

FLAGS = tf.flags.FLAGS


def get_embeddings(hparams):
    if hparams.glove_path and hparams.vocab_path:
        tf.logging.info("Loading Glove embeddings...")
        vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
        glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.glove_path, vocab=set(vocab_array))
        initializer = helpers.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors,
                                                             hparams.embedding_dim)
    else:
        tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
        initializer = tf.random_uniform_initializer(-0.25, 0.25)

    if hparams.glove_path and hparams.vocab_path:
        return tf.get_variable(
            "word_embeddings",
            initializer=initializer)
    elif hparams.vocab_path:
        vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
        return tf.get_variable(
            "word_embeddings",
            shape=[len(vocab_dict), hparams.embedding_dim],
            initializer=initializer)
    else:
        return tf.get_variable(
            "word_embeddings",
            shape=[hparams.vocab_size, hparams.embedding_dim],
            initializer=initializer)


def make_cell(dimension, residual, dropout):
    cell = tf.nn.rnn_cell.LSTMCell(
            dimension,
            forget_bias=2.0,
            use_peepholes=True,
            state_is_tuple=True)
    if dropout > 0:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-dropout)
    if residual:
        cell = tf.nn.rnn_cell.ResidualWrapper(cell)
    return cell

def dual_encoder_model(
        hparams,
        mode,
        context,
        context_len,
        utterances,
        utterances_len,
        targets,
        batch_size,
        bidirectional=False,
        attention=False,
        feature_type="default"):
    # Initialize embeddings randomly or with pre-trained vectors if available
    embeddings_W = get_embeddings(hparams)

    # Embed the context and the utterance
    context_embedded = tf.nn.embedding_lookup(
        embeddings_W, context, name="embed_context")
    utterances_embedded = tf.nn.embedding_lookup(
        embeddings_W, utterances, name="embed_utterance")

    # Set dropout and residual options
    residual, dropout = False, 0

    # Calculate output embedding dimension
    M_dim = hparams.rnn_dim
    if bidirectional:
        M_dim *= 2

    # Build the Context Encoder RNN
    with tf.variable_scope("context-rnn") as vs:
        # We use an LSTM Cell
        cell_context = make_cell(hparams.rnn_dim, residual, dropout)

        if bidirectional:
            # Create cell for reverse direction
            cell_context_reverse = make_cell(hparams.rnn_dim, residual, dropout)

            (context_encoded_outputs_f, context_encoded_outputs_b), (context_encoded_f, context_encoded_b) = tf.nn.bidirectional_dynamic_rnn(cell_context, 
                                                                            cell_context_reverse, context_embedded,
                                                                            context_len, dtype=tf.float32)
            context_encoded_outputs = tf.concat([context_encoded_outputs_f, context_encoded_outputs_b], -1)
            context_encoded         = tf.concat([context_encoded_f, context_encoded_b], -1)
        else:
            # Run context through the RNN
            context_encoded_outputs, context_encoded = tf.nn.dynamic_rnn(cell_context, context_embedded,
                                                                            context_len, dtype=tf.float32)

        # Use last state / sum of states / maxpool of states
        if feature_type == "sum":
            context_encoded_feature = tf.reduce_sum(context_encoded_outputs, 1)
        elif feature_type == "mean":
            context_encoded_feature = tf.reduce_mean(context_encoded_outputs, 1)
        elif feature_type == "max":
            context_encoded_feature = tf.reduce_max(context_encoded_outputs, 1)
        elif feature_type == "cnn":
            context_encoded_feature = tf.reduce_max(tf.layers.conv1d(inputs=context_encoded_outputs,
                                                        filters=M_dim,
                                                        kernel_size=2,
                                                        padding='same',
                                                        activation=tf.tanh,
                                                        name='context_conv'), 1)
        else:
            context_encoded_feature = context_encoded[1]

        # Initialize attention weights of attention specified
        if attention:
            W_am = tf.get_variable("W_am",
                            shape=[M_dim, M_dim],
                            initializer=tf.truncated_normal_initializer())
            W_qm = tf.get_variable("W_qm",
                            shape=[M_dim, M_dim],
                            initializer=tf.truncated_normal_initializer())
            w_ms = tf.get_variable("w_ms",
                            shape=[M_dim, M_dim],
                            initializer=tf.truncated_normal_initializer())


    # Build the Utterance Encoder RNN
    with tf.variable_scope("utterance-rnn") as vs:
        # We use an LSTM Cell
        cell_utterance = make_cell(hparams.rnn_dim, residual, dropout)

        # Construct reverse-direction cell if bidirectional architecture specified
        if bidirectional:
            cell_utterance_reverse = make_cell(hparams.rnn_dim, residual, dropout)

        # Initialize CNN kernels if CNN-pooling specified
        if feature_type == "cnn":
            utterance_cnn = tf.keras.layers.Conv1D(filters=M_dim,
                                                    kernel_size=2,
                                                    padding='same',
                                                    activation=tf.tanh,
                                                    name='utterance_conv')

        # Run all utterances through the RNN batch by batch
        # TODO: Needs to be parallelized
        all_utterances_encoded = []
        for i in range(batch_size):
            if bidirectional:
                (temp_outputs_f, temp_outputs_b), (temp_states_f, temp_states_b) = tf.nn.bidirectional_dynamic_rnn(cell_utterance,
                                                                            cell_utterance_reverse, utterances_embedded[:,i],
                                                                            utterances_len[i], dtype=tf.float32)
                temp_outputs = tf.concat([temp_outputs_f, temp_outputs_b], -1)
                temp_states  = tf.concat([temp_states_f, temp_states_b], -1)
            else:
                temp_outputs, temp_states = tf.nn.dynamic_rnn(cell_utterance, utterances_embedded[:,i],
                                                            utterances_len[i], dtype=tf.float32)

            # Modify using attention (if specified):
            if attention:
                assert feature_type in ["sum", "mean", "max", "cnn"] 
                m_aq = tf.tanh(tf.add(tf.map_fn(lambda x: tf.matmul(x, W_am), temp_outputs),
                    tf.matmul(tf.expand_dims(context_encoded_feature[i], 0), W_qm)))
                s_aq = tf.nn.softmax(tf.map_fn(lambda x: tf.matmul(x, w_ms), m_aq))
                temp_outputs = tf.multiply(temp_outputs, s_aq)

            # Use last state / sum of states / maxpool of states
            if feature_type == "sum":
                utterance_encoded_feature = tf.reduce_sum(temp_outputs, 1)
            elif feature_type == "mean":
                utterance_encoded_feature = tf.reduce_mean(temp_outputs, 1)
            elif feature_type == "max":
                utterance_encoded_feature = tf.reduce_max(temp_outputs, 1)
            elif feature_type == "cnn":
                utterance_encoded_feature = tf.reduce_max(utterance_cnn(temp_outputs), 1)
            else:
                utterance_encoded_feature = temp_states[1]

            all_utterances_encoded.append(utterance_encoded_feature) # since it's a tuple, use the hidden states

        all_utterances_encoded = tf.stack(all_utterances_encoded, axis=0)

    with tf.variable_scope("prediction") as vs:
        M = tf.get_variable("M",
                            shape=[M_dim, M_dim],
                            initializer=tf.truncated_normal_initializer())
        
        # "Predict" a  response: c * M
        generated_response = tf.matmul(context_encoded_feature, M) #[1], M) # using the hidden states
        generated_response = tf.expand_dims(generated_response, 1)
        all_utterances_encoded = tf.transpose(all_utterances_encoded, perm=[0, 2, 1]) # transpose last two dimensions

        # Dot product between generated response and actual response
        # (c * M) * r
        logits = tf.matmul(generated_response, all_utterances_encoded)
        logits = tf.squeeze(logits, [1])

        # Apply sigmoid to convert logits to probabilities
        probs = tf.nn.softmax(logits)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return probs, None

        # Calculate the binary cross-entropy loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(targets))

    # Mean loss across the batch of examples
    mean_loss = tf.reduce_mean(losses, name="mean_loss")
    return probs, mean_loss
