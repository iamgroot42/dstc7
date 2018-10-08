import tensorflow as tf
from models import helpers

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops.rnn_cell_impl import RNNCell

def gen_non_linearity(A, non_linearity):
    '''
    Returns required activation for a tensor based on the inputs
    '''
    if non_linearity == "tanh":
        return math_ops.tanh(A)
    elif non_linearity == "sigmoid":
        return math_ops.sigmoid(A)
    elif non_linearity == "relu":
        return gen_math_ops.maximum(A, 0.0)
    elif non_linearity == "quantTanh":
        return gen_math_ops.maximum(gen_math_ops.minimum(A, 1.0), -1.0)
    elif non_linearity == "quantSigm":
        A = (A + 1.0) / 2.0
        return gen_math_ops.maximum(gen_math_ops.minimum(A, 1.0), 0.0)
    else:
        return math_ops.tanh(A)


class FastGRNNCell(RNNCell):
    '''
    FastGRNN Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    gate_non_linearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_non_linearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of W matrix (creates two matrices if not None)
    uRank = rank of U matrix (creates two matrices if not None)
    zetaInit = init for zeta, the scale param
    nuInit = init for nu, the translation param

    FastGRNN architecture and compression techniques are found in
    FastGRNN(LINK) paper

    Basic architecture is like:

    z_t = gate_nl(Wx_t + Uh_{t-1} + B_g)
    h_t^ = update_nl(Wx_t + Uh_{t-1} + B_h)
    h_t = z_t*h_{t-1} + (sigmoid(zeta)(1-z_t) + sigmoid(nu))*h_t^

    W and U can further parameterised into low rank version by
    W = matmul(W_1, W_2) and U = matmul(U_1, U_2)
    '''

    def __init__(self, hidden_size, gate_non_linearity="sigmoid",
                 update_non_linearity="tanh", wRank=None, uRank=None,
                 zetaInit=1.0, nuInit=-4.0, name="FastGRNN"):
        super(FastGRNNCell, self).__init__()
        self._hidden_size = hidden_size
        self._gate_non_linearity = gate_non_linearity
        self._update_non_linearity = update_non_linearity
        self._num_weight_matrices = [1, 1]
        self._wRank = wRank
        self._uRank = uRank
        self._zetaInit = zetaInit
        self._nuInit = nuInit
        if wRank is not None:
            self._num_weight_matrices[0] += 1
        if uRank is not None:
            self._num_weight_matrices[1] += 1
        self._name = name

    @property
    def state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_non_linearity(self):
        return self._gate_non_linearity

    @property
    def update_non_linearity(self):
        return self._update_non_linearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "FastGRNN"

    def call(self, inputs, state):
        with vs.variable_scope(self._name + "/FastGRNNcell"):

            if self._wRank is None:
                W_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W = vs.get_variable(
                    "W", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W_matrix_init)
                wComp = math_ops.matmul(inputs, self.W)
            else:
                W_matrix_1_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W1 = vs.get_variable(
                    "W1", [inputs.get_shape()[-1], self._wRank],
                    initializer=W_matrix_1_init)
                W_matrix_2_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W2 = vs.get_variable(
                    "W2", [self._wRank, self._hidden_size],
                    initializer=W_matrix_2_init)
                wComp = math_ops.matmul(
                    math_ops.matmul(inputs, self.W1), self.W2)

            if self._uRank is None:
                U_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U = vs.get_variable(
                    "U", [self._hidden_size, self._hidden_size],
                    initializer=U_matrix_init)
                uComp = math_ops.matmul(state, self.U)
            else:
                U_matrix_1_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U1 = vs.get_variable(
                    "U1", [self._hidden_size, self._uRank],
                    initializer=U_matrix_1_init)
                U_matrix_2_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U2 = vs.get_variable(
                    "U2", [self._uRank, self._hidden_size],
                    initializer=U_matrix_2_init)
                uComp = math_ops.matmul(
                    math_ops.matmul(state, self.U1), self.U2)
            # Init zeta to 6.0 and nu to -6.0 if this doesn't give good
            # results. The inits are hyper-params.
            zeta_init = init_ops.constant_initializer(
                self._zetaInit, dtype=tf.float32)
            self.zeta = vs.get_variable("zeta", [1, 1], initializer=zeta_init)

            nu_init = init_ops.constant_initializer(
                self._nuInit, dtype=tf.float32)
            self.nu = vs.get_variable("nu", [1, 1], initializer=nu_init)

            pre_comp = wComp + uComp

            bias_gate_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_gate = vs.get_variable(
                "B_g", [1, self._hidden_size], initializer=bias_gate_init)
            z = gen_non_linearity(pre_comp + self.bias_gate,
                                  self._gate_non_linearity)

            bias_update_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_update = vs.get_variable(
                "B_h", [1, self._hidden_size], initializer=bias_update_init)
            c = gen_non_linearity(
                pre_comp + self.bias_update, self._update_non_linearity)

            new_h = z * state + (math_ops.sigmoid(self.zeta) * (1.0 - z) +
                                 math_ops.sigmoid(self.nu)) * c
        return new_h, new_h

    def getVars(self):
        Vars = []
        if self._num_weight_matrices[0] == 1:
            Vars.append(self.W)
        else:
            Vars.extend([self.W1, self.W2])

        if self._num_weight_matrices[1] == 1:
            Vars.append(self.U)
        else:
            Vars.extend([self.U1, self.U2])

        Vars.extend([self.bias_gate, self.bias_update])
        Vars.extend([self.zeta, self.nu])

        return Vars



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


def make_cell(dimension, residual, keep_prob, fastGRNN):
    if fastGRNN:
        cell = FastGRNNCell(dimension)
    else:
        cell = tf.nn.rnn_cell.LSTMCell(
                dimension,
                forget_bias=2.0,
                use_peepholes=True,
                state_is_tuple=True)
    if keep_prob < 1:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
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
        feature_type="default",
        dssm=None,
        lcs=None,):
    # Initialize embeddings randomly or with pre-trained vectors if available
    embeddings_W = get_embeddings(hparams)

    # Embed the context and the utterance
    context_embedded = tf.nn.embedding_lookup(
        embeddings_W, context, name="embed_context")
    utterances_embedded = tf.nn.embedding_lookup(
        embeddings_W, utterances, name="embed_utterance")

    # Calculate output embedding dimension
    M_dim = hparams.rnn_dim
    if bidirectional:
        M_dim *= 2

    # Extract DSSM features if given
    if dssm:
        (context_dssm, utterances_dssm) = dssm

    # Build the Context Encoder RNN
    with tf.variable_scope("context-rnn") as vs:
        # We use an LSTM Cell
        cell_context = make_cell(hparams.rnn_dim, hparams.residual, hparams.keep_rate, hparams.fastgrnn)

        if bidirectional:
            # Create cell for reverse direction
            cell_context_reverse = make_cell(hparams.rnn_dim, hparams.residual, hparams.keep_rate, hparams.fastgrnn)

            (context_encoded_outputs_f, context_encoded_outputs_b), (context_encoded_f, context_encoded_b) = tf.nn.bidirectional_dynamic_rnn(cell_context,
                                                                            cell_context_reverse, context_embedded,
                                                                            context_len, dtype=tf.float32)
            context_encoded_outputs = tf.concat([context_encoded_outputs_f, context_encoded_outputs_b], -1)
            context_encoded         = tf.concat([context_encoded_f, context_encoded_b], -1)
        else:
            # Run context through the RNN
            context_encoded_outputs, context_encoded = tf.nn.dynamic_rnn(cell_context, context_embedded,
                                                                            context_len, dtype=tf.float32)

        # Use last state / mean of states / maxpool of states
        if feature_type == "mean":
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
        cell_utterance = make_cell(hparams.rnn_dim, hparams.residual, hparams.keep_rate, hparams.fastgrnn)

        # Construct reverse-direction cell if bidirectional architecture specified
        if bidirectional:
            cell_utterance_reverse = make_cell(hparams.rnn_dim, hparams.residual, hparams.keep_rate, hparams.fastgrnn)

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
                assert feature_type in ["mean", "max", "cnn"]
                m_aq = tf.tanh(tf.add(tf.map_fn(lambda x: tf.matmul(x, W_am), temp_outputs),
                    tf.matmul(tf.expand_dims(context_encoded_feature[i], 0), W_qm)))
                s_aq = tf.nn.softmax(tf.map_fn(lambda x: tf.matmul(x, w_ms), m_aq))
                temp_outputs = tf.multiply(temp_outputs, s_aq)

            # Use last state / mean of states / maxpool of states
            if feature_type == "mean":
                utterance_encoded_feature = tf.reduce_mean(temp_outputs, 1)
            elif feature_type == "max":
                utterance_encoded_feature = tf.reduce_max(temp_outputs, 1)
            elif feature_type == "cnn":
                utterance_encoded_feature = tf.reduce_max(utterance_cnn(temp_outputs), 1)
            else:
                utterance_encoded_feature = temp_states[1]

            # Use DSSM features if available
            if dssm:
                utterances_dssm_use = tf.stack([utterances_dssm[j][i] for j in range(utterance_encoded_feature.shape[0])])
                utterance_encoded_feature = tf.concat([utterances_dssm_use, utterance_encoded_feature], -1)

            all_utterances_encoded.append(utterance_encoded_feature) # since it's a tuple, use the hidden states

        all_utterances_encoded = tf.stack(all_utterances_encoded, axis=0)

    with tf.variable_scope("prediction") as vs:
        if hparams.factorization != -1:
            M1 = tf.get_variable("M_1",
                                shape=[M_dim, hparams.factorization],
                                initializer=tf.truncated_normal_initializer())
            M2 = tf.get_variable("M_2",
                                shape=[hparams.factorization, M_dim],
                                initializer=tf.truncated_normal_initializer())
            # "Predict" an intermediate response: c * M1
            intermediate_generated_response = tf.matmul(context_encoded_feature, M1)
            # "Predict" a  response: (c * M1) * M2
            generated_response = tf.matmul(intermediate_generated_response, M2)
        else:
            # Concat DSSM embedding if available
            if dssm:
                context_encoded_feature = tf.concat([context_dssm, context_encoded_feature], -1)
                M_dim += context_dssm.shape[1]

            # Create matrix M
            M = tf.get_variable("M",
                                shape=[M_dim, M_dim],
                                initializer=tf.truncated_normal_initializer())

            # "Predict" a  response: c * M
            generated_response = tf.matmul(context_encoded_feature, M)

        generated_response = tf.expand_dims(generated_response, 1)
        all_utterances_encoded = tf.transpose(all_utterances_encoded, perm=[0, 2, 1]) # transpose last two dimensions

        # Dot product between generated response and actual response
        # (c * M) * r
        logits = tf.matmul(generated_response, all_utterances_encoded)
        logits = tf.squeeze(logits, [1])

        if lcs:
            lcs_combined = tf.concat([tf.stack(lcs, axis=1), tf.expand_dims(logits, -1)], -1)
            logits = tf.squeeze(tf.layers.dense(lcs_combined, 1, name="logits_with_lcs"), -1)

        # Apply sigmoid to convert logits to probabilities
        probs = tf.nn.softmax(logits)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return probs, None

        # Calculate the binary cross-entropy loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(targets))

    # Mean loss across the batch of examples
    mean_loss = tf.reduce_mean(losses, name="mean_loss")
    return probs, mean_loss
