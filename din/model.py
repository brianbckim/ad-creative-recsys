import math

import tensorflow as _tf

tf = _tf.compat.v1
tf.disable_v2_behavior()
keras = _tf.keras

MODEL_VERSION = 'hybrid-din-v2'


class Model(object):

  def __init__(self, user_count, item_count,
               cate_count, cate_list,
               theme_count, theme_list,
               predict_batch_size, predict_ads_num):

    del predict_batch_size  # Candidate batches are dynamic in the graph.

    self.u = tf.placeholder(tf.int32, [None,], name='user_id')
    self.i = tf.placeholder(tf.int32, [None,], name='item_id')
    self.j = tf.placeholder(tf.int32, [None,], name='negative_item_id')
    self.y = tf.placeholder(tf.float32, [None], name='label')
    self.hist_i = tf.placeholder(tf.int32, [None, None], name='history_item_ids')
    self.sl = tf.placeholder(tf.int32, [None], name='history_length')
    self.lr = tf.placeholder(tf.float32, shape=(), name='lr')
    self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')

    self.cand_ids = tf.placeholder(tf.int32, [None, None], name='cand_ids')
    self.use_cand_ids = tf.placeholder_with_default(False, shape=(), name='use_cand_ids')

    item_dim = 64
    cate_dim = 64
    theme_dim = 64
    hidden_units = item_dim + cate_dim + theme_dim

    user_emb_w = tf.get_variable('user_emb_w', [user_count, hidden_units])
    item_emb_w = tf.get_variable('item_emb_w', [item_count, item_dim])
    item_b = tf.get_variable(
      'item_b', [item_count], initializer=tf.constant_initializer(0.0)
    )
    cate_emb_w = tf.get_variable('cate_emb_w', [cate_count, cate_dim])
    theme_emb_w = tf.get_variable('theme_emb_w', [theme_count, theme_dim])

    cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)
    theme_list = tf.convert_to_tensor(theme_list, dtype=tf.int64)

    ic = tf.gather(cate_list, self.i)
    jc = tf.gather(cate_list, self.j)
    hist_ic = tf.gather(cate_list, self.hist_i)

    ti = tf.gather(theme_list, self.i)
    tj = tf.gather(theme_list, self.j)
    hist_ti = tf.gather(theme_list, self.hist_i)

    i_emb = tf.concat([
      tf.nn.embedding_lookup(item_emb_w, self.i),
      tf.nn.embedding_lookup(cate_emb_w, ic),
      tf.nn.embedding_lookup(theme_emb_w, ti),
    ], axis=1)

    j_emb = tf.concat([
      tf.nn.embedding_lookup(item_emb_w, self.j),
      tf.nn.embedding_lookup(cate_emb_w, jc),
      tf.nn.embedding_lookup(theme_emb_w, tj),
    ], axis=1)

    h_emb = tf.concat([
      tf.nn.embedding_lookup(item_emb_w, self.hist_i),
      tf.nn.embedding_lookup(cate_emb_w, hist_ic),
      tf.nn.embedding_lookup(theme_emb_w, hist_ti),
    ], axis=2)

    user_id_emb = tf.nn.embedding_lookup(user_emb_w, self.u)

    self.user_emb_w = user_emb_w
    self.theme_emb_w = theme_emb_w
    self.user_id_emb = user_id_emb

    # LayerNorm is stateless across train/eval and works with Keras 3 while the
    # rest of this program remains in TF1 graph mode.
    user_norm = keras.layers.LayerNormalization(
      axis=-1, epsilon=1e-6, name='user_id_layer_norm'
    )
    history_norm = keras.layers.LayerNormalization(
      axis=-1, epsilon=1e-6, name='history_layer_norm'
    )
    fusion_gate = keras.layers.Dense(
      hidden_units, activation='sigmoid', name='user_history_gate'
    )
    context_norm = keras.layers.LayerNormalization(
      axis=-1, epsilon=1e-6, name='user_context_layer_norm'
    )

    fcn_input_norm = keras.layers.LayerNormalization(
      axis=-1, epsilon=1e-6, name='din_input_layer_norm'
    )
    fcn_1 = keras.layers.Dense(128, activation='relu', name='din_dense_1')
    fcn_2 = keras.layers.Dense(64, activation='relu', name='din_dense_2')
    fcn_out = keras.layers.Dense(1, activation=None, name='din_logit')

    user_id_emb_norm = user_norm(user_id_emb)

    def target_attention(keys, query, sl):
      """Scaled target-aware attention for one item or a candidate slate.

      keys:  [B, H, D]
      query: [B, D] or [B, K, D]
      output has the same leading dimensions as query.
      """
      scale = tf.constant(math.sqrt(float(hidden_units)), dtype=tf.float32)
      hist_mask = tf.sequence_mask(
        sl, maxlen=tf.shape(keys)[1], dtype=tf.bool
      )

      if query.shape.ndims == 2:
        scores = tf.einsum('bhd,bd->bh', keys, query) / scale
        masked_scores = tf.where(
          hist_mask,
          scores,
          tf.fill(tf.shape(scores), tf.constant(-1e9, tf.float32)),
        )
        alpha = tf.nn.softmax(masked_scores, axis=-1)
        out = tf.einsum('bh,bhd->bd', alpha, keys)
        valid = tf.cast(tf.greater(sl, 0), tf.float32)[:, None]
        return out * valid

      if query.shape.ndims == 3:
        scores = tf.einsum('bhd,bkd->bkh', keys, query) / scale
        cand_mask = tf.tile(
          tf.expand_dims(hist_mask, axis=1),
          [1, tf.shape(query)[1], 1],
        )
        masked_scores = tf.where(
          cand_mask,
          scores,
          tf.fill(tf.shape(scores), tf.constant(-1e9, tf.float32)),
        )
        alpha = tf.nn.softmax(masked_scores, axis=-1)
        out = tf.einsum('bkh,bhd->bkd', alpha, keys)
        valid = tf.cast(tf.greater(sl, 0), tf.float32)[:, None, None]
        return out * valid

      raise ValueError('query must have rank 2 or 3')

    def fuse_user_history(static_user, history_context):
      """Gated hybrid profile: learned user ID + target-aware history."""
      hist = history_norm(history_context)
      if history_context.shape.ndims == 3:
        static_user = tf.tile(
          tf.expand_dims(static_user, axis=1),
          [1, tf.shape(history_context)[1], 1],
        )
      gate_features = tf.concat(
        [static_user, hist, static_user * hist], axis=-1
      )
      gate = fusion_gate(gate_features)
      fused = gate * static_user + (1.0 - gate) * hist
      return context_norm(fused)

    def din_fcn(user_context, item_context):
      # Difference and element-wise product are standard matching features in
      # industrial ranking towers and make the interaction explicit.
      x = tf.concat([
        user_context,
        item_context,
        user_context - item_context,
        user_context * item_context,
      ], axis=-1)
      x = fcn_input_norm(x)
      x = fcn_1(x)
      x = fcn_2(x)
      x = fcn_out(x)
      return tf.squeeze(x, axis=-1)

    hist_i_att = target_attention(h_emb, i_emb, self.sl)
    hist_j_att = target_attention(h_emb, j_emb, self.sl)
    user_context_i = fuse_user_history(user_id_emb_norm, hist_i_att)
    user_context_j = fuse_user_history(user_id_emb_norm, hist_j_att)

    self.hist_i_att = hist_i_att
    self.user_context_i = user_context_i

    d3_i = din_fcn(user_context_i, i_emb)
    d3_j = din_fcn(user_context_j, j_emb)
    i_b = tf.nn.embedding_lookup(item_b, self.i)
    j_b = tf.nn.embedding_lookup(item_b, self.j)

    self.logits = i_b + d3_i

    item_emb_all = tf.concat([
      item_emb_w,
      tf.nn.embedding_lookup(cate_emb_w, cate_list),
      tf.nn.embedding_lookup(theme_emb_w, theme_list),
    ], axis=1)

    K = predict_ads_num
    hidden_dim = item_emb_all.get_shape().as_list()[1]

    def _take_prefix():
      sub = item_emb_all[:K, :]
      sub = tf.expand_dims(sub, 0)
      batch_size = tf.shape(self.hist_i)[0]
      sub = tf.tile(sub, [batch_size, 1, 1])
      bsub = item_b[:K]
      bsub = tf.tile(tf.expand_dims(bsub, 0), [batch_size, 1])
      return sub, bsub

    def _take_custom():
      flat_ids = tf.reshape(self.cand_ids, [-1])
      sub = tf.nn.embedding_lookup(item_emb_all, flat_ids)
      sub = tf.reshape(
        sub, [-1, tf.shape(self.cand_ids)[1], hidden_dim]
      )
      bsub = tf.nn.embedding_lookup(item_b, flat_ids)
      bsub = tf.reshape(bsub, [-1, tf.shape(self.cand_ids)[1]])
      return sub, bsub

    item_emb_sub, item_b_sub = tf.cond(
      self.use_cand_ids, _take_custom, _take_prefix
    )

    # Every candidate gets its own target-aware history representation. This
    # avoids leaking the pointwise target query into the entire listwise slate.
    hist_cand_att = target_attention(h_emb, item_emb_sub, self.sl)
    user_context_sub = fuse_user_history(user_id_emb_norm, hist_cand_att)
    d3 = din_fcn(user_context_sub, item_emb_sub)

    self.hist_cand_att = hist_cand_att
    self.user_context_sub = user_context_sub
    self.logits_sub_raw = item_b_sub + d3
    self.logits_sub = tf.sigmoid(self.logits_sub_raw)

    self.score_i = tf.reshape(tf.sigmoid(i_b + d3_i), [-1, 1])
    self.score_j = tf.reshape(tf.sigmoid(j_b + d3_j), [-1, 1])
    self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)

    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = tf.Variable(
      0, trainable=False, name='global_epoch_step'
    )
    self.global_epoch_step_op = tf.assign(
      self.global_epoch_step, self.global_epoch_step + 1
    )

    self.loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.logits, labels=self.y
      )
    )

    # This is a hard graph-level invariant: exported user embeddings must be
    # connected to the training objective.
    self.user_embedding_grad = tf.gradients(self.loss, user_emb_w)[0]
    if self.user_embedding_grad is None:
      raise RuntimeError('user_emb_w is disconnected from the DIN loss')

    trainable_params = tf.trainable_variables()
    grads = tf.gradients(self.loss, trainable_params)
    grad_var = [(g, v) for g, v in zip(grads, trainable_params) if g is not None]
    clip_grads, _ = tf.clip_by_global_norm([g for g, _ in grad_var], 5.0)
    optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
    self.train_op = optimizer.apply_gradients(
      [(g, v) for g, (_, v) in zip(clip_grads, grad_var)],
      global_step=self.global_step,
    )

  def test(self, sess, uij):
    import numpy as np
    u, i, j, hist_i, sl = uij
    feed = {
      self.u: u,
      self.i: i,
      self.j: j,
      self.hist_i: hist_i,
      self.sl: sl,
      self.is_training: False,
    }
    out = sess.run(self.p_and_n, feed_dict=feed)
    return np.asarray(out)

  def eval(self, sess, uij):
    return None, self.test(sess, uij)

  def train(self, sess, uij, lr):
    u, i, y, hist_i, sl = uij
    feed = {
      self.u: u,
      self.i: i,
      self.y: y,
      self.hist_i: hist_i,
      self.sl: sl,
      self.lr: lr,
      self.is_training: True,
    }
    loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed)
    return float(loss)
