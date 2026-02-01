import tensorflow as _tf
tf = _tf.compat.v1
tf.disable_v2_behavior()

class Model(object):

  def __init__(self, user_count, item_count,
               cate_count, cate_list,
               theme_count, theme_list,
               predict_batch_size, predict_ads_num):

    self.u = tf.placeholder(tf.int32, [None,])
    self.i = tf.placeholder(tf.int32, [None,])
    self.j = tf.placeholder(tf.int32, [None,])
    self.y = tf.placeholder(tf.float32, [None,])
    self.hist_i = tf.placeholder(tf.int32, [None, None])
    self.sl = tf.placeholder(tf.int32, [None,])
    self.lr = tf.placeholder(tf.float64, [])

    self.cand_ids = tf.placeholder(tf.int32, [None, None], name='cand_ids')
    self.use_cand_ids = tf.placeholder_with_default(False, shape=(), name='use_cand_ids')

    item_dim = 64
    cate_dim = 64
    theme_dim = 64
    hidden_units = item_dim + cate_dim + theme_dim

    user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
    item_emb_w = tf.get_variable("item_emb_w", [item_count, item_dim])
    item_b     = tf.get_variable("item_b",     [item_count],
                                 initializer=tf.constant_initializer(0.0))
    cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, cate_dim])
    theme_emb_w = tf.get_variable("theme_emb_w", [theme_count, theme_dim])

    cate_list  = tf.convert_to_tensor(cate_list, dtype=tf.int64)
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

    self.user_emb_w = user_emb_w
    self.theme_emb_w = theme_emb_w

    def attention(keys, query, sl):
      q = tf.expand_dims(query, 1)
      scores = tf.reduce_sum(keys * q, axis=-1)

      mask = tf.sequence_mask(sl, maxlen=tf.shape(keys)[1], dtype=tf.float32)
      scores = scores + (1.0 - mask) * (-1e9)

      alpha = tf.nn.softmax(scores)
      alpha = tf.expand_dims(alpha, -1)

      out = tf.reduce_sum(keys * alpha, axis=1)
      return out

    hist_i_att = attention(h_emb, i_emb, self.sl)
    hist_j_att = attention(h_emb, j_emb, self.sl)
    hist_i_att = tf.layers.batch_normalization(
      hist_i_att, name='hist_bn', reuse=tf.AUTO_REUSE
    )
    hist_j_att = tf.layers.batch_normalization(
      hist_j_att, name='hist_bn', reuse=True
    )

    def din_fcn(u_emb, item_emb):
      x = tf.concat([u_emb, item_emb, u_emb * item_emb], axis=-1)
      x = tf.layers.batch_normalization(
        x, name='b1', reuse=tf.AUTO_REUSE
      )
      x = tf.layers.dense(
        x, 80, activation=tf.nn.sigmoid, name='f1', reuse=tf.AUTO_REUSE
      )
      x = tf.layers.dense(
        x, 40, activation=tf.nn.sigmoid, name='f2', reuse=tf.AUTO_REUSE
      )
      x = tf.layers.dense(
        x, 1, activation=None, name='f3', reuse=tf.AUTO_REUSE
      )
      return tf.reshape(x, [-1])

    d3_i = din_fcn(hist_i_att, i_emb)
    d3_j = din_fcn(hist_j_att, j_emb)
    i_b = tf.nn.embedding_lookup(item_b, self.i)
    j_b = tf.nn.embedding_lookup(item_b, self.j)

    self.logits = i_b + d3_i

    item_emb_all = tf.concat([
      item_emb_w,
      tf.nn.embedding_lookup(cate_emb_w, cate_list),
      tf.nn.embedding_lookup(theme_emb_w, theme_list),
    ], axis=1)

    K = predict_ads_num
    B = predict_batch_size
    hidden_dim = item_emb_all.get_shape().as_list()[1]

    def _take_prefix():
      sub = item_emb_all[:K, :]
      sub = tf.expand_dims(sub, 0)
      sub = tf.tile(sub, [B, 1, 1])
      bsub = item_b[:K]
      bsub = tf.tile(tf.expand_dims(bsub, 0), [B,1])
      return sub, bsub

    def _take_custom():
      flat_ids = tf.reshape(self.cand_ids, [-1])
      sub = tf.nn.embedding_lookup(item_emb_all, flat_ids)
      sub = tf.reshape(sub, [-1, tf.shape(self.cand_ids)[1], hidden_dim])
      bsub = tf.nn.embedding_lookup(item_b, flat_ids)
      bsub = tf.reshape(bsub, [-1, tf.shape(self.cand_ids)[1]])
      return sub, bsub

    item_emb_sub, item_b_sub = tf.cond(self.use_cand_ids, _take_custom, _take_prefix)

    u_emb_sub = tf.tile(tf.expand_dims(hist_i_att, 1), [1, tf.shape(item_emb_sub)[1], 1])
    din_sub = tf.concat([u_emb_sub, item_emb_sub, u_emb_sub * item_emb_sub], axis=-1)
    din_sub = tf.layers.batch_normalization(
      din_sub, name='b1', reuse=True
    )
    d1 = tf.layers.dense(
      din_sub, 80, activation=tf.nn.sigmoid, name='f1', reuse=True
    )
    d2 = tf.layers.dense(
      d1, 40, activation=tf.nn.sigmoid, name='f2', reuse=True
    )
    d3 = tf.layers.dense(
      d2, 1, activation=None, name='f3', reuse=True
    )
    d3 = tf.reshape(d3, [-1, tf.shape(item_emb_sub)[1]])

    self.logits_sub_raw = item_b_sub + d3
    self.logits_sub      = tf.sigmoid(self.logits_sub_raw)

    self.score_i = tf.sigmoid(i_b + d3_i)
    self.score_j = tf.sigmoid(j_b + d3_j)
    self.score_i = tf.reshape(self.score_i, [-1, 1])
    self.score_j = tf.reshape(self.score_j, [-1, 1])
    self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)

    self.y = tf.placeholder(tf.float32, shape=[None], name='label')
    self.lr = tf.placeholder(tf.float32, shape=(), name='lr')
    self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')

    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

    self.loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y)
    )

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      trainable_params = tf.trainable_variables()
      grads = tf.gradients(self.loss, trainable_params)
      clip_grads, _ = tf.clip_by_global_norm(grads, 5.0)
      optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
      self.train_op = optimizer.apply_gradients(
        zip(clip_grads, trainable_params), global_step=self.global_step
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
    }
    if hasattr(self, "is_training"):
      feed[self.is_training] = False
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
      self.is_training: True
    }
    loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed)
    return float(loss)