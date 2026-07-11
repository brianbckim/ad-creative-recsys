import unittest

import numpy as np
import tensorflow as _tf

from din.model import Model
from din.train_listwise_kd import evaluate_teacher_and_student
from make_din_dataset import build_histories, collect_histories


tf = _tf.compat.v1


class DinHybridModelTest(unittest.TestCase):

    def _build_model(self):
        tf.reset_default_graph()
        tf.set_random_seed(7)
        return Model(
            user_count=4,
            item_count=6,
            cate_count=2,
            cate_list=[0, 1, 0, 1, 0, 1],
            theme_count=3,
            theme_list=[0, 1, 2, 0, 1, 2],
            predict_batch_size=2,
            predict_ads_num=3,
        )

    def test_user_embedding_has_gradient_and_updates(self):
        model = self._build_model()
        with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as sess:
            sess.run(tf.global_variables_initializer())
            feed = {
                model.u: np.asarray([0, 1], dtype=np.int32),
                model.i: np.asarray([2, 3], dtype=np.int32),
                model.j: np.asarray([4, 5], dtype=np.int32),
                model.y: np.asarray([1.0, 0.0], dtype=np.float32),
                model.hist_i: np.asarray([[0, 1], [1, 0]], dtype=np.int32),
                model.sl: np.asarray([2, 1], dtype=np.int32),
                model.lr: 1e-3,
                model.is_training: True,
            }
            before = sess.run(model.user_emb_w)
            grad, _ = sess.run(
                [model.user_embedding_grad, model.train_op], feed_dict=feed
            )
            after = sess.run(model.user_emb_w)

        grad_values = grad.values if hasattr(grad, "values") else grad
        self.assertGreater(float(np.linalg.norm(grad_values)), 0.0)
        self.assertGreater(float(np.mean(np.abs(after - before))), 0.0)

    def test_candidate_attention_is_candidate_specific(self):
        model = self._build_model()
        with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as sess:
            sess.run(tf.global_variables_initializer())
            attended = sess.run(
                model.hist_cand_att,
                feed_dict={
                    model.u: np.asarray([0], dtype=np.int32),
                    model.hist_i: np.asarray([[0, 1]], dtype=np.int32),
                    model.sl: np.asarray([2], dtype=np.int32),
                    model.cand_ids: np.asarray([[2, 3, 4]], dtype=np.int32),
                    model.use_cand_ids: True,
                    model.is_training: False,
                },
            )

        self.assertEqual(attended.shape, (1, 3, 192))
        self.assertGreater(float(np.linalg.norm(attended[0, 0] - attended[0, 1])), 0.0)

    def test_empty_history_is_finite_and_masked(self):
        model = self._build_model()
        with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as sess:
            sess.run(tf.global_variables_initializer())
            attended, logits = sess.run(
                [model.hist_cand_att, model.logits_sub_raw],
                feed_dict={
                    model.u: np.asarray([0], dtype=np.int32),
                    model.hist_i: np.asarray([[0]], dtype=np.int32),
                    model.sl: np.asarray([0], dtype=np.int32),
                    model.cand_ids: np.asarray([[2, 3, 4]], dtype=np.int32),
                    model.use_cand_ids: True,
                    model.is_training: False,
                },
            )

        self.assertTrue(np.isfinite(logits).all())
        np.testing.assert_allclose(attended, 0.0, atol=1e-7)

    def test_held_out_teacher_student_metrics_are_reported(self):
        model = self._build_model()
        test_set = [(0, [0, 1], [2, 5])]
        candidates = {0: ([2, 3, 4], [3.0, 2.0, 1.0])}

        with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as sess:
            sess.run(tf.global_variables_initializer())
            metrics = evaluate_teacher_and_student(
                sess,
                model,
                test_set,
                candidates,
                K=3,
                ks=(1, 3),
                batch_size=2,
            )

        self.assertEqual(metrics["eval_users"], 1)
        self.assertEqual(metrics["teacher_ndcg@1"], 1.0)
        self.assertIn("student_ndcg@3", metrics)
        self.assertIn("delta_ndcg@3", metrics)
        self.assertTrue(np.isfinite(metrics["student_ndcg@3"]))


class DinHistoryConstructionTest(unittest.TestCase):

    def test_validation_inherits_positive_train_history(self):
        train_rows = [
            (0, 1, 1, "2026-01-01"),
            (0, 2, 0, "2026-01-02"),
            (0, 3, 1, "2026-01-03"),
        ]
        validation_rows = [(0, 4, 1, "2026-01-04")]

        seed = collect_histories(train_rows, positive_only=True, max_history_len=50)
        validation = build_histories(
            validation_rows,
            initial_histories=seed,
            positive_only=True,
            max_history_len=50,
        )

        self.assertEqual(seed[0], [1, 3])
        self.assertEqual(validation[0][1], [1, 3])


if __name__ == "__main__":
    unittest.main()
