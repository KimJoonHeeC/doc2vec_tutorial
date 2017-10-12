import tensorflow as tf
import random, math, os

from config import FLAGS
from model import Seq2Seq
from dialog import Dialog
from datetime import datetime

def train(dialog, batch_size=100, epoch=10):
    model = Seq2Seq(dialog.vocab_size)

    with tf.Session() as sess:
        # TODO: 세션을 로드하고 로그를 위한 summary 저장등의 로직을 Seq2Seq 모델로 넣을 필요가 있음
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("now loading from this file", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("now creating new model")
            sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        total_batch = int(math.ceil(len(dialog.examples)/float(batch_size)))
        print('Total batch: %d' % total_batch)
        print('epoch: %d' % epoch)

        for step in range(total_batch * epoch):
            enc_input, dec_input, targets = dialog.next_batch(batch_size)

            print(len(enc_input), len(dec_input), len(targets))

            _, loss = model.train(sess, enc_input, dec_input, targets)

            if (step + 1) % 5 == 0:
                model.write_logs(sess, writer, enc_input, dec_input, targets)

                print('Step:', '%06d' % model.global_step.eval(),
                      'cost =', '{:.6f}'.format(loss),
                      'time: %s' % str(datetime.now()))

        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.ckpt_name)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

    print('optimization completed.')


def test(dialog, batch_size=100):
    print("\n=== prediction test ===")

    model = Seq2Seq(dialog.vocab_size)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        print("now loading from this file", ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        enc_input, dec_input, targets = dialog.next_batch(batch_size)

        expect, outputs, accuracy = model.test(sess, enc_input, dec_input, targets)

        expect = dialog.decode(expect)
        outputs = dialog.decode(outputs)

        pick = random.randrange(0, len(expect) / 2)
        input = dialog.decode([dialog.examples[pick * 2]], True)
        expect = dialog.decode([dialog.examples[pick * 2 + 1]], True)
        outputs = dialog.cut_eos(outputs[pick])

        print("\naccuracy value:", accuracy)
        print("random result\n")
        print("    input value:", input)
        print("    real result:", expect)
        print("    predicted result:", ' '.join(outputs))


def main(_):
    dialog = Dialog()

    dialog.load_vocab(FLAGS.voc_path)
    dialog.load_examples(FLAGS.data_path)

    if FLAGS.train:
        train(dialog, batch_size=FLAGS.batch_size, epoch=FLAGS.epoch)
    elif FLAGS.test:
        test(dialog, batch_size=FLAGS.batch_size)

if __name__ == "__main__":
    tf.app.run()
