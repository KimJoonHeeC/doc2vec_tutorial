import tensorflow as tf

tf.app.flags.DEFINE_string("train_dir", "./model", "folder for NN model")
tf.app.flags.DEFINE_string("log_dir", "./logs", "folder for logs used in tensorboard")
tf.app.flags.DEFINE_string("ckpt_name", "conversation.ckpt", "checkpoint filename")

tf.app.flags.DEFINE_boolean("train", True, "progress learning.")
tf.app.flags.DEFINE_boolean("test", True, "run test.")
tf.app.flags.DEFINE_boolean("data_loop", False, "use in mini-data set experiment")
tf.app.flags.DEFINE_integer("batch_size", 100, "mini batch size")
tf.app.flags.DEFINE_integer("epoch", 10, "# of repeated learning")

tf.app.flags.DEFINE_string("data_path", "./data/chat.log", "location of conversation file")
tf.app.flags.DEFINE_string("voc_path", "./data/chat.voc", "vocabulary dict file")

tf.app.flags.DEFINE_boolean("voc_test", True, "test vocabulary dict")
tf.app.flags.DEFINE_boolean("voc_build", False, "make vocabulary dict by using conversation file")

tf.app.flags.DEFINE_integer("max_decode_len", 20, "max decoder cell size = max answer size")


FLAGS = tf.app.flags.FLAGS
