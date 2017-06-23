# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import collections

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# from tensorflow.models.rnn.translate import data_utils
import data_utils
# from tensorflow.models.rnn.translate import seq2seq_model
import seq2seq_model

from shutil import copyfile

tf.app.flags.DEFINE_float("learning_rate", 2, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95,"Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,"Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 4, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 150000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 150000, "French vocabulary size.")
tf.app.flags.DEFINE_integer("num_samples", 512, "Num samples for sampled softmax.")

# tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
# tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size",  0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1500, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_integer("total_epoch",6, "Number of epoch to run")
tf.app.flags.DEFINE_integer("data_size", 13000000, "Size of Data")
# added flags for Philly use
tf.app.flags.DEFINE_boolean("local", False, "Local run status")
tf.app.flags.DEFINE_string("lstm_type", "basic", "Use BasicLSTMCell(basic) or LSTMCell(advance)"  )
tf.app.flags.DEFINE_string("tmp_model_folder", "/tmp", "Safe file to write tmp checkpoints to")
# tf.app.flags.DEFINE_string("model_folder", str(os.path.join("/hdfs/pnrsy/sys/jobs", os.environ['PHILLY_JOB_ID'], "models")),"master file to write models to")
tf.app.flags.DEFINE_string("model_folder", str(os.path.join("/hdfs/pnrsy/sys/jobs", "models")),"master file to write models to")
tf.app.flags.DEFINE_string("model_dir", "default/","master file to write models to")

# tf.app.flags.DEFINE_string("model_folder", str(os.path.join("/hdfs/pnrsy/sys/jobs", "models")),"master file to write models to")
tf.app.flags.DEFINE_string("data_dir", "./", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./models", "Training directory.") # this is not actually used
# tf.app.flags.DEFINE_string("second_data_dir", sys.argv[1], 'Training directory.')
FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
# _buckets = [(30, 4), (40, 6), (50, 8), (100, 20)] # buckets for sentence to answer
_buckets = [(3, 7), (5, 12), (7, 17), (9, 20)]

if FLAGS.local == False:
  print("Printing Parameters.........................")
  print("sys.version : " + sys.version)
  print("learning_rate = %f" %  FLAGS.learning_rate)
  print("learning_rate_decay_factor = %f" %  FLAGS.learning_rate_decay_factor)
  print("max_gradient_norm = %d" %  FLAGS.max_gradient_norm)
  print("batch_size = %d" %  FLAGS.batch_size)
  print("num_layers = %d" %  FLAGS.num_layers)
  print("hidden_size = %d" %  FLAGS.size)
  print("en_vocab_size = %d" %  FLAGS.en_vocab_size)
  print("fr_vocab_size = %d" %  FLAGS.fr_vocab_size)
  print("num_samples = %d" %  FLAGS.num_samples)
  print("_buckets = %s" %  _buckets)
  print("steps_per_checkpoint = %d" %  FLAGS.steps_per_checkpoint)
  print("use_lstm = True")
  print("lstm_type : " + FLAGS.lstm_type)
  print("total_epoch : %d" % FLAGS.total_epoch)
  MAX_ITERATION_COUNT = FLAGS.total_epoch * (FLAGS.data_size)/(FLAGS.batch_size)
  print("total_iteration : %d" % MAX_ITERATION_COUNT)
  print("data dir : " + FLAGS.data_dir)
  # print("data dir : " + FLAGS.second_data_dir)
  print("model dir : " + FLAGS.model_folder)
  print("Printing Parameters end.....................")


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.en_vocab_size, FLAGS.fr_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, use_lstm=True,
      num_samples = FLAGS.num_samples,forward_only=forward_only, bidirectional=True, lstm_type=FLAGS.lstm_type)

  # print the trainable variables
  for v in tf.trainable_variables():
    print(v.name)

  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  print(ckpt)

  # one-off <ignore>
  # tmp_model_path = os.path.join(FLAGS.data_dir, "my-model-00418500")
  # tmp_model_path = os.path.join("..\..\..\RQnA\PhillyExperiments\FirstIteration_2_layer_lstm_advance\Checkpoints", "my-model-00418500")
  # model.saver.restore(session, ckpt.tmp_model_path)
  # try:
  #   model.saver.restore(session, tmp_model_path)
  #   print("Restored Model ! ")
  # except Exception as e:
  #   print("Not able to Restore !")  
  
  try:
    print("Master, Trying to read model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
    print("Master, Parameters Read !!")
  except Exception as e:
    print("Master, Creating model with fresh parameters.")
    session.run(tf.initialize_all_variables())
    print("Master, Created model with fresh parameters.")
  
  return model


def train():
  """Train a en->fr translation model using WMT data."""
  # Prepare WMT data.
  print("Preparing WMT data in %s" % FLAGS.data_dir)
  en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_squad_data(FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocab_size)
  print("TF Version")
  print(tf.__version__)

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)
    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."% FLAGS.max_train_data_size)
    print("query_dev Path : " + en_dev )
    print("question_dev Path : " + fr_dev )
    dev_set = read_data(en_dev, fr_dev)
    dev_bucket_sizes = [len(dev_set[b]) for b in xrange(len(_buckets))]
    print("Development Data read ! ")
    train_set = read_data(en_train, fr_train, FLAGS.max_train_data_size)
    print("Training Data read")
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    

    print("**** Bucket Sizes :  ")
    print(dev_bucket_sizes)
    print(train_bucket_sizes)
    print("****")
    
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    model_checkpoint_list = [] # maintain list of past checkpoints
    min_eval_perplex = collections.defaultdict(float)
    
    for bucket_id in xrange(len(_buckets)):
      min_eval_perplex[bucket_id] = float("inf")
    

    while current_step < MAX_ITERATION_COUNT:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      
      # print("Iteration : ")
      # print(current_step)
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        num = (current_step/MAX_ITERATION_COUNT)*100
        print()
        print('LOSS : %.2f%%' % loss)
        print()
        print('PROGRESS: %.2f%%' % num)
        print()
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        if FLAGS.local is False:
          tmp_model_checkpoint_path = os.path.join(FLAGS.tmp_model_folder, "model.ckpt-" + str(current_step))
          # model_checkpoint_path = os.path.join(FLAGS.model_dir, "model.ckpt-" + str(current_step))
          model_checkpoint_path = os.path.join(FLAGS.model_dir, "my-model")
          tmp_checkpoint_path   = os.path.join(FLAGS.tmp_model_folder, "checkpoint")
          # checkpoint_path = os.path.join("/hdfs/pnrsy/sys/jobs", os.environ['PHILLY_JOB_ID'], "models", "translate.ckpt")
          checkpoint_path = os.path.join(FLAGS.model_dir, "checkpoint")

          if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
            print("Created Folder !")
          try:
            # print(tmp_checkpoint_path)
            # print(tmp_model_checkpoint_path)
            print(model_checkpoint_path)
            model.saver.save(sess, model_checkpoint_path, global_step=model.global_step)
            print("Saved Model")
            # model_checkpoint_list.append(model_checkpoint_path)
          except Exception as e:
           print("FAILED TO COPY FOR CHECKPOINT FOR FILE %s" % model_checkpoint_path)
           try:
             print(e.message)
           except Exception as ee:
             print("NO EXCEPTION MESSAGE")
          if len(model_checkpoint_list) > 5:
            os.remove(model_checkpoint_list[0])
            model_checkpoint_list.pop(0)
        else:
          checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
          # model.saver.save(sess, checkpoint_path, global_step=model.global_step)

        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        new_eval_perplex = collections.defaultdict(float)
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          new_eval_perplex[bucket_id] = eval_ppx
        sys.stdout.flush()
        stop = True
        for k, v in new_eval_perplex.items():
          if v <= 1.5 * min_eval_perplex[k]:
            stop = False
          if v < min_eval_perplex[k]:
            min_eval_perplex[k] = v
        if stop:
          break


def decode():
  with tf.Session() as sess:
    # with tf.variable_scope("decode"):
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,"vocab%d.en" % FLAGS.en_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,"vocab%d.fr" % FLAGS.fr_vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
