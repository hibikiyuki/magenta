# Copyright 2023 The Magenta Authors.
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

"""MusicVAE generation script."""

# TODO(adarob): Add support for models with conditioning.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from magenta.models.music_vae import configs
from magenta.models.music_vae import TrainedModel
import note_seq
import numpy as np
import tensorflow.compat.v1 as tf

import glob

flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'run_dir', None,
    'Path to the directory where the latest checkpoint will be loaded from.')
flags.DEFINE_string(
    'checkpoint_file', None,
    'Path to the checkpoint file. run_dir will take priority over this flag.')
flags.DEFINE_string(
    'output_dir', '/tmp/music_vae/generated',
    'The directory where MIDI files will be saved to.')
flags.DEFINE_string(
    'config', None,
    'The name of the config to use.')
flags.DEFINE_string(
    'mode', 'sample',
    'Generate mode (either `sample` or `interpolate`).')
flags.DEFINE_string(
    'input_midi_1', None,
    'Path of start MIDI file for interpolation.')
flags.DEFINE_string(
    'input_midi_2', None,
    'Path of end MIDI file for interpolation.')
flags.DEFINE_integer(
    'num_outputs', 5,
    'In `sample` mode, the number of samples to produce. In `interpolate` '
    'mode, the number of steps (including the endpoints).')
flags.DEFINE_integer(
    'max_batch_size', 8,
    'The maximum batch size to use. Decrease if you are seeing an OOM.')
flags.DEFINE_float(
    'temperature', 0.5,
    'The randomness of the decoding process.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')
# flags.DEFINE_string(
#     'z_vector_file', None,
#     'The path to the .npy file containing the z vector '
#     'to use for generation in `single_z` mode.')
flags.DEFINE_string(
    'vectors_dir', None,
    'Directory to store the source z.')
flags.DEFINE_bool(
    'savez', False,
    'Whether to save latent vector in sample mode.')
flags.DEFINE_string(
    'input_attribute_1', None,
    'Path of attribute vector 1.')
flags.DEFINE_string(
    'input_attribute_2', None,
    'Path of attribute vector 2.')
flags.DEFINE_float(
    'magnitude_1', 0.5,
    'Magnitude of input_attribute_1.')
flags.DEFINE_float(
    'magnitude_2', 0.5,
    'Magnitude of input_attribute_2.')
flags.DEFINE_string(
    'sample_mean', None,
    'Path of mean array npy in sample mode.')
flags.DEFINE_string(
    'sample_std', None,
    'Path of std array npy in sample mode.')


def _slerp(p0, p1, t):
  """Spherical linear interpolation."""
  omega = np.arccos(
      np.dot(np.squeeze(p0/np.linalg.norm(p0)),
             np.squeeze(p1/np.linalg.norm(p1))))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


def run(config_map):
  """Load model params, save config file and start trainer.

  Args:
    config_map: Dictionary mapping configuration name to Config object.

  Raises:
    ValueError: if required flags are missing or invalid.
  """
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')

  if FLAGS.run_dir is None == FLAGS.checkpoint_file is None:
    raise ValueError(
        'Exactly one of `--run_dir` or `--checkpoint_file` must be specified.')
  if FLAGS.output_dir is None:
    raise ValueError('`--output_dir` is required.')
  tf.gfile.MakeDirs(FLAGS.output_dir)

  """
  Added extrapolation mode and z vector mode to the executable modes.
  However, extrapolation was different from what I expected.
  241024 Update: Various modes added
  """
  # if FLAGS.mode != 'sample' and FLAGS.mode != 'interpolate':
  if FLAGS.mode != 'sample' and FLAGS.mode != 'interpolate' and \
      FLAGS.mode != 'extrapolate' and \
      FLAGS.mode != 'single_z' and \
      FLAGS.mode != 'vectors' and \
      FLAGS.mode != 'vectors_withattr' and \
      FLAGS.mode != 'sample_withattr':
    raise ValueError('Invalid value for `--mode`: %s' % FLAGS.mode)

  if FLAGS.config not in config_map:
    raise ValueError('Invalid config name: %s' % FLAGS.config)
  config = config_map[FLAGS.config]
  config.data_converter.max_tensors_per_item = None

  """
  In extrapolation mode, an error will be thrown even if there are not two input_midi.
  The error message has also been slightly revised accordingly.
  """
  # if FLAGS.mode == 'interpolate':
  if FLAGS.mode == 'interpolate' or FLAGS.mode == 'extrapolate':
    if FLAGS.input_midi_1 is None or FLAGS.input_midi_2 is None:
      # raise ValueError(
      #     '`--input_midi_1` and `--input_midi_2` must be specified in '
      #     '`interpolate` mode.')
      raise ValueError(
          '`--input_midi_1` and `--input_midi_2` must be specified in '
          '`interpolate` or `exterpolate` mode.')
    input_midi_1 = os.path.expanduser(FLAGS.input_midi_1)
    input_midi_2 = os.path.expanduser(FLAGS.input_midi_2)
    if not os.path.exists(input_midi_1):
      raise ValueError('Input MIDI 1 not found: %s' % FLAGS.input_midi_1)
    if not os.path.exists(input_midi_2):
      raise ValueError('Input MIDI 2 not found: %s' % FLAGS.input_midi_2)
    input_1 = note_seq.midi_file_to_note_sequence(input_midi_1)
    input_2 = note_seq.midi_file_to_note_sequence(input_midi_2)

    def _check_extract_examples(input_ns, path, input_number):
      """Make sure each input returns exactly one example from the converter."""
      tensors = config.data_converter.to_tensors(input_ns).outputs
      if not tensors:
        print(
            'MusicVAE configs have very specific input requirements. Could not '
            'extract any valid inputs from `%s`. Try another MIDI file.' % path)
        sys.exit()
      elif len(tensors) > 1:
        basename = os.path.join(
            FLAGS.output_dir,
            '%s_input%d-extractions_%s-*-of-%03d.mid' %
            (FLAGS.config, input_number, date_and_time, len(tensors)))
        for i, ns in enumerate(config.data_converter.from_tensors(tensors)):
          note_seq.sequence_proto_to_midi_file(
              ns, basename.replace('*', '%03d' % i))
        print(
            '%d valid inputs extracted from `%s`. Outputting these potential '
            'inputs as `%s`. Call script again with one of these instead.' %
            (len(tensors), path, basename))
        sys.exit()
    logging.info(
        'Attempting to extract examples from input MIDIs using config `%s`...',
        FLAGS.config)
    _check_extract_examples(input_1, FLAGS.input_midi_1, 1)
    _check_extract_examples(input_2, FLAGS.input_midi_2, 2)

  if FLAGS.mode == 'vectors' or FLAGS.mode == 'vectors_withattr':
    if FLAGS.vectors_dir is None:
        raise ValueError('`--z_vector_file` must be specified in `vectors` mode.')
    npy_path = os.path.expanduser(FLAGS.vectors_dir)
    if not os.path.exists(npy_path):
        raise ValueError('Vectors not found: %s' % FLAGS.vectors_dir)
    npy_files = glob.glob(os.path.join(npy_path, '*.npy'))
    if not npy_files:  # If the npy_files list is empty, i.e. no .npy files exist
        raise ValueError('No .npy files found in directory: %s' % npy_path)
    
  # if FLAGS.mode == 'sample_withattr':
    # if FLAGS.input_attribute_1 is None or FLAGS.input_attribute_2 is None:
    #   raise ValueError(
    #       '`--input_attribute_1` and `--input_attribute_2` must be specified in '
    #       '`sample_withattr` mode.')
    # input_attribute_1 = os.path.expanduser(FLAGS.input_attribute_1)
    # input_attribute_2 = os.path.expanduser(FLAGS.input_attribute_2)
    # if not os.path.exists(input_attribute_1):
    #   raise ValueError('Input attribute_1 not found: %s' % FLAGS.input_attribute_1)
    # if not os.path.exists(input_attribute_2):
    #   raise ValueError('Input attribute_2 not found: %s' % FLAGS.input_attribute_2)
    # attributes = [np.load(input_attribute_1), np.load(input_attribute_2)]

  # 後付けの属性ベクトル
  attributes = []
  
  input_attribute_1 = os.path.expanduser(FLAGS.input_attribute_1) if FLAGS.input_attribute_1 else None
  if input_attribute_1 and os.path.exists(input_attribute_1):
      attributes.append(np.load(input_attribute_1))
  else:
      attributes.append(np.zeros(512))

  input_attribute_2 = os.path.expanduser(FLAGS.input_attribute_2) if FLAGS.input_attribute_2 else None
  if input_attribute_2 and os.path.exists(input_attribute_2):
      attributes.append(np.load(input_attribute_2))
  else:
      attributes.append(np.zeros(512))

  # 生成時の分布をいじるetc
  sample_mean = 0
  sample_std = 1

  sample_mean_path = os.path.expanduser(FLAGS.sample_mean) if FLAGS.sample_mean else None
  if sample_mean_path and os.path.exists(sample_mean_path):
    sample_mean = np.load(sample_mean_path)

  sample_std_path = os.path.expanduser(FLAGS.sample_std) if FLAGS.sample_std else None
  if sample_std_path and os.path.exists(sample_std_path):
    sample_std = np.load(sample_std_path)

  logging.info('Loading model...')
  if FLAGS.run_dir:
    checkpoint_dir_or_path = os.path.expanduser(
        os.path.join(FLAGS.run_dir, 'train'))
  else:
    checkpoint_dir_or_path = os.path.expanduser(FLAGS.checkpoint_file)
  model = TrainedModel(
      config, batch_size=min(FLAGS.max_batch_size, FLAGS.num_outputs),
      checkpoint_dir_or_path=checkpoint_dir_or_path)

  if FLAGS.mode == 'interpolate':
    logging.info('Interpolating...')
    _, mu, _ = model.encode([input_1, input_2])
    z = np.array([
        _slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, FLAGS.num_outputs)])
    results = model.decode(
        length=config.hparams.max_seq_len,
        z=z,
        temperature=FLAGS.temperature)
  elif FLAGS.mode == 'sample':
    logging.info('Sampling...')
    # results = model.sample(
    #     n=FLAGS.num_outputs,
    #     length=config.hparams.max_seq_len,
    #     temperature=FLAGS.temperature,
    #     savez=FLAGS.savez)
    sampling_outputs = model.sample(
        n=FLAGS.num_outputs,
        length=config.hparams.max_seq_len,
        temperature=FLAGS.temperature,
        savez=FLAGS.savez)
    if FLAGS.savez:
      results, z = sampling_outputs
    else:
      results = sampling_outputs
  elif FLAGS.mode == 'sample_withattr':
    """
    sampling + attribute
    """
    logging.info('Sampling with attributes...')
    sampling_outputs = model.sample_withattr(
        n=FLAGS.num_outputs,
        length=config.hparams.max_seq_len,
        temperature=FLAGS.temperature,
        savez=FLAGS.savez,
        z_vectors=attributes,
        magnitudes=[FLAGS.magnitude_1, FLAGS.magnitude_2],
        mean=sample_mean,
        std=sample_std)
    if FLAGS.savez:
      results, z = sampling_outputs
    else:
      results = sampling_outputs
  elif FLAGS.mode == 'extrapolate':
    """
    外挿。linspace（等差数列）の最終値を1から2にしただけ。
    思ってたのと違った
    """
    logging.info('Extrapolating...')
    _, mu, _ = model.encode([input_1, input_2])
    z = np.array([
        _slerp(mu[0], mu[1], t) for t in np.linspace(0, 2, FLAGS.num_outputs)])  # 0 to 2 instead of 0 to 1
    results = model.decode(
        length=config.hparams.max_seq_len,
        z=z,
        temperature=FLAGS.temperature)
  elif FLAGS.mode == 'single_z':
    """
    zベクトルをこちらで指定した生成。モデルのデコード関数に指定したzを渡す。
    zはnpyファイルで用意。それから属性ベクトルを足し引きする。
    """
    if FLAGS.z_vector_file is None:
        raise ValueError('`--z_vector_file` must be specified in `single_z` mode.')
    # .npyファイルからzベクトルをロード
    z_vector_path = os.path.expanduser(FLAGS.z_vector_file)
    if not os.path.exists(z_vector_path):
        raise ValueError('z vector file not found: %s' % FLAGS.z_vector_file)
    z_vector = np.load(z_vector_path) # - np.load(r"C:\Users\hibiki\Documents\c_diatonic_flattened.npy")
    if z_vector.ndim != 1:
        raise ValueError('The loaded z vector must be a 1D array.')
    logging.info('Generating from a single z vector...')
    results = model.decode_single_z(
        z=z_vector,
        length=config.hparams.max_seq_len,
        temperature=FLAGS.temperature)
  elif FLAGS.mode == 'vectors':
    """
    zベクトルの集合のパスを渡して生成
    """
    logging.info('Generating from vectors...')
    arrays = [np.load(file) for file in npy_files]
    z = np.stack(arrays)
    results = model.decode(
        length=config.hparams.max_seq_len,
        z=z,
        temperature=FLAGS.temperature)
  elif FLAGS.mode == 'vectors_withattr':
    """
    zベクトルの集合のパスを渡して生成、そこに属性を付加
    """
    logging.info('Generating from vectors with attributes...')
    z_mode = np.load(
      r"C:\Users\arkw\GitHub\magenta\tmp\attribute0906\mode.npy") # +V Major
    z_sync8th = np.load(
      r"C:\Users\arkw\GitHub\magenta\tmp\attribute0906\syncopation_8th.npy")
    z_sync16th = np.load(
      r"C:\Users\arkw\GitHub\magenta\tmp\attribute0906\syncopation_16th.npy")
    z_pitch = np.load(
      r"C:\Users\arkw\GitHub\magenta\tmp\attribute0906\average_pitch.npy") # +V 高く
    z_staccato_level = np.load(
      r"C:\Users\arkw\GitHub\magenta\tmp\attribute0906\staccato_level.npy") # +A スタッカート気味
    z_density = np.load(
      r"C:\Users\arkw\GitHub\magenta\tmp\attribute0906\note_density.npy") # +A ノート密度
    # z_arousal = 0.25*z_sync8th + 0.25*z_sync16th + 0.5*z_pitch
    z_valence = 0.5*(z_mode + z_pitch)
    z_arousal = 0.5*(z_staccato_level + z_density)
    z_attr = z_valence
    arrays = [z_attr + np.load(file) for file in npy_files]
    z = np.stack(arrays)
    results = model.decode(
        length=config.hparams.max_seq_len,
        z=z,
        temperature=FLAGS.temperature)

  basename = os.path.join(
      FLAGS.output_dir,
      '%s_%s_%s-*-of-%03d.mid' %
      (FLAGS.config, FLAGS.mode, date_and_time, FLAGS.num_outputs))
  logging.info('Outputting %d files as `%s`...', FLAGS.num_outputs, basename)
  for i, ns in enumerate(results):
    note_seq.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))
    if (FLAGS.mode == 'sample' or FLAGS.mode == 'sample_withattr') and FLAGS.savez:
      np.save(basename.replace('*', '%03d' % i).replace('mid', 'npy'), z[i, :])

  logging.info('Done.')


def main(unused_argv):
  logging.set_verbosity(FLAGS.log)
  run(configs.CONFIG_MAP)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
