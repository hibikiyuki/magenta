import os
from magenta.models.music_vae import configs
from magenta.models.music_vae import TrainedModel
import note_seq
import tensorflow.compat.v1 as tf
import numpy as np
import glob
tf.disable_v2_behavior()

def encode_note_sequence(note_sequence_path, model_config, checkpoint_path, batch_size=1):
    """
    指定されたMIDIファイルのnote_sequenceをエンコードして、対応する潜在ベクトルzを返します。

    Args:
    - note_sequence_path: エンコードするMIDIファイルのパス。
    - model_config: 使用するモデルの設定名。
    - checkpoint_path: モデルのチェックポイントファイルまたはディレクトリのパス。
    - batch_size: バッチサイズ（デフォルトは1）。

    Returns:
    - z: note_sequenceに対応する潜在ベクトル。
    """
    # MIDIファイルからNoteSequenceを作成
    note_sequence = note_seq.midi_file_to_note_sequence(note_sequence_path)

    # モデル設定を取得
    config = configs.CONFIG_MAP[model_config]

    # TrainedModelインスタンスを作成
    model = TrainedModel(
        config=config,
        batch_size=batch_size,
        checkpoint_dir_or_path=checkpoint_path)

    # NoteSequenceをエンコード
    z, _, _ = model.encode([note_sequence])

    return z

def save_latent_vectors(directory, output_directory, model_config, checkpoint_path, batch_size=1):
    """
    指定されたディレクトリ内の全MIDIファイルに対して潜在ベクトルzを計算し、保存します。

    Args:
    - directory: MIDIファイルが含まれているディレクトリのパス。
    - output_directory: 生成されたzベクトルを保存するディレクトリのパス。
    - model_config: 使用するモデルの設定名。
    - checkpoint_path: モデルのチェックポイントファイルまたはディレクトリのパス。
    - batch_size: バッチサイズ（デフォルトは1）。
    """
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # 指定されたディレクトリ内の全MIDIファイルを探索
    for filename in os.listdir(directory):
        if filename.endswith(".mid") or filename.endswith(".midi"):
            filepath = os.path.join(directory, filename)
            try:
                # MIDIファイルから潜在ベクトルzを計算
                z = encode_note_sequence(filepath, model_config, checkpoint_path, batch_size)
                # zをファイルとして保存
                output_filepath = os.path.join(output_directory, filename + '.npy')
                np.save(output_filepath, z)
                print(f"Saved: {output_filepath}")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

def save_z(directory, output_directory, model_config, checkpoint_path):
    """
    指定されたディレクトリ内の全MIDIファイルに対して潜在ベクトルzを計算し、保存します。

    Args:
    - directory: MIDIファイルが含まれているディレクトリのパス。
    - output_directory: 生成されたベクトルzを保存するディレクトリのパス。
    - model_config: 使用するモデルの設定名。
    - checkpoint_path: モデルのチェックポイントファイルまたはディレクトリのパス。
    """    
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    note_sequences = []
    midi_files = glob.glob(os.path.join(directory, '*.mid'))
    for file in midi_files:
        note_sequences.append(note_seq.midi_file_to_note_sequence(file))

    # モデル設定を取得
    config = configs.CONFIG_MAP[model_config]

    # TrainedModelインスタンスを作成
    model = TrainedModel(
        config=config,
        batch_size=min(8, len(note_sequences)),
        checkpoint_dir_or_path=checkpoint_path)

    # NoteSequenceをエンコード
    z, _, _ = model.encode(note_sequences)

    # print(z)
    # print(z.shape)
    for i in range(z.shape[0]):
        # zをファイルとして保存
        output_filepath = os.path.join(output_directory, f'{i:02d}.npy')
        # np.save(output_filepath, np.expand_dims(z[i, :], axis=0))
        np.save(output_filepath, z[i, :])
        print(f"Saved: {output_filepath}")



# 使用例
# directory = 'C:\\Users\\hibiki\\OneDrive\\ドキュメント\\Python Scripts\\midi_2bar'
# output_directory = 'output_z'
# model_config = 'cat-mel_2bar_big'  # 例: 'mel_2bar_small'
# checkpoint_path = r'C:\Users\hibiki\Documents\checkpoints\cat-mel_2bar_big.tar'

# save_latent_vectors(directory, output_directory, model_config, checkpoint_path)

# 使用例改
# directory = r"C:\Users\hibiki\Documents\iec240424"
# output_directory = r"C:\Users\hibiki\Documents\iec240424\z"
# model_config = 'cat-mel_2bar_big'  # 例: 'mel_2bar_small'
# checkpoint_path = r'C:\Users\hibiki\Documents\checkpoints\cat-mel_2bar_big.tar'

# save_z(directory, output_directory, model_config, checkpoint_path)