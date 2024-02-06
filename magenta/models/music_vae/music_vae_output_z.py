import os
from magenta.models.music_vae import configs
from magenta.models.music_vae import TrainedModel
import note_seq
import tensorflow.compat.v1 as tf
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
    note_sequence = note_seq.midi_file_to_note_sequence(midi_path)

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

# 使用例
model_config = 'cat-mel_2bar_big' # 例: 'mel_2bar_small'
checkpoint_path = r'C:\Users\hibiki\Documents\checkpoints\cat-mel_2bar_big.tar'
midi_path = r'C:\Users\hibiki\OneDrive\ドキュメント\Python Scripts\midi_2bar\cat-mel_2bar_big_sample_2024-02-06_110527-000-of-200.mid'

z = encode_note_sequence(midi_path, model_config, checkpoint_path)
print(z)
