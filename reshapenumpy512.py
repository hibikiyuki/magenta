import numpy as np
import os

# 処理したいフォルダのパスを指定
folder_path = 'vectors0701'

# フォルダ内の全ての.npyファイルを処理
for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        file_path = os.path.join(folder_path, filename)
        
        # ファイルを読み込む
        arr = np.load(file_path)
        
        # 配列の形状を確認
        if arr.shape == (1, 512):
            # (1, 512)から(512,)に形状を変更
            arr = arr.reshape(512,)
            
            # 変更した配列を同じファイル名で保存
            np.save(file_path, arr)
            print(f'{filename}の形状を(512,)に変更しました。')
        else:
            print(f'{filename}の形状は既に(512,)または他の形状です。スキップします。')