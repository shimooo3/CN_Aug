# CN_Aug
ControlNetを活用したデータ拡張
## 備忘録
### jupyterの出力を消してコミット
- pip install nbstripout
- cd git-repository
- nbstripout --install
Gitのpre-commitフックを設定し，コミット前に処理を行う

### tensorboardで学習結果の確認
```
$ tensorboard --logdir ./logs
```

### 二値分類の損失関数
import torch as nn  
nn.BCEWithLogits Lossの出力は生の値(logits)であるため，場合によってはシグモイド関数を通す必要がある．

### 拡散モデル訓練時のランダムサンプリング
タイムステップに依存して，加えるノイズの量が決定される．どのタイムステップから使用するかがランダムであるだけで，ノイズ量は決定的である．1-(t/T)^2　^2を使用して，タイムステップ初期-中期の影響を強くする．後期はノイズが多いため影響を減らす．  
学習が上手くいっていないため，重みの変化を見直す．最小値が0になっているのはよくなさそう

## controlnet
- 一貫性損失
- conditionの特徴量強化
- デコーダの追加
    - エンコーダからの入力
    - デコーダからの入力
    - 段階的な学習
- アテンションブロックの追加

## vscodeの入力
|ではなく白い箱が出ているときは挿入モードとなっている．Ins(insert)キーを押せば治る

## 減らないnoise_loss
最大値は下がるようになった，品質全体の底上げ

## フォルダ内のファイル数を数える
$ ls -l |wc -l