# CN_Aug

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
タイムステップに依存して，加えるノイズの量が決定される．どのタイムステップから使用するかがランダムであるだけで，ノイズ量は決定的である．