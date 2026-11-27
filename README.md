# accent-recorder

英語の発音練習（イギリス寄り・アメリカ寄り）の実験用ページとサーバー。

| ファイル | 役割 |
|---|---|
| `index.html` | 協力者が使う画面（GitHub Pages で公開）。目標を選ぶ → 6 文を読む → 結果 → 練習 → 6 文をもう一度読む → 比較 |
| `server/server.py` | 自分の PC で動かすサーバー。録音を受け取り、MFA → WavLM → 単語単位モデル（5 fold の平均）で判定して返す。録音と結果を `server/collected/` に保存する |
| `server/items.json` | 読む文と対象語の定義（画面とサーバーの両方が使う） |
| `record.html` | 録音だけ集めるページ（前の版。いつも通り／アメリカ風／イギリス風） |

## 1. サーバーの設定（最初の 1 回）

`server/server.py` の「設定」を自分の環境に合わせる。

- `RESULT_DIR`：単語単位モデルの結果フォルダ（`model_fold0.pt`〜`model_fold4.pt` がある所）
- `VCTK_DIR`：練習文のお手本（VCTK の母語話者の音声）を探す所
- `MFA_CMD`・`MFA_DICT`・`MFA_ACOUSTIC`：**学習データを作ったときと同じ** MFA の辞書と音響モデル
  - MFA が conda の環境に入っているなら、例：`MFA_CMD = ["conda", "run", "-n", "aligner", "mfa"]`

追加のインストールはいらない（torch・transformers・librosa は学習の環境のものを使う。サーバーは Python の標準ライブラリで動く）。

## 2. 試運転（モデルも MFA も使わない）

```
cd server
set MOCK=1          （Windows のコマンドプロンプト。PowerShell なら $env:MOCK="1"）
python server.py
```

ブラウザで `index.html` を開く（`?server=http://localhost:8000` は省略できる）。判定の数字は適当なので、画面の流れの確認だけに使う。

## 3. 本番

```
cd server
python server.py
```

`サーバーを起動しました` と出たら準備完了。1 回の判定に、MFA の分で数秒〜十数秒かかる。

### 離れた協力者に使ってもらうとき

別のターミナルで（[cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) を入れておく）：

```
cloudflared tunnel --url http://localhost:8000
```

表示された `https://xxxx.trycloudflare.com` を使って、協力者にこのリンクを送る：

```
https://dangeroushunter-dotcom.github.io/accent-recorder/?server=https://xxxx.trycloudflare.com&id=p01
```

- 実験の間は PC とサーバーとトンネルをつけたままにする
- トンネルの URL は起動のたびに変わる
- GitHub Pages は、リポジトリの Settings → Pages で `main` ブランチを公開しておく

## 4. 集まるデータ（`server/collected/`）

- `<ID>/<stage>_<item>_a<回数>_<日時>.wav`：録音そのもの（stage は pre / practice / post）
- `results.jsonl`：1 回の録音ごとに 1 行。対象語ごとの区間（MFA）、P(米)、目標への近さ
- `events.jsonl`：開始、練習の計画、お手本の再生、終了などの記録

## 文の出典

- The Rainbow Passage（Fairbanks, 1960）
- Speech Accent Archive の文（Weinberger, George Mason University）
- VCTK コーパス（CSTR, University of Edinburgh, ODC-BY 1.0）
