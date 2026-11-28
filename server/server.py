"""
発音練習アプリのサーバー（自分の PC で動かす）

  画面（index.html）から録音を受け取り、
    1. MFA で単語の区間を求める（学習のときと同じ）
    2. 単語ごとに WavLM の 13 層の特徴量を計算する（学習のときと同じ前処理）
    3. 単語単位モデル（5 fold の平均）で P(米) を出す
  結果を画面に返し、録音と結果をすべて collected/ に保存する。

  使い方
    python server.py              本番（GPU があれば GPU を使う）
    set MOCK=1 && python server.py   Windows：モデルも MFA も使わない試運転（画面の確認用）
    MOCK=1 python server.py          Mac / Linux：同上

  外から使ってもらうとき（別のターミナルで）
    cloudflared tunnel --url http://localhost:8000
    → 表示された https://xxxx.trycloudflare.com を、画面のリンクの ?server= に付ける

  追加のインストールはいらない（torch, transformers, librosa, soundfile, numpy は学習の環境に入っているもの）。
  サーバーは Python の標準ライブラリ（http.server）で動く。
"""
import difflib
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

# ================================================================
# 設定（自分の環境に合わせて書き換える）
# ================================================================
HOST, PORT = "127.0.0.1", int(os.environ.get("PORT", "8000"))

DATA_PATH = Path(r"C:\Users\林琢磨\Downloads\Attention")
# 単語単位モデルの結果フォルダ（model_fold0.pt 〜 model_fold4.pt が入っている所）
RESULT_DIR = DATA_PATH / "word_wavlm_data" / "results" / "target_att_w_per_phenomenon_m0"
VCTK_DIR = DATA_PATH / "VCTK-Corpus-0.92"           # 練習文のお手本（母語話者の音声）を探す所。なければお手本なし

MODEL_NAME = "microsoft/wavlm-base-plus"
PHENOMENA = ["BATH", "LOT", "R", "T", "YOD"]       # 学習と同じ順番（出力層の順番）
N_FOLDS = 5
MARGIN = 0.0                                       # 学習と同じ（単語の前後の余白）

# MFA（データ準備のノートブック 2026_1104 のセル 19〜21 と同じにする）
MFA_CMD = ["conda", "run", "-p", r"D:\mfa_env", "mfa"]      # MFA が入っている conda 環境（D:\mfa_env）
MFA_ROOT_DIR = r"D:\mfa_work\root"                        # モデルの保存場所（set MFA_ROOT_DIR=... と同じ）
MFA_DICT = r"D:\mfa_work\english_us_uk.dict"              # 英米の辞書をまとめたもの（学習と同じ）
MFA_ACOUSTIC = "english_mfa"                              # 音響モデル（学習と同じ）
MFA_EXTRA = []                                            # 学習のときは追加の設定なし
MFA_TMP = Path(r"D:\mfa_work\server_tmp")                 # 作業用の一時フォルダ（パスに日本語が入ると MFA が失敗するため）

N_REF = 3                                          # 練習文のお手本を、英米それぞれ何人分用意するか

HERE = Path(__file__).resolve().parent
ITEMS_JSON = HERE / "items.json"
SAVE_DIR = HERE / "collected"                      # 録音と結果の保存先
MOCK = os.environ.get("MOCK") == "1"

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
STAGES = ("pre", "practice", "post")


# ================================================================
# 文を単語に分ける（画面の index.html と同じ分け方）
# ================================================================
def tokens(text):
    out = []
    for w in text.split():
        w = re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))
        if w:
            out.append(w)
    return out


def normalize(text):
    return " ".join(tokens(text))


# ================================================================
# 文の定義を読む・対象語を決める
#   対象語は、データ準備（2026_1104）の TARGET_WORDS（items.json の train_vocab）と find_targets と同じ規則で決める
# ================================================================
ITEMS = json.loads(ITEMS_JSON.read_text(encoding="utf-8"))
TRAIN_VOCAB = {ph: set(ws) for ph, ws in ITEMS["train_vocab"].items()}


def find_targets(words):
    """データ準備の find_targets と同じ：(現象, 単語, 何番目の単語か) のリスト
       R：語末の r の直後が母音で始まる単語なら数えない（linking r）"""
    hits = []
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else ""
        for ph in PHENOMENA:
            if w not in TRAIN_VOCAB.get(ph, ()):
                continue
            if ph == "R" and w.endswith(("r", "re")) and nxt and nxt[0] in "aeiou":
                continue
            hits.append((ph, w, i))
    return hits


def targets_of(text, only_ph=None):
    """文の対象語 → [{"i": 何番目, "w": 単語, "ph": [現象, …]}]。only_ph を指定すると、その現象だけ"""
    by_i = {}
    for ph, w, i in find_targets(tokens(text)):
        if only_ph and ph != only_ph:
            continue
        by_i.setdefault(i, {"i": i, "w": w, "ph": []})["ph"].append(ph)
    return [by_i[i] for i in sorted(by_i)]


TEST = [dict(s, targets=targets_of(s["text"])) for s in ITEMS["test"]]
TEST_NORM = {normalize(s["text"]) for s in TEST}
TEST_WORDS = {t["w"] for s in TEST for t in s["targets"]}                     # 本番の対象語
TEST_WORDS_BY_PH = {ph: sorted({t["w"] for s in TEST for t in s["targets"] if ph in t["ph"]}) for ph in PHENOMENA}
ITEM_BY_ID = {s["id"]: dict(s, kind="test") for s in TEST}
PRACTICE = {}      # グループ → 現象 → [練習文, …]（起動時に build_practice で作る）
REFS = {}          # 練習文の id → {"uk": [音声ファイル, …], "us": […]}


# ================================================================
# 音声の読み書き（標準ライブラリの wave で、16bit の WAV だけ扱う）
# ================================================================
def read_wav(data):
    with wave.open(io.BytesIO(data)) as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError("16bit の WAV だけ受け付けます")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(path, x, sr):
    y = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())


# ================================================================
# MFA で単語の区間を求める
# ================================================================
def read_textgrid_words(path):
    """MFA が出す TextGrid（long 形式）から、words の段の（開始, 終了, 単語）を読む。空白の区間は除く"""
    txt = Path(path).read_text(encoding="utf-8")
    for block in re.split(r"\n\s*item \[\d+\]:", txt)[1:]:
        m = re.search(r'name = "([^"]*)"', block)
        if not m or not m.group(1).endswith("words"):
            continue
        iv = re.findall(r'xmin = ([-\d.eE+]+)\s*\n\s*xmax = ([-\d.eE+]+)\s*\n\s*text = "((?:[^"]|"")*)"', block)
        return [(float(a), float(b), t.replace('""', '"')) for a, b, t in iv if t.strip()]
    raise ValueError("TextGrid に words の段がありません")


def mfa_align(wave16, text):
    MFA_TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=MFA_TMP) as d:
        d = Path(d)
        write_wav(d / "a.wav", wave16, 16000)
        (d / "a.lab").write_text(" ".join(tokens(text)), encoding="utf-8")   # 学習と同じく小文字・記号なし
        cmd = MFA_CMD + ["align_one", str(d / "a.wav"), str(d / "a.lab"), MFA_DICT, MFA_ACOUSTIC,
                         str(d / "a.TextGrid")] + MFA_EXTRA
        env = dict(os.environ, MFA_ROOT_DIR=MFA_ROOT_DIR)
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        if r.returncode != 0 or not (d / "a.TextGrid").exists():
            raise RuntimeError("MFA が失敗しました：" + (r.stderr or r.stdout)[-400:])
        return read_textgrid_words(d / "a.TextGrid")


def match_words(sent_words, mfa_words):
    """文の単語と MFA の単語を、順番どおりに対応させる。対応しない単語は None"""
    b = [normalize(w) for _, _, w in mfa_words]
    sm = difflib.SequenceMatcher(a=sent_words, b=b, autojunk=False)
    spans = [None] * len(sent_words)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):   # <unk> などで綴りが違っても、数が同じなら対応させる
            for k in range(i2 - i1):
                spans[i1 + k] = (mfa_words[j1 + k][0], mfa_words[j1 + k][1])
    return spans


# ================================================================
# モデル（学習のノートブック 2026_1121_train_word_WavLM_att と同じ形。POOLING = "att"）
# ================================================================
if not MOCK:
    import librosa
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class WordAccentClassifier(nn.Module):
        """層の重み付き和 → FFN（768→256→256）→ attention pooling → 出力層（現象ごと）"""

        def __init__(self, n_layers=13, dim=768, hid=256, n_phen=5, dropout=0.25):
            super().__init__()
            self.layer_w = nn.Parameter(torch.zeros(n_layers))
            self.ffn = nn.Sequential(
                nn.Linear(dim, hid), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hid, hid), nn.GELU(), nn.Dropout(dropout),
            )
            self.att = nn.Linear(hid, 1)
            self.drop = nn.Dropout(dropout)
            self.heads = nn.Linear(hid, n_phen)

        def forward(self, x):                                          # x: [1, 13, T, 768]（1 単語ずつ）
            w = torch.softmax(self.layer_w, dim=0)
            h = (w[None, :, None, None] * x).sum(dim=1)                # [1, T, 768]
            h = self.ffn(h)                                            # [1, T, 256]
            att = torch.softmax(self.att(h).squeeze(-1), dim=1)        # [1, T]
            pooled = (att.unsqueeze(-1) * h).sum(dim=1)                # [1, 256]
            return self.heads(self.drop(pooled))                       # [1, 5]

    print("WavLM を読み込み中…", MODEL_NAME, DEVICE)
    WAVLM = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    CLASSIFIERS = []
    for k in range(N_FOLDS):
        f = RESULT_DIR / f"model_fold{k}.pt"
        m = WordAccentClassifier(n_phen=len(PHENOMENA))
        m.load_state_dict(torch.load(f, map_location="cpu"), strict=True)   # 形が違えばここでエラーになる
        CLASSIFIERS.append(m.to(DEVICE).eval())
    print(f"単語単位モデル {len(CLASSIFIERS)} 個を読み込みました：{RESULT_DIR}")

    def utterance_gain(x, target_rms=0.05):
        # 学習と同じ：前後の無音を除いた発話の RMS が 0.05 になる倍率（小さすぎる音は増幅しない）
        yt, _ = librosa.effects.trim(x, top_db=30)
        rms = float(np.sqrt(np.mean(yt ** 2))) if yt.size > 0 else 0.0
        return target_rms / rms if rms >= 0.01 else 1.0

    @torch.inference_mode()
    def p_us_of_word(wave16, gain, start, end):
        s = max(0, int(round((start - MARGIN) * 16000)))
        e = min(len(wave16), int(round((end + MARGIN) * 16000)))
        seg = np.clip(wave16[s:e] * gain, -1.0, 1.0).astype(np.float32)
        if len(seg) < 400:
            seg = np.pad(seg, (0, 400 - len(seg)))
        out = WAVLM(torch.from_numpy(seg)[None].to(DEVICE), output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1)                     # [1, 13, T, 768]
        hs = hs.half().float()                                         # 学習のときは float16 で保存していたので、そろえる
        probs = [torch.sigmoid(m(hs))[0].cpu().numpy() for m in CLASSIFIERS]
        return np.mean(probs, axis=0)                                  # 5 fold の平均。[5]（現象ごとの P(米)）

    def to16k(x, sr):
        return x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)


# ================================================================
# 1 回分の録音を判定する
# ================================================================
LOCK = threading.Lock()                            # MFA と GPU は 1 つずつ順番に使う


def score_recording(item, x, sr, target):
    words = tokens(item["text"])
    if MOCK:
        # 試運転：区間は等分、P(米) は録音の中身から決まる適当な数
        dur = len(x) / sr
        spans = [(dur * i / len(words), dur * (i + 1) / len(words)) for i in range(len(words))]
        seed = int(hashlib.md5(x[:: max(1, len(x) // 2000)].tobytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        p_all = {t["i"]: rng.uniform(0.05, 0.95, len(PHENOMENA)) for t in item["targets"]}
        time.sleep(1.0)
    else:
        x16 = to16k(x, sr)
        spans = match_words(words, mfa_align(x16, item["text"]))
        gain = utterance_gain(x16)
        p_all = {t["i"]: p_us_of_word(x16, gain, *spans[t["i"]]) for t in item["targets"] if spans[t["i"]]}

    out = []
    for t in item["targets"]:
        r = {"i": t["i"], "w": t["w"], "ph": t["ph"], "span": spans[t["i"]], "p_us": None, "score": None}
        if t["i"] in p_all:
            p = p_all[t["i"]]
            r["p_us"] = {ph: float(p[PHENOMENA.index(ph)]) for ph in t["ph"]}
            # 目標への近さ：目標がアメリカなら P(米)、イギリスなら 1 − P(米)
            r["score"] = {ph: (v if target == "us" else 1.0 - v) for ph, v in r["p_us"].items()}
        out.append(r)
    return out


# ================================================================
# 練習文を選ぶ（VCTK から。お手本の音声も付ける）
#   ① 練習する本番の単語を含む文      … グループ（A / B）ごとに、items.json の practice_words の単語
#   ② 本番にない同じ音の単語を含む文  … n_other 個（BATH は items.json の fixed_other）
#   どちらも、本番の文そのものと、練習しない本番の単語（どの現象でも）を含む文は使わない
#   選ぶ順番：英米で少ない方の読んだ人数が多い → 英米の合計が多い → その音の単語が多い → 短い
# ================================================================
def scan_vctk():
    """VCTK の全部の文 → {正規化した文: {"text", "uk": [音声], "us": [音声]}}。VCTK がなければ空"""
    info = VCTK_DIR / "speaker-info.txt"
    if not info.exists():
        print("VCTK が見つからないので、練習文は items.json の fallback を使います（お手本なし）：", VCTK_DIR)
        return {}
    accent = {}
    with open(info, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.split()
            if len(p) >= 4:
                accent[p[0] if not p[0][0].isdigit() else "p" + p[0]] = p[3]
    sents = {}
    for txt in sorted((VCTK_DIR / "txt").glob("*/*.txt")):
        acc = {"English": "uk", "American": "us"}.get(accent.get(txt.parent.name))
        text = txt.read_text(encoding="utf-8").strip()
        d = sents.setdefault(normalize(text), {"text": text, "uk": [], "us": [], "spk": set()})
        audio = VCTK_DIR / "wav48_silence_trimmed" / txt.parent.name / f"{txt.stem}_mic1.flac"
        if acc and audio.exists() and txt.parent.name not in d["spk"]:
            d[acc].append(audio)
            d["spk"].add(txt.parent.name)
    print(f"VCTK：{len(sents)} 種類の文を読みました")
    return sents


def usable(norm, text):
    n = len(norm.split())
    return (5 <= n <= 16 and norm not in TEST_NORM
            and not re.search(r"[0-9\-–—]", text))          # 数字やハイフンを含む文は、学習と同じく使わない


def rank_key(d, n_hits):
    return (-min(len(d["uk"]), len(d["us"])), -(len(d["uk"]) + len(d["us"])), -n_hits, len(d["text"].split()))


def build_practice():
    sents = scan_vctk()
    used = {}                                            # グループ → もう使った文（同じ文を 2 回使わない）

    def add(group, ph, kind, text, refs, word=None):
        sid = f"{group}_{ph}_{kind}{len(PRACTICE[group][ph]) + 1}"
        it = {"id": sid, "text": text, "kind": "practice", "phenomenon": ph, "practice_type": kind,
              "targets": targets_of(text, only_ph=ph), "practiced_word": word, "group": group}
        if not it["targets"]:
            print("注意：練習文に対象語がありません", sid, text)
        note = [v for k, v in ITEMS.get("notes", {}).items() if k in tokens(text)]
        if note:
            it["note"] = " ".join(note)
        PRACTICE[group][ph].append(it)
        used.setdefault(group, set()).add(normalize(text))
        ITEM_BY_ID[sid] = it
        REFS[sid] = {"uk": refs["uk"][:N_REF], "us": refs["us"][:N_REF]} if refs else {"uk": [], "us": []}

    for group, words in ITEMS["practice_words"].items():
        PRACTICE[group] = {ph: [] for ph in PHENOMENA}
        practiced = set(words.values())
        unpracticed = TEST_WORDS - practiced
        for ph in PHENOMENA:
            # ① 練習する本番の単語を含む文
            w = words.get(ph)
            if w:
                best = None
                for norm, d in sents.items():
                    toks = norm.split()
                    if not usable(norm, d["text"]) or set(toks) & unpracticed or norm in used.get(group, ()):
                        continue
                    hits = [h for h in find_targets(toks) if h[0] == ph]
                    if any(h[1] == w for h in hits):
                        k = rank_key(d, len(hits))
                        if best is None or k < best[0]:
                            best = (k, norm, d)
                if best:
                    add(group, ph, "p", best[2]["text"], best[2], word=w)
                else:
                    add(group, ph, "p", ITEMS["fallback"]["practiced"][w], None, word=w)
            # ② 本番にない同じ音の単語を含む文
            if ph in ITEMS.get("fixed_other", {}):
                for text in ITEMS["fixed_other"][ph]:
                    d = sents.get(normalize(text))
                    add(group, ph, "o", d["text"] if d else text, d)
            elif TEST_WORDS_BY_PH.get(ph):
                cand = []
                for norm, d in sents.items():
                    toks = norm.split()
                    if not usable(norm, d["text"]) or set(toks) & TEST_WORDS or norm in used.get(group, ()):
                        continue
                    hits = [h for h in find_targets(toks) if h[0] == ph]
                    if hits:
                        cand.append((rank_key(d, len(hits)), norm, d))
                cand.sort(key=lambda c: c[0])
                picked = [c for c in cand[:ITEMS["n_other"]]]
                if picked:
                    for _, norm, d in picked:
                        add(group, ph, "o", d["text"], d)
                else:
                    for text in ITEMS["fallback"]["other"].get(ph, [])[:ITEMS["n_other"]]:
                        add(group, ph, "o", text, None)

    for group in PRACTICE:
        for ph, lst in PRACTICE[group].items():
            for it in lst:
                r = REFS[it["id"]]
                print(f"練習文 {it['id']}（英 {len(r['uk'])}・米 {len(r['us'])}）："
                      f"{it['text']}　対象語：{', '.join(t['w'] for t in it['targets'])}")


# ================================================================
# 保存
# ================================================================
SAVE_DIR.mkdir(exist_ok=True)
LOG_LOCK = threading.Lock()


def append_jsonl(name, obj):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK, open(SAVE_DIR / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ================================================================
# HTTP サーバー
# ================================================================
class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/api/health":
            return self._json(200, {"ok": True, "mock": MOCK})
        if u.path == "/api/items":
            refs = {sid: {a: len(v) for a, v in d.items()} for sid, d in REFS.items()}
            return self._json(200, {
                "phenomena": PHENOMENA, "names": ITEMS["names"], "tips": ITEMS["tips"],
                "test": TEST, "test_words": TEST_WORDS_BY_PH,
                "practice_words": ITEMS["practice_words"], "practice": PRACTICE, "refs": refs})
        if u.path == "/api/ref":
            try:
                f = REFS[q["item"]][q["accent"]][int(q["n"])]
            except (KeyError, ValueError, IndexError):
                return self._json(404, {"ok": False, "error": "お手本がありません"})
            data = f.read_bytes()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "audio/flac")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            if u.path == "/api/score":
                return self._json(200, self.handle_score(q, body))
            if u.path == "/api/log":
                pid = q.get("pid", "")
                if not NAME_RE.match(pid):
                    raise ValueError("ID が正しくありません")
                append_jsonl("events.jsonl", {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": pid,
                                              "event": json.loads(body.decode("utf-8"))})
                return self._json(200, {"ok": True})
            return self._json(404, {"ok": False, "error": "not found"})
        except Exception as e:                       # 画面にエラーの内容を返す（サーバーは止めない）
            print("エラー:", repr(e))
            return self._json(400, {"ok": False, "error": str(e)})

    def handle_score(self, q, body):
        pid, target, stage, iid = q.get("pid", ""), q.get("target", ""), q.get("stage", ""), q.get("item", "")
        attempt = int(q.get("attempt", "1"))
        group = q.get("group", "")
        if not NAME_RE.match(pid):
            raise ValueError("ID が正しくありません")
        if target not in ("uk", "us") or stage not in STAGES or iid not in ITEM_BY_ID or group not in ("", "A", "B"):
            raise ValueError("target / stage / item が正しくありません")
        item = ITEM_BY_ID[iid]
        x, sr = read_wav(body)
        if len(x) < sr * 0.5:
            raise ValueError("録音が短すぎます")

        stamp = time.strftime("%Y%m%d-%H%M%S")
        folder = SAVE_DIR / pid
        folder.mkdir(parents=True, exist_ok=True)
        wav_name = f"{stage}_{iid}_a{attempt}_{stamp}.wav"
        (folder / wav_name).write_bytes(body)

        t0 = time.time()
        with LOCK:
            words = score_recording(item, x, sr, target)
        sec = time.time() - t0
        rec = {"time": stamp, "pid": pid, "group": group, "target": target, "stage": stage, "item": iid, "attempt": attempt,
               "wav": f"{pid}/{wav_name}", "sec": round(sec, 2), "words": words}
        append_jsonl("results.jsonl", rec)
        print(f"{pid} {stage} {iid} #{attempt}：{sec:.1f} 秒  " +
              " ".join(f"{w['w']}={'-' if w['score'] is None else round(np.mean(list(w['score'].values())) * 100)}"
                       for w in words))
        return {"ok": True, **rec}

    def log_message(self, fmt, *args):             # アクセスのたびの表示は出さない
        pass


if __name__ == "__main__":
    for s_ in TEST:
        print(f"本番 {s_['id']}：対象語 {', '.join(t['w'] + '(' + '・'.join(t['ph']) + ')' for t in s_['targets'])}")
    build_practice()
    print(f"保存先：{SAVE_DIR}")
    print(f"サーバーを起動しました：http://localhost:{PORT}" + ("　【MOCK：試運転】" if MOCK else ""))
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
