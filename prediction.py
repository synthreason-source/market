"""
aapl_prediction_single_file.py

Everything in one file, no local module imports:
  1. Download real AAPL daily closes.
  2. Symbolize returns into a small alphabet and pack them into
     combinatorial block tokens (so each "word" the model sees covers
     several real trading days).
  3. Train an n-gram model (the fuller version from ngram_model.py --
     bigram/trigram backoff + lexical-similarity/influence reweighting)
     directly on that block-token stream.
  4. Sample a continuation from the trained model.
  5. Decode the generated block tokens back into a synthetic price path.
  6. Export history.csv / prediction.csv / actual_future.csv for
     plot_prediction.m.

Only standard library is used (csv, math, random, re, urllib, dataclasses,
collections, pathlib, typing) -- no project-local imports.
"""

from __future__ import annotations

import csv
import math
import random
import re
import shutil
import subprocess
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ============================================================
# Config
# ============================================================

AAPL_CSV_URL = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
AAPL_CSV_LOCAL = "aapl_real.csv"

N_HOLDOUT = 15                       # bars held out to compare prediction vs reality
ALPHABET = ("D", ".", "U")           # down / flat / up
BLOCK_SIZE = 3                       # 1 block token = 3 trading days
SENTENCE_LENGTH = 20                 # purely cosmetic chunking, no trading meaning

TEMPERATURE = 0.8
TOP_K = 5
MIN_COUNT = 1
INFLUENCE_TAU = 0.6

RANDOM_SEED = 2026
random.seed(RANDOM_SEED)


# ============================================================
# 1. Real data download
# ============================================================

def download_real_prices(url: str = AAPL_CSV_URL, path: str = AAPL_CSV_LOCAL) -> List[float]:
    """Download real AAPL daily closes (cached locally after first run)."""
    if not Path(path).exists():
        urllib.request.urlretrieve(url, path)

    closes = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            closes.append(float(row["AAPL.Close"]))
    return closes


# ============================================================
# 2. Symbolizing raw price moves
# ============================================================

def price_to_returns(prices: Sequence[float]) -> List[float]:
    out = []
    for p0, p1 in zip(prices, prices[1:]):
        out.append(0.0 if p0 <= 0 or p1 <= 0 else math.log(p1 / p0))
    return out


@dataclass
class Symbolizer:
    """Maps returns to a small alphabet via equal-mass quantile thresholds,
    and records each bucket's mean return so a symbol can later be turned
    back into a representative price move.
    """

    symbols: Tuple[str, ...] = ALPHABET
    thresholds: List[float] = field(default_factory=list)
    representative_return: dict = field(default_factory=dict)

    def fit(self, returns: Sequence[float]) -> "Symbolizer":
        if not returns:
            raise ValueError("need at least one return to fit thresholds")
        n_buckets = len(self.symbols)
        sorted_r = sorted(returns)
        cuts = []
        for i in range(1, n_buckets):
            q = i / n_buckets
            idx = min(int(q * len(sorted_r)), len(sorted_r) - 1)
            cuts.append(sorted_r[idx])
        self.thresholds = cuts

        buckets = {s: [] for s in self.symbols}
        for r in returns:
            buckets[self._bucket_symbol(r)].append(r)
        bounds = [-math.inf] + cuts + [math.inf]
        self.representative_return = {}
        for i, s in enumerate(self.symbols):
            vals = buckets[s]
            if vals:
                self.representative_return[s] = sum(vals) / len(vals)
            else:
                lo, hi = bounds[i], bounds[i + 1]
                self.representative_return[s] = 0.0 if math.isinf(lo) or math.isinf(hi) else (lo + hi) / 2
        return self

    def _bucket_symbol(self, r: float) -> str:
        i = 0
        while i < len(self.thresholds) and r > self.thresholds[i]:
            i += 1
        return self.symbols[i]

    def transform(self, returns: Sequence[float]) -> List[str]:
        if not self.thresholds:
            raise RuntimeError("call fit() before transform()")
        return [self._bucket_symbol(r) for r in returns]


# ============================================================
# 3. Combinatorial block packing
# ============================================================

@dataclass
class BlockEncoder:
    """Packs `block_size` consecutive alphabet symbols into one token via a
    mixed-radix combinatorial index -- decode() is an exact inverse.
    Prefix is lowercase since downstream tokenization lowercases everything.
    """

    alphabet: Tuple[str, ...]
    block_size: int
    stride: Optional[int] = None
    prefix: str = "blk"

    def __post_init__(self):
        if self.stride is None:
            self.stride = self.block_size
        self._sym_to_i = {s: i for i, s in enumerate(self.alphabet)}
        self.vocab_size = len(self.alphabet) ** self.block_size

    def _encode_one(self, symbols: Sequence[str]) -> int:
        a = len(self.alphabet)
        index = 0
        for s in symbols:
            index = index * a + self._sym_to_i[s]
        return index

    def _decode_one(self, index: int) -> List[str]:
        a = len(self.alphabet)
        digits = []
        for _ in range(self.block_size):
            digits.append(index % a)
            index //= a
        digits.reverse()
        return [self.alphabet[d] for d in digits]

    def encode(self, symbols: Sequence[str]) -> List[str]:
        tokens = []
        L = self.block_size
        for start in range(0, len(symbols) - L + 1, self.stride):
            idx = self._encode_one(symbols[start:start + L])
            tokens.append(f"{self.prefix}{L}_{idx}")
        return tokens

    def decode(self, token: str) -> List[str]:
        body = token[len(self.prefix):]
        L_str, idx_str = body.split("_", 1)
        assert int(L_str) == self.block_size
        return self._decode_one(int(idx_str))


def build_block_corpus(
    prices: Sequence[float],
    alphabet: Tuple[str, ...],
    block_size: int,
    sentence_length: int,
) -> Tuple[List[str], BlockEncoder, Symbolizer]:
    """prices -> returns -> symbols -> block tokens (a single-level scheme,
    matching what this consolidated script actually trains on). Returns the
    raw token list (not a joined string -- the n-gram model ingests tokens
    directly, no re-tokenization needed).
    """
    returns = price_to_returns(prices)
    symbolizer = Symbolizer(symbols=alphabet).fit(returns)
    symbols = symbolizer.transform(returns)

    encoder = BlockEncoder(alphabet=alphabet, block_size=block_size, prefix="lvl1_")
    tokens = encoder.encode(symbols)
    return tokens, encoder, symbolizer


def decode_to_price_path(
    tokens: Sequence[str],
    encoder: BlockEncoder,
    symbolizer: Symbolizer,
    start_price: float,
) -> List[float]:
    """Decode generated block tokens back to base symbols, then to a
    synthetic price path. Non-block tokens (e.g. "<bos>"/"<eos>" that a
    generic n-gram vocabulary can contain) are silently skipped.
    """
    pattern = re.compile(rf"^{re.escape(encoder.prefix)}\d+_\d+$")
    valid_tokens = []
    for raw in tokens:
        cleaned = re.sub(r"[^0-9a-zA-Z_]+$", "", raw)  # strip trailing punctuation
        if pattern.match(cleaned):
            valid_tokens.append(cleaned)

    symbols = [s for t in valid_tokens for s in encoder.decode(t)]
    prices = [start_price]
    for s in symbols:
        r = symbolizer.representative_return.get(s, 0.0)
        prices.append(prices[-1] * math.exp(r))
    return prices


def export_history_and_prediction_csv(
    history_prices: Sequence[float],
    predicted_prices: Sequence[float],
    history_path: str = "history.csv",
    prediction_path: str = "prediction.csv",
) -> None:
    with open(history_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "price"])
        for i, p in enumerate(history_prices):
            w.writerow([i, p])

    start_idx = len(history_prices) - 1
    with open(prediction_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "price"])
        for i, p in enumerate(predicted_prices):
            w.writerow([start_idx + i, p])


# ============================================================
# MATLAB/Octave plotting -- embedded so this stays a single file.
# Writes plot_prediction.m alongside the CSVs, then tries to actually
# run it (MATLAB first, then Octave) so the PNG gets produced without
# a separate manual step.
# ============================================================

PLOT_PREDICTION_M = r"""
% plot_prediction.m (auto-written by aapl_prediction_single_file.py)
%
% Plots historical prices together with the model-generated predicted
% continuation, with a shaded band marking the forecast region and a
% marker at the handoff point where prediction begins.
%
% Expects history.csv / prediction.csv (and optional actual_future.csv)
% in the current directory, each with columns [idx, price].

history = csvread('history.csv', 1, 0);
prediction = csvread('prediction.csv', 1, 0);

hist_idx   = history(:, 1);
hist_price = history(:, 2);
pred_idx   = prediction(:, 1);
pred_price = prediction(:, 2);

have_actual = exist('actual_future.csv', 'file') == 2;
if have_actual
    actual = csvread('actual_future.csv', 1, 0);
    actual_idx = actual(:, 1);
    actual_price = actual(:, 2);
end

handoff_idx   = hist_idx(end);
handoff_price = hist_price(end);

figure('Color', 'w', 'Position', [100 100 900 500], 'Visible', 'off');
hold on;

y_lims_pad_prices = [hist_price; pred_price];
if have_actual
    y_lims_pad_prices = [y_lims_pad_prices; actual_price];
end
y_lims_pad = 0.03 * (max(y_lims_pad_prices) - min(y_lims_pad_prices));
y_min = min(y_lims_pad_prices) - y_lims_pad;
y_max = max(y_lims_pad_prices) + y_lims_pad;
fill([handoff_idx, pred_idx(end), pred_idx(end), handoff_idx], ...
     [y_min, y_min, y_max, y_max], ...
     [0.93 0.95 1.0], 'EdgeColor', 'none', 'HandleVisibility', 'off');

plot(hist_idx, hist_price, '-', 'Color', [0.15 0.15 0.15], 'LineWidth', 1.6, ...
     'DisplayName', 'Historical price');

plot(pred_idx, pred_price, '--', 'Color', [0.85 0.20 0.20], 'LineWidth', 1.8, ...
     'DisplayName', 'Model prediction');

if have_actual
    plot(actual_idx, actual_price, '-', 'Color', [0.15 0.45 0.15], 'LineWidth', 1.6, ...
         'DisplayName', 'Actual (held out)');
end

plot(handoff_idx, handoff_price, 'o', 'MarkerSize', 7, ...
     'MarkerFaceColor', [0.15 0.15 0.15], 'MarkerEdgeColor', 'k', ...
     'HandleVisibility', 'off');
text(handoff_idx, handoff_price, '  now', 'FontSize', 9, 'VerticalAlignment', 'bottom');

xlabel('Bar index');
ylabel('Price');
title('AAPL: historical price with n-gram model predicted continuation');
legend('Location', 'best');
ylim([y_min y_max]);
grid on;
box on;
hold off;

% print() (not saveas) works reliably with no display attached, which
% matters when this runs headless via `matlab -batch` or `octave --no-gui`.
print(gcf, 'prediction_chart.png', '-dpng', '-r150');
fprintf('Saved prediction_chart.png\n');
"""


def render_chart_with_matlab_or_octave(m_script_path: str = "plot_prediction.m") -> bool:
    """Write out the .m script, then actually try to run it so the PNG
    gets produced as part of this one script -- rather than leaving that
    as a separate manual step. Tries MATLAB first, falls back to Octave.
    Returns True if prediction_chart.png was successfully created.
    """
    Path(m_script_path).write_text(PLOT_PREDICTION_M)

    candidates = [
        ["matlab", "-batch", "run('plot_prediction.m')"],
        ["octave", "--no-gui", "--eval", "run('plot_prediction.m')"],
        ["octave-cli", "--eval", "run('plot_prediction.m')"],
    ]

    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            continue
        print(f"Running {cmd[0]} to render the chart...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except Exception as exc:
            print(f"  {cmd[0]} failed to run: {exc}")
            continue
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())
        if Path("prediction_chart.png").exists():
            print("prediction_chart.png created.")
            return True
        print(f"  {cmd[0]} ran but no prediction_chart.png appeared "
              f"(exit code {result.returncode}) -- trying next option if any.")

    print(
        "\nNo MATLAB or Octave installation found on PATH (checked: matlab, octave, "
        "octave-cli), so prediction_chart.png was NOT generated automatically.\n"
        "plot_prediction.m has been written to disk -- install Octave "
        "(https://octave.org/download) or MATLAB and run it manually:\n"
        "    octave --no-gui --eval \"run('plot_prediction.m')\"\n"
    )
    return False


# ============================================================
# 4. N-gram model (bigram/trigram backoff + lexical-similarity
#    reweighting), same design as ngram_model.py's NGramModel
# ============================================================

def safe_log(value: float, floor: float = 1e-12) -> float:
    return math.log(max(value, floor))


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class NGramModel:
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_count: int = MIN_COUNT
    influence_tau: float = INFLUENCE_TAU
    curve_k: float = 8.0
    curve_midpoint: float = 0.5

    unigram: Counter = field(default_factory=Counter)
    bigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    trigram: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    lexical_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    influence_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vocabulary: List[str] = field(default_factory=list)
    finalized: bool = False

    def ingest_tokens(self, tokens: List[str]) -> None:
        if not tokens:
            return
        sequence = ["<bos>", "<bos>"] + tokens + [self.eos_token]
        if len(sequence) < 3:
            return
        for token in sequence:
            self.unigram[token] += 1
        for left, right in zip(sequence, sequence[1:]):
            self.bigram[left][right] += 1
        for a, b, c in zip(sequence, sequence[1:], sequence[2:]):
            self.trigram[f"{a}\t{b}"][c] += 1
        self.finalized = False

    def finalize(self) -> None:
        self.vocabulary = sorted(t for t, c in self.unigram.items() if c >= self.min_count)
        if self.unk_token not in self.vocabulary:
            self.vocabulary.append(self.unk_token)

        token_contexts = defaultdict(Counter)
        for context, counts in self.bigram.items():
            for token, count in counts.items():
                token_contexts[token][context] += count

        self.lexical_vectors = {}
        for token in self.vocabulary:
            counts = token_contexts.get(token, Counter())
            total = sum(counts.values()) or 1
            self.lexical_vectors[token] = {ctx: c / total for ctx, c in counts.items()}

        self.influence_vectors = {}
        for source in self.vocabulary:
            source_vec = self.lexical_vectors.get(source, {})
            scores = {}
            for target in self.vocabulary:
                if source == target:
                    continue
                sim = cosine_similarity(source_vec, self.lexical_vectors.get(target, {}))
                if sim >= self.influence_tau:
                    scores[target] = sim
            self.influence_vectors[source] = scores
        self.finalized = True

    def _backoff_distribution(self, prev: str, prev_prev: Optional[str]) -> Dict[str, float]:
        if prev_prev is not None:
            counts = self.trigram.get(f"{prev_prev}\t{prev}")
            if counts:
                return self._normalize(counts)
        counts = self.bigram.get(prev)
        if counts:
            return self._normalize(counts)
        return self._normalize(self.unigram)

    @staticmethod
    def _normalize(counts: Counter) -> Dict[str, float]:
        total = sum(counts.values())
        return {t: c / total for t, c in counts.items()} if total else {}

    def _curve_weight(self, p_eos: float) -> float:
        p_eos = min(1.0, max(0.0, p_eos))
        z = self.curve_k * (p_eos - self.curve_midpoint)
        return 1.0 / (1.0 + math.exp(z))

    def _score_next_token(self, prev: str, prev_prev: Optional[str], candidate_limit: int = 64) -> Dict[str, float]:
        if not self.finalized:
            self.finalize()
        base = self._backoff_distribution(prev, prev_prev)
        if not base:
            return {}
        candidates = sorted(base, key=base.get, reverse=True)[:candidate_limit]
        source_vec = self.lexical_vectors.get(prev, {})
        influences = self.influence_vectors.get(prev, {})
        curve = self._curve_weight(base.get(self.eos_token, 0.0))
        scores = {}
        for token in candidates:
            similarity = cosine_similarity(source_vec, self.lexical_vectors.get(token, {}))
            influence = influences.get(token, 0.0)
            scores[token] = safe_log(base[token]) + curve * 0.35 * similarity + curve * 0.65 * influence
        return scores

    def _probabilities(self, prev: str, prev_prev: Optional[str], temperature: float, candidate_limit: int) -> Dict[str, float]:
        scores = self._score_next_token(prev, prev_prev, candidate_limit)
        if not scores:
            return {}
        temperature = max(temperature, 1e-5)
        scaled = {t: s / temperature for t, s in scores.items()}
        maximum = max(scaled.values())
        exps = {t: math.exp(s - maximum) for t, s in scaled.items()}
        total = sum(exps.values())
        return {t: v / total for t, v in exps.items()} if total else {}

    def sample_next(self, prev: str, prev_prev: Optional[str], temperature: float = 0.8, top_k: int = 5) -> str:
        probs = self._probabilities(prev, prev_prev, temperature, max(top_k, 1))
        if not probs:
            return self.eos_token
        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        tokens, weights = zip(*items)
        return random.choices(tokens, weights=weights, k=1)[0]


def generate_tokens(model: NGramModel, n_steps: int, temperature: float = TEMPERATURE, top_k: int = TOP_K) -> List[str]:
    """Repeatedly call sample_next to produce a token stream -- this
    NGramModel exposes single-step sample_next() rather than a generate()
    loop, so the loop lives here instead.
    """
    generated: List[str] = []
    prev, prev_prev = "<bos>", "<bos>"
    for _ in range(n_steps):
        token = model.sample_next(prev, prev_prev, temperature, top_k)
        if token == model.eos_token:
            break
        generated.append(token)
        prev_prev, prev = prev, token
    return generated


# ============================================================
# 5. Put it all together
# ============================================================

def main() -> None:
    prices = download_real_prices()
    print(f"Loaded {len(prices)} real AAPL daily closes "
          f"(range ${min(prices):.2f}-${max(prices):.2f})")

    train_prices = prices[:-N_HOLDOUT]
    actual_future = prices[-N_HOLDOUT:]

    tokens, encoder, symbolizer = build_block_corpus(
        train_prices, alphabet=ALPHABET, block_size=BLOCK_SIZE, sentence_length=SENTENCE_LENGTH
    )
    print(f"Block corpus: {len(tokens)} tokens (vocab size {encoder.vocab_size})")

    model = NGramModel(min_count=MIN_COUNT, influence_tau=INFLUENCE_TAU)
    model.ingest_tokens(tokens)
    model.finalize()
    print(f"Trained NGramModel: vocabulary={len(model.vocabulary)}, "
          f"bigram contexts={len(model.bigram)}, trigram contexts={len(model.trigram)}")

    n_tokens_needed = -(-N_HOLDOUT // BLOCK_SIZE)  # ceil(N_HOLDOUT / BLOCK_SIZE)
    generated_tokens = generate_tokens(model, n_steps=n_tokens_needed * 4)
    print(f"Generated {len(generated_tokens)} raw tokens: {generated_tokens}")

    predicted_path = decode_to_price_path(generated_tokens, encoder, symbolizer, start_price=train_prices[-1])

    export_history_and_prediction_csv(train_prices, predicted_path)
    print("Wrote history.csv and prediction.csv")

    with open("actual_future.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "price"])
        start_idx = len(train_prices) - 1
        w.writerow([start_idx, train_prices[-1]])
        for i, p in enumerate(actual_future, start=1):
            w.writerow([start_idx + i, p])
    print("Wrote actual_future.csv")

    render_chart_with_matlab_or_octave()


if __name__ == "__main__":
    main()
