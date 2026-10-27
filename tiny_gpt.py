"""A tiny GPT from scratch: numpy only, forward pass, backward pass and training.

It learns to write English words letter by letter from the 210,773 lowercase words of
``/usr/share/dict/words`` (Webster's Second International, 1934, public domain), joined into one
long text with "." between them. The Bot Academy video "GPT from scratch" shows this file.

    python tiny_gpt.py train      # about 800,000 parameters, about an hour
    python tiny_gpt.py check      # gradient check against brute force
    python tiny_gpt.py sample     # words from the trained model
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RUN = Path(__file__).parent / "run"

# --- text to tokens ----------------------------------------------------------------------------

chars = ".abcdefghijklmnopqrstuvwxyz"
stoi = {c: i for i, c in enumerate(chars)}


def encode(text):
    return [stoi[c] for c in text]


def decode(tokens):
    return "".join(chars[t] for t in tokens)


WORDS = os.environ.get("WORDS", "/usr/share/dict/words")  # any list of words, one per line


def load_words():
    words = Path(WORDS).read_text().split()
    return [w for w in words if w.isalpha() and w.islower()]


# --- the model ---------------------------------------------------------------------------------

V, T, C, H, L = len(chars), 24, 128, 8, 4  # vocabulary, context, width, heads, layers
D = C // H  # width of one head


def init(seed=0):
    rng = np.random.default_rng(seed)

    def w(*shape):
        return rng.normal(0, 0.02, shape)

    p = {"tok": w(V, C), "pos": w(T, C), "lnf_g": np.ones(C), "lnf_b": np.zeros(C), "out": w(C, V)}
    for i in range(L):
        p |= {
            f"{i}.ln1_g": np.ones(C), f"{i}.ln1_b": np.zeros(C),
            f"{i}.q": w(C, C), f"{i}.k": w(C, C), f"{i}.v": w(C, C), f"{i}.o": w(C, C),
            f"{i}.ln2_g": np.ones(C), f"{i}.ln2_b": np.zeros(C),
            f"{i}.up": w(C, 4 * C), f"{i}.up_b": np.zeros(4 * C),
            f"{i}.down": w(4 * C, C), f"{i}.down_b": np.zeros(C),
        }  # fmt: skip
    return p


def layer_norm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + 1e-5)
    return xhat * g + b, (xhat, var)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def split(x):  # (B, n, C) -> (B, H, n, D)
    B, n, _ = x.shape
    return x.reshape(B, n, H, D).swapaxes(1, 2)


def merge(x):  # (B, H, n, D) -> (B, n, C)
    B, _, n, _ = x.shape
    return x.swapaxes(1, 2).reshape(B, n, C)


def attention(x, p, i):
    q = split(x @ p[f"{i}.q"])
    k = split(x @ p[f"{i}.k"])
    v = split(x @ p[f"{i}.v"])
    scores = q @ k.swapaxes(2, 3) / np.sqrt(D)
    n = x.shape[1]
    later = np.triu(np.ones((n, n), bool), 1)
    scores[..., later] = -np.inf  # no peeking
    weights = softmax(scores)
    mixed = merge(weights @ v)
    out = mixed @ p[f"{i}.o"]
    return out, (q, k, v, weights, mixed)


def mlp(x, p, i):
    a = x @ p[f"{i}.up"] + p[f"{i}.up_b"]
    r = np.maximum(a, 0)  # negatives to zero
    out = r @ p[f"{i}.down"] + p[f"{i}.down_b"]
    return out, (a, r)


def forward(p, tokens):
    n = tokens.shape[1]
    x = p["tok"][tokens] + p["pos"][:n]
    cache = []
    for i in range(L):
        g, b = p[f"{i}.ln1_g"], p[f"{i}.ln1_b"]
        h1, ln1 = layer_norm(x, g, b)
        att, at = attention(h1, p, i)
        x = x + att
        g, b = p[f"{i}.ln2_g"], p[f"{i}.ln2_b"]
        h2, ln2 = layer_norm(x, g, b)
        m, ml = mlp(h2, p, i)
        x = x + m
        cache.append((h1, ln1, at, h2, ln2, ml))
    hf, lnf = layer_norm(x, p["lnf_g"], p["lnf_b"])
    logits = hf @ p["out"]
    return logits, (tokens, cache, hf, lnf)


def loss_fn(logits, targets):
    probs = softmax(logits)
    i = targets[..., None]
    right = np.take_along_axis(probs, i, -1)
    return -np.log(right).mean(), probs


# --- the backward pass: the chain rule, step by step, in reverse --------------------------------


def layer_norm_back(dy, g, saved):
    xhat, var = saved
    dg, db = (dy * xhat).sum((0, 1)), dy.sum((0, 1))
    d = dy * g
    d = d - d.mean(-1, keepdims=True)
    d = d - xhat * (dy * g * xhat).mean(-1, keepdims=True)
    return d / np.sqrt(var + 1e-5), dg, db


def backward(p, saved, probs, targets):
    tokens, cache, hf, lnf = saved
    g = {}
    dlogits = probs - np.eye(V)[targets]
    dlogits /= targets.size
    g["out"] = flat(hf).T @ flat(dlogits)
    d = dlogits @ p["out"].T
    dx, dg, db = layer_norm_back(d, p["lnf_g"], lnf)
    g["lnf_g"], g["lnf_b"] = dg, db
    for i in reversed(range(L)):
        h1, ln1, at, h2, ln2, (a, r) = cache[i]
        q, k, v, weights, mixed = at
        # the small network, backwards
        g[f"{i}.down_b"] = dx.sum((0, 1))
        g[f"{i}.down"] = flat(r).T @ flat(dx)
        da = (dx @ p[f"{i}.down"].T) * (a > 0)
        g[f"{i}.up_b"] = da.sum((0, 1))
        g[f"{i}.up"] = flat(h2).T @ flat(da)
        d = da @ p[f"{i}.up"].T
        dh2, dg, db = layer_norm_back(d, p[f"{i}.ln2_g"], ln2)
        g[f"{i}.ln2_g"], g[f"{i}.ln2_b"] = dg, db
        dx = dx + dh2
        # attention, backwards
        g[f"{i}.o"] = flat(mixed).T @ flat(dx)
        dmixed = split(dx @ p[f"{i}.o"].T)
        dweights = dmixed @ v.swapaxes(2, 3)
        dv = weights.swapaxes(2, 3) @ dmixed
        dsum = (dweights * weights).sum(-1, keepdims=True)
        dscores = weights * (dweights - dsum) / np.sqrt(D)
        dq = dscores @ k
        dk = dscores.swapaxes(2, 3) @ q
        dh1 = 0
        for name, d in (("q", dq), ("k", dk), ("v", dv)):
            d = merge(d)
            g[f"{i}.{name}"] = flat(h1).T @ flat(d)
            dh1 = dh1 + d @ p[f"{i}.{name}"].T
        dh1, dg, db = layer_norm_back(dh1, p[f"{i}.ln1_g"], ln1)
        g[f"{i}.ln1_g"], g[f"{i}.ln1_b"] = dg, db
        dx = dx + dh1
    g["tok"] = np.zeros_like(p["tok"])
    np.add.at(g["tok"], tokens, dx)
    g["pos"] = np.zeros_like(p["pos"])
    g["pos"][: tokens.shape[1]] = dx.sum(0)
    return g


def flat(x):  # (B, n, width) -> (B * n, width)
    return x.reshape(-1, x.shape[-1])


# --- training: nudge every number a little downhill, again and again ---------------------------


def batches(data, rng, size=64):
    while True:
        starts = rng.integers(0, len(data) - T - 1, size)
        x = np.stack([data[s : s + T] for s in starts])
        y = np.stack([data[s + 1 : s + T + 1] for s in starts])
        yield x, y


def adam(p, g, state, step, lr, b1=0.9, b2=0.99):
    for name in p:
        m, v = state.get(name, (0, 0))
        m = b1 * m + (1 - b1) * g[name]
        v = b2 * v + (1 - b2) * g[name] ** 2
        state[name] = (m, v)
        m_hat = m / (1 - b1**step)
        v_hat = v / (1 - b2**step)
        p[name] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def generate(p, text=".", temp=1.0, rng=None):
    rng = rng or np.random.default_rng()
    tokens = encode(text)
    for _ in range(30):
        context = np.array([tokens[-T:]])
        logits, _ = forward(p, context)
        probs = softmax(logits[0, -1] / temp)
        token = rng.choice(V, p=probs)
        tokens.append(token)
        if token == 0:  # a dot ends the word
            break
    return decode(tokens)


def sample_words(p, k, seed, temperature=1.0):
    rng = np.random.default_rng(seed)
    return [generate(p, ".", temperature, rng).strip(".") for _ in range(k)]


def evaluate(p, data, rng, n=8):
    test_batches = batches(data, rng, 256)
    losses = []
    for _ in range(n):
        x, y = next(test_batches)
        losses.append(loss_fn(forward(p, x)[0], y)[0])
    return float(np.mean(losses))


def split_text(seed=0):
    """The words in a fixed shuffle: 95% to train on, 5% held back to test."""
    words = load_words()
    order = np.random.default_rng(seed).permutation(len(words))
    cut = int(0.95 * len(words))
    train_text = "." + ".".join(words[j] for j in order[:cut]) + "."
    test_text = "." + ".".join(words[j] for j in order[cut:]) + "."
    return words, np.array(encode(train_text)), np.array(encode(test_text))


SNAPSHOTS = [0, 50, 100, 200, 400, 700, 1000, 2000, 4000, 8000, 15000, 30000, 45000, 60000]


def train(steps=60000, lr=1e-3, seed=0):
    words, data, test = split_text(seed)
    rng = np.random.default_rng(seed)
    rng.permutation(len(words))  # the shuffle above used the first draws of this seed
    p = init(seed)
    print(
        f"{len(words):,} words, {len(data):,} letters, {sum(v.size for v in p.values()):,} numbers"
    )
    state, log, snapshots = {}, [], {}
    RUN.mkdir(exist_ok=True)
    began = time.time()
    for step, (x, y) in enumerate(batches(data, rng)):
        if step in SNAPSHOTS:
            test_loss = evaluate(p, test, np.random.default_rng(1))
            snapshots[step] = {"test_loss": test_loss, "words": sample_words(p, 24, seed=7)}
            np.savez(RUN / f"step{step:05d}.npz", **p)
            print(step, round(test_loss, 3), " ".join(snapshots[step]["words"][:8]))
        if step == steps:
            break
        logits, saved = forward(p, x)
        loss, probs = loss_fn(logits, y)
        g = backward(p, saved, probs, y)
        rate = lr * (1 + np.cos(np.pi * step / steps)) / 2
        adam(p, g, state, step + 1, rate)
        log.append(float(loss))
    took = time.time() - began
    (RUN / "log.json").write_text(json.dumps({"loss": log, "snapshots": snapshots, "s": took}))
    print(f"{steps} steps in {took:.0f} s")


def gradient_check():
    global T, C, H, D, L
    T, C, H, L = 5, 8, 2, 2
    D = C // H
    rng = np.random.default_rng(4)
    p = {k: v + rng.normal(0, 0.3, v.shape) for k, v in init(3).items()}
    x, y = rng.integers(0, V, (3, T)), rng.integers(0, V, (3, T))
    logits, saved = forward(p, x)
    _, probs = loss_fn(logits, y)
    g = backward(p, saved, probs, y)
    worst = 0.0
    for name in p:
        flat = p[name].reshape(-1)
        for j in rng.choice(flat.size, min(6, flat.size), replace=False):
            old = flat[j]
            flat[j] = old + 1e-5
            up = loss_fn(forward(p, x)[0], y)[0]
            flat[j] = old - 1e-5
            down = loss_fn(forward(p, x)[0], y)[0]
            flat[j] = old
            numeric, exact = (up - down) / 2e-5, g[name].reshape(-1)[j]
            worst = max(worst, abs(numeric - exact) / max(1e-8, abs(numeric) + abs(exact)))
    print(f"{len(p)} weight arrays, worst relative gradient error: {worst:.1e}")


def load(step=60000):
    with np.load(RUN / f"step{step:05d}.npz") as f:
        return {k: f[k] for k in f.files}


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "sample"
    if command == "check":
        gradient_check()
    elif command == "train":
        train()
    else:
        print(" ".join(sample_words(load(), 30, seed=int(time.time()))))
