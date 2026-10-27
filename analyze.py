"""Numbers and pictures from the trained tiny GPT for the video: writes ``run/facts.json``.

python analyze.py
"""

import json

import numpy as np
import tiny_gpt as g

p = g.load()
words, data, test = g.split_text()
dictionary = set(words)
facts: dict[str, object] = {}

# The simplest guesser: count which letter follows which (a bigram table).
counts = np.ones((g.V, g.V))
np.add.at(counts, (data[:-1], data[1:]), 1)
bigram = counts / counts.sum(1, keepdims=True)
facts["bigram_test_loss"] = float(-np.log(bigram[test[:-1], test[1:]]).mean())
brng = np.random.default_rng(7)


def bigram_word() -> str:
    t, out = 0, []
    while True:
        t = int(brng.choice(g.V, p=bigram[t]))
        if t == 0 or len(out) > 25:
            return "".join(out)
        out.append(g.chars[t])


facts["bigram_words"] = [bigram_word() for _ in range(12)]
facts["bigram_after_q"] = {g.chars[j]: float(bigram[g.stoi["q"], j]) for j in range(g.V)}

# The tiny GPT's own test loss, and how many of its words are new.
facts["gpt_test_loss"] = float(g.evaluate(p, test, np.random.default_rng(1), 32))
facts["uniform_loss"] = float(np.log(g.V))
sampled = g.sample_words(p, 1000, seed=21)
novel = [w for w in sampled if w and w not in dictionary]
facts["novel_share"] = len(novel) / len(sampled)
facts["novel_examples"] = novel[:60]
facts["real_examples"] = [w for w in sampled if w in dictionary][:30]
facts["parameters"] = {k: int(v.size) for k, v in p.items()}
facts["parameter_total"] = int(sum(v.size for v in p.values()))

# Temperature: the same seeds at three temperatures.
facts["temperature"] = {
    str(t): g.sample_words(p, 12, seed=5, temperature=t) for t in (0.5, 1.0, 1.5)
}

# Prompts: completions of a start.
facts["prompts"] = {
    s: [g.generate(p, "." + s, 0.8, np.random.default_rng(k)).strip(".") for k in range(8)]
    for s in ("pyth", "robo", "comput", "neur", "bot")
}


# The next-letter probabilities for a few starts.
def next_probs(text: str) -> dict[str, float]:
    logits, _ = g.forward(p, np.array([g.encode(text)[-g.T :]]))
    probs = g.softmax(logits[0, -1])
    return {g.chars[j]: float(probs[j]) for j in np.argsort(-probs)[:8]}


facts["next"] = {s: next_probs("." + s) for s in ("q", "th", "chemistr", "pyth", "robo", "x", "")}

# Letter embeddings in two dimensions (PCA): do the vowels end up together?
emb = p["tok"] - p["tok"].mean(0)
_, _, vt = np.linalg.svd(emb, full_matrices=False)
xy = emb @ vt[:2].T
facts["embedding_2d"] = {c: [float(a), float(b)] for c, (a, b) in zip(g.chars, xy, strict=True)}
vowels = [g.stoi[c] for c in "aeiou"]
cons = [i for i in range(1, g.V) if i not in vowels and g.chars[i] != "y"]
unit = p["tok"] / np.linalg.norm(p["tok"], axis=1, keepdims=True)
sim = unit @ unit.T
facts["cos_vowel_vowel"] = float(np.mean([sim[a, b] for a in vowels for b in vowels if a != b]))
facts["cos_vowel_consonant"] = float(np.mean([sim[a, b] for a in vowels for b in cons]))
facts["cos_cons_cons"] = float(np.mean([sim[a, b] for a in cons for b in cons if a != b]))
facts["nearest"] = {
    c: [g.chars[j] for j in np.argsort(-sim[g.stoi[c]])[1:5]] for c in "aeiouqxzpst."
}


# Attention weights of every head for a few words (the last row is the next-letter guess).
def attention_maps(text: str) -> list[list[list[list[float]]]]:
    tokens = np.array([g.encode(text)])
    _, cache = g.forward(p, tokens)
    return [layer[2][3][0].round(4).tolist() for layer in cache[1]]


facts["attention"] = {w: attention_maps("." + w) for w in ("chemistry", "dinosaur", "programming")}
(g.RUN / "facts.json").write_text(json.dumps(facts, indent=1))
for k, v in facts.items():
    if k not in ("attention", "embedding_2d", "parameters"):
        print(k, json.dumps(v)[:400])
