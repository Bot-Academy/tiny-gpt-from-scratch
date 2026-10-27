# tiny-gpt-from-scratch

A GPT in one file of plain Python and numpy: the forward pass, the backward pass (every
gradient written out by hand, no autograd), and the training loop. It reads 210,773 English words
letter by letter and learns to make up new ones.

It is the code from the video
"ChatGPT Explained from Scratch using Python" on [Bot Academy](https://www.youtube.com/@BotAcademyYT).

```
python tiny_gpt.py sample
bespark schistotherapy gummi rebrokenness rigmainess freighter overcurst salacer fluxilite horseless
```

## Run it

```bash
pip install numpy
python tiny_gpt.py sample   # 30 words from the trained model in run/step60000.npz
python tiny_gpt.py check    # checks every hand-written gradient against brute force
python tiny_gpt.py train    # trains from scratch (about an hour on a laptop)
```

Training reads `/usr/share/dict/words`, which macOS ships (Webster's Second International, 1934).
On another system, point it at any list of words, one per line:

```bash
WORDS=my-words.txt python tiny_gpt.py train
```

To finish a word you start, give `generate` a dot and the first letters:

```python
import tiny_gpt as g
p = g.load()
print(g.generate(p, ".pyth", temp=0.8))
```

## What's inside

| | |
|---|---|
| tokens | 27: the letters a to z, and "." between words |
| looks back | 24 letters |
| numbers per letter | 128 |
| attention heads | 8 per block |
| blocks | 4 (attention, then a small network, each with layer norm and a shortcut) |
| parameters | 801,280 |
| training | 60,000 steps of 64 windows, Adam, a cosine learning-rate schedule |

The file reads top to bottom in the order of the video: text to tokens, the model (`attention`,
`mlp`, `forward`), the loss, the backward pass, training, and sampling.

## Results

Loss is how surprised the model is by the right next letter, on words it never trained on
(5% held back):

| guesser | test loss |
|---|---|
| any letter, equally likely | 3.30 |
| counting which letter follows which | 2.53 |
| tiny GPT after 60,000 steps | 1.72 |

67% of the words it writes (674 of 1,000) aren't in the dictionary: *splashedly*, *banglement*,
*exylite*, *petriscid*. The rest are real words it rebuilt from the patterns it learned.

`run/` holds the trained weights (`step60000.npz`), the loss of every training step
(`log.json`), and what `analyze.py` measured for the video (`facts.json`). Training took 54 minutes
on an Apple M5 laptop and is deterministic: the same seed gives the same model.

## License

MIT
