# Autocomplete and Autocorrect Data Analytics

> **Oasis Infobyte SIP · Data Analytics · Level 2 · Task 5**
> Analysing the efficiency and accuracy of autocomplete and autocorrect algorithms using NLP.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![NLP](https://img.shields.io/badge/NLP-n--grams%20%7C%20edit%20distance-green) ![Status](https://img.shields.io/badge/status-complete-brightgreen)

---

## 1. Project Overview

Every smartphone keyboard predicts the next word you will type (**autocomplete**) and silently fixes your typos (**autocorrect**).
This project rebuilds simplified versions of both from scratch on a real text corpus, then **measures** how well each approach works.

| Part | What I built | Core technique |
|---|---|---|
| Autocomplete | Next-word and prefix prediction | Frequency-based **bigram**, **trigram** and **trigram-with-backoff** models |
| Autocorrect | Spelling correction | **Edit distance** (Levenshtein / Norvig), `pyspellchecker`, and a **context-aware** re-ranker |
| Analytics | Precision, recall, accuracy, F1, confusion matrix, speed | Evaluated on unseen (held-out) text |

## 2. Objectives

- Collect a large text corpus and run a full NLP preprocessing pipeline
- Build n-gram autocomplete and show the top-3 predictions for 10+ prefixes
- Build edit-distance autocorrect and test it on 20+ misspelled words
- Define and compute **precision and recall** for both systems
- Compare at least two algorithms for each task
- Discuss limitations versus production systems such as Google Gboard

## 3. Dataset

| Item | Detail |
|---|---|
| Source | [Project Gutenberg](https://www.gutenberg.org) (public domain) |
| Books | *Pride and Prejudice*, *The Adventures of Sherlock Holmes*, *Alice's Adventures in Wonderland*, *Frankenstein* |
| Size | ~500K words (several hundred thousand tokens) |
| Split | 80 % training / 20 % testing, **split by sentence** so test data is never seen in training |

The notebook downloads the books automatically into `data/`. To use your own corpus, drop `.txt` files into `data/`.

## 4. Tech Stack

Python · pandas · NumPy · NLTK (stopwords) · pyspellchecker · matplotlib · seaborn · collections (`Counter`, `defaultdict`) · Jupyter Notebook

## 5. Repository Structure

```
OIBSIP/
└── DataAnalytics-L2-AutocompleteAutocorrect/
    ├── autocomplete_autocorrect_analysis.ipynb   # main notebook (all code + commentary)
    ├── README.md
    ├── requirements.txt
    ├── data/                                     # corpus (auto-downloaded)
    ├── top20_words.png                           # generated charts
    ├── autocomplete_comparison.png
    ├── autocorrect_comparison.png
    ├── autocorrect_confusion_matrix.png
    └── demo_video_link.txt                       # LinkedIn demo link
```

## 6. Methodology

### Step 1 - Preprocessing
Lowercasing → quote normalisation → sentence splitting → punctuation removal (regex) → tokenisation.
Stopwords are removed **only** for the word-frequency chart; they are kept for the models because predicting "of", "the" or "to" is exactly what an autocomplete system must do.

### Step 2 - Autocomplete (frequency-based n-grams)

| Model | Idea |
|---|---|
| Bigram | P(next word \| previous word) |
| Trigram | P(next word \| previous two words) |
| Trigram + backoff | Use trigram, fall back to bigram, then to the most common words, so a suggestion is always available |
| Prefix completion | Finish a half-typed word (e.g. `beau` → `beautiful`), ranked by context then frequency |

### Step 3 - Autocorrect (edit distance)

| Method | Idea |
|---|---|
| **Norvig corrector** | Generate all strings within edit distance 1 (then 2) of the typo (deletes, inserts, replaces, transposes); keep real words; choose the most frequent |
| **pyspellchecker** | Library Levenshtein-distance corrector with its own dictionary |
| **Context-aware** | Same candidates, but re-ranked by P(word) × P(word \| previous word) |

Levenshtein distance is also implemented from scratch (dynamic programming) for error analysis.

### Step 4 - Evaluation

**Autocomplete** (on unseen sentences - model sees the previous words and must predict the real next word):

| Metric | Meaning |
|---|---|
| Top-1 / Top-3 accuracy | True word is the first / one of three suggestions |
| Coverage | % of cases where the model had any suggestion |
| **Precision@3** | hits ÷ cases where suggestions were shown |
| **Recall@3** | hits ÷ all test cases |

**Autocorrect** (20 hand-picked typos + 400 auto-generated typos from held-out words):

| Metric | Meaning |
|---|---|
| Accuracy | Typos fixed correctly |
| **Precision** | Correct fixes ÷ all words the system changed |
| **Recall** | Correct fixes ÷ all typos |
| F1 | Harmonic mean of precision and recall |
| False-correction rate | Already-correct words that were wrongly altered |
| ms / word | Speed |

A **before/after confusion matrix** (correct vs. incorrect) is produced for each corrector.

## 7. Results

> Run the notebook, then paste your own numbers below. (Results vary slightly with the corpus and random seed.)

### Autocomplete
| Model | Top-1 | Top-3 (Recall@3) | Precision@3 | Coverage |
|---|---|---|---|---|
| Bigram | _ % | _ % | _ % | _ % |
| Trigram | _ % | _ % | _ % | _ % |
| Trigram + backoff | _ % | _ % | _ % | _ % |

### Autocorrect
| Method | Accuracy | Precision | Recall | F1 | False-correction | ms/word |
|---|---|---|---|---|---|---|
| Norvig | _ | _ | _ | _ | _ | _ |
| Context-aware | _ | _ | _ | _ | _ | _ |
| pyspellchecker | _ | _ | _ | _ | _ | _ |

### Visualisations
![Top words](top20_words.png)
![Autocomplete comparison](autocomplete_comparison.png)
![Autocorrect comparison](autocorrect_comparison.png)
![Confusion matrix](autocorrect_confusion_matrix.png)

## 8. Key Insights

1. **Zipf's law:** a small number of words make up most of the text, which is why frequency-based prediction works for common words but fails on the long tail.
2. **Precision vs. coverage trade-off:** trigrams are precise but rarely have enough data; bigrams cover more but guess worse; **backoff** balances both.
3. **Edit distance 1 fixes most typos**; distance 2 recovers more but is much slower (the candidate set explodes).
4. **Context ranking** helps when several candidates are equally close, at extra computational cost.
5. A corrector must be judged on **false corrections** too - changing a correct word is worse than missing a typo.

## 9. Limitations vs. Production Systems (e.g. Google Gboard)

| Area | This project | Production keyboard |
|---|---|---|
| Model | Count-based n-grams, 2 words of context | Neural language models using long context |
| Data | A few classic novels | Billions of modern messages, slang, names, emojis |
| Personalisation | None | Learns your vocabulary on-device (federated learning) |
| Error model | Uniform random typos | Keyboard geometry, touch-point probabilities, swipe typing |
| Real-word errors | Not detected (their/there) | Context-aware detection |
| Unknown words | Cannot correct out-of-vocabulary words | Sub-word / character-level models |
| Speed | Brute-force edit generation | Tries / FSTs, quantised models, <10 ms latency |

**Future work:** Kneser-Ney smoothing, keyboard-distance error model, BK-tree/trie lookup, small LSTM/Transformer language model, larger modern corpus.

## 10. How to Run

```bash
git clone https://github.com/<your-username>/OIBSIP.git
cd OIBSIP/DataAnalytics-L2-AutocompleteAutocorrect
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
jupyter notebook autocomplete_autocorrect_analysis.ipynb
```
Then choose **Kernel → Restart & Run All**. (First run needs internet to download the books.)

## 11. Task Checklist Mapping

- [x] Collect a large text corpus (Project Gutenberg)
- [x] NLP preprocessing: tokenisation, lowercasing, punctuation and stopword removal
- [x] Autocomplete: bigram/trigram frequency model
- [x] Tested on 10 prefixes with top-3 predictions
- [x] Autocorrect: edit-distance (custom Levenshtein/Norvig + pyspellchecker)
- [x] Tested on 20 misspelled words with accuracy measured
- [x] Precision and recall for both systems
- [x] Algorithm comparison (3 correctors, 3 autocomplete models)
- [x] Bar chart of top 20 words; confusion matrix for autocorrect
- [x] Discussion of limitations vs. Google Keyboard

## 12. Demo Video & LinkedIn

- First 2 seconds: static title card with **Full Name · Data Analytics · Autocomplete and Autocorrect Data Analytics**
- Then run the notebook end-to-end and narrate the outputs
- Post on LinkedIn, tag **Oasis Infobyte**, and include `#oasisinfobyte #dataanalytics #python #nlp #datascience #internship`

## 13. References

- Norvig, P. *How to Write a Spelling Corrector* - https://norvig.com/spell-correct.html
- Jurafsky & Martin, *Speech and Language Processing*, ch. 3 (N-gram Language Models)
- NLTK documentation - https://www.nltk.org
- pyspellchecker - https://pypi.org/project/pyspellchecker/
- Project Gutenberg - https://www.gutenberg.org

## Author
**Your Full Name** · [LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)
