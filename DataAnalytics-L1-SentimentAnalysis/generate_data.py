"""Generates a synthetic product-review / tweet-style sentiment dataset (positive / negative / neutral).
Includes noise on purpose: mentions, URLs, hashtags, slang, negation ('not good'), mixed opinions and a few
mislabelled rows. To use a real dataset (e.g. Kaggle 'Twitter Sentiment Analysis' or 'IMDB'), save it as
sentiment_reviews.csv with columns: text, sentiment."""
import numpy as np, pandas as pd, random
rng = random.Random(21); nrng = np.random.default_rng(21)
subjects = ["phone","laptop","delivery","customer service","battery","app","packaging","screen","headphones",
            "build quality","store staff","price","camera","charger","website","speaker","keyboard","tablet"]
pos = ["amazing","excellent","fantastic","great","wonderful","superb","perfect","brilliant","impressive","delightful","outstanding","awesome","smooth","reliable"]
neg = ["terrible","awful","horrible","disappointing","poor","useless","broken","frustrating","pathetic","disgusting","unreliable","cheap","slow","overpriced"]
neu = ["okay","average","ordinary","standard","regular","fine","so-so","as described","nothing special","acceptable","decent","normal"]
fillers = ["ordered it on Monday","my brother has one too","came in a brown box","bought it last week","delivery took 3 days",
           "used it for a month","got it from the online store","after two weeks of use","I paid 200 for it","as a first time buyer",
           "for daily use","compared to my old one","at this price","from the local shop","just saying"]
tags_p = ["#happy","#loveit","#recommended","#win"]; tags_n = ["#fail","#worst","#disappointed","#refund"]; tags_u = ["#review","#new","#unboxing","#tech"]
P = ["The {s} is {a}!","I really love the {s}, it is {a}.","{a} {s}, would buy again","Very happy with the {s}. {a}!","Highly recommend: the {s} is {a}","Honestly the {s} is {a} and works well","So pleased, {s} is {a} :)"]
N = ["The {s} is {a}.","I hate the {s}, totally {a}","Waste of money, the {s} is {a}","Very unhappy with the {s}. {a}!","Would not recommend, the {s} is {a}","Honestly the {s} is {a} and keeps failing","So annoyed, {s} is {a} :("]
U = ["The {s} is {a}.","Received the {s} today. It is {a}.","The {s} arrived on Tuesday.","Bought the {s}, it is {a}.","The {s} works {a}.","Tried the {s}, {a}, nothing more to say"]
def sent(label):
    s = rng.choice(subjects)
    if label == "positive":
        r = rng.random()
        if r < .10: return f"Not bad at all, the {s} is {rng.choice(pos)}"
        return rng.choice(P).format(s=s, a=rng.choice(pos))
    if label == "negative":
        r = rng.random()
        if r < .10: return f"The {s} is not {rng.choice(pos)}"
        return rng.choice(N).format(s=s, a=rng.choice(neg))
    r = rng.random()
    if r < .12: return f"The {s} is not {rng.choice(neg)}, not {rng.choice(pos)} either"
    return rng.choice(U).format(s=s, a=rng.choice(neu))
def make(label):
    t = sent(label)
    r = rng.random()
    if r < .13:  # mixed opinion: label follows the last clause (ambiguous on purpose)
        other = "positive" if label != "positive" else "negative"
        t = (sent(other) + ", but " + t[0].lower() + t[1:]) if rng.random() < .5 else (t + ", but " + sent(other).lower())
    if rng.random() < .45: t += ", " + rng.choice(fillers)
    if rng.random() < .35: t += " " + rng.choice({"positive":tags_p,"negative":tags_n,"neutral":tags_u}[label] if rng.random() < .8 else tags_u)
    if rng.random() < .15: t = "@" + rng.choice(["SupportTeam","ShopOnline","TechStore"]) + " " + t
    if rng.random() < .08: t += " http://t.co/" + "".join(rng.choices("abcdefghij0123456789", k=7))
    if rng.random() < .06: t = t.upper()
    if rng.random() < .10: t = t.replace("!", "!!!")
    return t
labels = nrng.choice(["positive","negative","neutral"], size=3200, p=[.40,.33,.27])
rows = [(make(l), l) for l in labels]
df = pd.DataFrame(rows, columns=["text","sentiment"]).drop_duplicates("text").reset_index(drop=True)
# label noise: 3% of rows get a wrong label (simulates human annotation errors)
idx = nrng.choice(len(df), int(.03*len(df)), replace=False)
df.loc[idx,"sentiment"] = [rng.choice([x for x in ["positive","negative","neutral"] if x != df.loc[i,"sentiment"]]) for i in idx]
df.insert(0,"id",range(1,len(df)+1))
df.to_csv("sentiment_reviews.csv", index=False); print(df.shape); print(df.sentiment.value_counts())
