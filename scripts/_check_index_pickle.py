import pickle
import sys

sys.path.insert(0, r"e:\Paper\SciCite-REAL")

with open(r"e:\Paper\SciCite-REAL\data\index\scicite_bm25l_train.pkl", "rb") as fh:
    index = pickle.load(fh)

print("class:", type(index).__name__)
print("corpus size:", len(index.corpus))
m = index._model
print("model class:", type(m).__name__)
print("model module:", type(m).__module__)
