from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

words = ["meeting", "happily", "playing", "chocolates"]

porter = PorterStemmer()
snowball = SnowballStemmer("english")
lancaster = LancasterStemmer()

print("Porter:", [porter.stem(w) for w in words])
print("Snowball:", [snowball.stem(w) for w in words])
print("Lancaster:", [lancaster.stem(w) for w in words])

# Lemmatization with spaCy
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("rocks corpora better went happier")

print("Lemmas:", [token.lemma_ for token in doc])
