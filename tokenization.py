from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')

text = "We love NLP! It helps computers understand humans."

print("Word Tokenization:", word_tokenize(text))
print("Sentence Tokenization:", sent_tokenize(text))
