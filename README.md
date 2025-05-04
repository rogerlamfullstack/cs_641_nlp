# cs_641_nlp
This is the practical code in Natural Language Processing lecture.
# Install Environment
# local
pip install -r requirement.txt
python -c "import nltk; nltk.download('punkt')"
python -m spacy download en_core_web_sm
# colab
!pip install nltk spacy "transformers>=4.39.3" torch spacy datasets scikit-learn "accelerate>=0.27.2" numpy==1.26.4<br>
!python -c "import nltk; nltk.download('punkt')"<br>
!python -m spacy download en_core_web_sm

# For each python script, it represents the example of NLP. You should try it one by one.

