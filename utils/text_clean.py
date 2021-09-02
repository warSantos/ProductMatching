import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Remove pontos, números e stopwords.
def clean_text(sentences, accept_num=True):

    stop_words = {word: True for word in stopwords.words("english")}
    table = str.maketrans("", "", string.punctuation)
    tokens = []
    for text in sentences:
        words = []
        for word in text.lower().translate(table).split():
            #if word not in stop_words:
            if (not word.isnumeric() or accept_num) and word not in stop_words:
                words.append(word)
        tokens.append(words)
    return tokens