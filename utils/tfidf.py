from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf(
    docs,
    strip_accents="ascii",
    use_idf=True,
    stop_words="english",
    ngram_range=(1, 1),
    min_df=1,
):
    tfidf_vectorizer = TfidfVectorizer(
        strip_accents=strip_accents,
        use_idf=use_idf,
        stop_words=stop_words,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    return tfidf_vectorizer.fit_transform(docs)
