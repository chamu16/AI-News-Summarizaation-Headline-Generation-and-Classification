import yake
from keytotext import pipeline

class TextHeadline:
    def __init__(self,doc):
        self.doc = doc


    # Extracting keywords from the article
    def _keywords(self):
        kw_extractor = yake.KeywordExtractor()
        keywords = kw_extractor.extract_keywords(self.doc)
        keys = []
        key_word = []
        for i in keywords:
            if i[1] > 0.1:
                keys.append(i)
        for kw in keys:
            key_word.append(kw[0])
        return key_word

    # Converting the keywords into sentence or headline
    def _keyword2text(self):
        list_key_word = self._keywords()
        nlp = pipeline("k2t-base")  #loading the pre-trained model
        params = {"do_sample":False, "num_beams":1, "no_repeat_ngram_size":0 , "early_stopping":False}    #decoding params
        headline = nlp(list_key_word, **params)  #keywords
        return headline