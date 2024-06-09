import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


class TextCategory:
    def __init__(self, text, training):
        self.text = text
        self.training = training
        self.vectorizer = self._vectorizer()


    # Define a function to preprocess and clean the text data
    def _clean_text(self):
        text = self.text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if not word in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        text = ' '.join(tokens)
        return text


    # Create a TfidfVectorizer to convert the text data into numerical vectors
    def _vectorizer(self):
        vectorizer = TfidfVectorizer()
        # Fit the vectorizer to the cleaned text column
        vectorizer.fit(self.training['cleaned_text'])
        return vectorizer


    # Transform the cleaned text column into a numerical vector representation
    def _clf(self):
        X = self.vectorizer.transform(self.training['cleaned_text'])
        y = self.training['categories']
        clf = MultinomialNB()
        clf.fit(X, y)
        return clf


    # Define a function to predict the category of a new news article
    def _predict_category(self):
        self.clf = self._clf()
        article_vector = self.vectorizer.transform([self._clean_text()])
        return self.clf.predict(article_vector)[0]