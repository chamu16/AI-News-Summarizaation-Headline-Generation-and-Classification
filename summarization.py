import os
import glob
import string
import numpy as np
import pandas as pd
from article_category import TextCategory
from article_headline import TextHeadline
from article_summary import TextSummary, generate_summary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.model_selection import train_test_split

articles_path = 'News Articles'
summaries_path = 'Summaries'
categories_list = ['politics', 'sport', 'tech', 'entertainment', 'business']

def read_files_from_folders(articles_path, summaries_path, categories_list, encoding='ISO-8859-1'):
    articles = []
    summaries = []
    categories = []
    for category in categories_list:
        article_paths = glob.glob(os.path.join(articles_path, category, '*.txt'), recursive=True)
        summary_paths = glob.glob(os.path.join(summaries_path, category, '*.txt'), recursive=True)
        
        print(f'found {len(article_paths)} file in articles/{category}, {len(summary_paths)} file in summaries/{category}')
        
        if len(article_paths) != len(summary_paths):
            print('number of files is not equal')
            return
        for idx_file in range(len(article_paths)):
            categories.append(category)
            with open(article_paths[idx_file], mode = 'r', encoding = encoding) as file:
                articles.append(file.read())
                
            with open(summary_paths[idx_file], mode = 'r', encoding = encoding ) as file:
                summaries.append(file.read())
                
    print(f'total {len(articles)} file in articles folders, {len(summaries)} file in summaries folders')
    return articles, summaries, categories

articles, summaries, categories = read_files_from_folders(articles_path, summaries_path, categories_list) # type: ignore
df = pd.DataFrame({'articles': articles,'summaries': summaries, 'categories': categories})

article_headline = []
for i in df['articles']:
    art = i.split('\n\n')
    article_headline.append(art[0])
df['headlines'] = article_headline

df['articles'] = df['articles'].str.encode('ascii', 'ignore').str.decode('ascii')
df['summaries'] = df['summaries'].str.encode('ascii', 'ignore').str.decode('ascii')
df = df.dropna()

df['articles_length']=df['articles'].apply(lambda x: len(x.split()))
df['summaries_length']=df['summaries'].apply(lambda x: len(x.split()))

df_sample = df.sample(n = 150, random_state = 1)
train_df, test_df = train_test_split(df_sample, test_size = 0.2, random_state = 0, stratify = df_sample[['categories']])

training=train_df[['articles','summaries','categories','headlines']]

def clean_text(text):
    # Convert the text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text into words
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join the tokens back into a single string
    text = ' '.join(tokens)
    return text

cleaned_texts = []
for article in training['articles']:
    cleaned_text = clean_text(article)
    cleaned_texts.append(cleaned_text)
training.insert(loc=4, column='cleaned_text', value=cleaned_texts)
# training.loc[:, 'cleaned_texts'] = cleaned_texts
# training['cleaned_text'] = training['articles'].apply(clean_text)

def summarizeArticle(article):
    summary = TextSummary(article)
    sent_tok = summary._sentence_token()
    spell_corr = summary._spell_correction()
    lent = int(np.round((len(sent_tok))/2))
    art_summary = generate_summary(spell_corr,lent)
    return art_summary

def headlineArticle(article):
    headline = TextHeadline(article)
    art_headline = headline._keyword2text()
    return art_headline

def classifyArticle(article, training=training):
    category = TextCategory(article,training)
    art_category = category._predict_category()
    return art_category
