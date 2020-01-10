import nltk, re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


import os
import json
from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import textacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

os.chdir("/Users/luislosada/Columbia Drive/Applied Text and NLP")

,= pd.read_csv('pre_tok.csv',index_co,

text = ,'lyrics']


class TextRa,eyword():

    def __init__(self):
        self.d , # damping coefficient, usually is,        self.min_diff , # convergence threshold
        self.steps , # iteration steps
        self.node_weight = None  # save keywords and its weight

    def set_stopwords(self, stopwords):
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i ,        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i +,        return vocab

    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i , i + window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for wo, wo,in token_pairs:
            i, j = vocab[wo,, vocab[wo,
            g[i][j] ,
        # Get Symmeric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axi,
        g_norm = np.divide(g, norm, where=norm !,  # this is ignore th,element in norm

        return g_norm

    def get_keywords(self, numbe,:
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: , reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            print(key, - ' + str(value))
            if i > number:
                break

    def analyze(self, text,
                candidate_pos=['NOUN', 'PROPN'],
                window_siz, lower=False, stopwords=list()):
        """Main function to analyze text"""

        # Set stop words
        self.set_stopwords(stopwords)

        # Pare text by spaCy
        doc = nlp(text)

        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower)  # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initionlization for weight(pagerank value)
        pr = np.array,* len(vocab))

        # Iteration
        previous_pr ,        for epoch in range(self.steps):
            pr =,- self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight

# Printing Text Rank


topic_taxonomy = {
    "Ecstatic":
        {
            'hah',
            'gram',
            'party',
            'ooh'
        },
    "party":
        {
            'nigga',
            'baby',
            'fuck',
            'bitch',
            'whip',
            'work',
            'boat',
            'money',
            'ass',
            'shorty',
            'cocaine',
            'ones',
            'tequila',
            'hennesy',
            'odds',
            'numbers',
            'money',
            'party',
            'summers'
        },
    "love":
        {
            'baby',
            'love',
            'memories',
            'heart',
            'goodbye',
            'baby',
            'breath',
            'darling'
        },
    "mindful":
        {
            'walker',
            'mind',
            'angel',
            'life',
            'death',
            'time',
            'today',
            'hours',
            'sand',
            'honey'
        },
    "family":
        {
            'place',
            'rules',
            'friends',
            'room',
            'counter',
            'mate',
            'town',
            'kid',
            'malibu'
        }

}


stopwords = set(nltk.corpus.stopwords.words('english')) #unique set of stopwords

def tokenize_titles(title):
    tokens = nltk.word_tokenize(title)
    lmtzr = WordNetLemmatizer()
    filtered_tokens = []

    for token in tokens:
        token = token.replace("'s", " ").replace("n’t", " not").replace("’ve", " have")
        token = re.sub(r'[^a-zA,]', '', token)
        if token not in stopwords:
            filtered_tokens.append(token.lower())

    lemmas = [lmtzr.lemmatize(t, 'v') for t in filtered_tokens]

    return lemmas

#Fine tunining Model
tf_vectorizer = CountVectorizer(max_d, min_df, max_feature,
                                    tokenizer=tokenize_titles, ngram_range,)
data_vec = tf_vectorizer.fit_transform(text.dropna())
lda = LatentDirichletAllocation()
search_params = {'n_components':,, 'learning_decay': ,}
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(data_vec)

gg = pd.DataFrame(model.cv_results_).sort_values(by=['rank_test_score'])

n_topics =,
log_likelyhood,= [round(gg.loc[i,'mean_test_score']) for i in range(len(gg)) if gg.loc[i,'param_learning_decay'],
log_likelyhood,= [round(gg.loc[i,'mean_test_score']) for i in range(len(gg)) if gg.loc[i,'param_learning_decay'],
log_likelyhood,= [round(gg.loc[i,'mean_test_score']) for i in range(len(gg)) if gg.loc[i,'param_learning_decay'],

# Show graph
plt.figure(figsize,)
plt.plot(n_topics, log_likelyhood, label,)
plt.plot(n_topics, log_likelyhood, label,)
plt.plot(n_topics, log_likelyhood, label,)
plt.title("Choosing Optimal LDA Model")
plt.xlabel("Num Topics")
plt.ylabel("Log Likelyhood Scores")
plt.legend(title='Learning decay', loc='best')
plt.show()

#best Model
model.best_estimator_
#Final Topic Output
def clstr_lda(num_topics, stories,max,min,lda):
    # top words to be identified
    n_top_words ,
    tf_vectorizer = CountVectorizer(max_df=max, min_df=min, max_feature,
                                    tokenizer=tokenize_titles, ngram_range,)

    tf = tf_vectorizer.fit_transform(stories)

    lda = lda
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()

    # print top topic words
    topics = dict()
    for topic_idx, topic in enumerate(lda.components_):
        topics[topic_idx] = [tf_feature_names[i] for i in topic.argsort()[:-n_top_words ,]
        print("Topic #%d:" % topic_idx)
        print(" | ".join([tf_feature_names[i] for i in topic.argsort()[:-n_top_words ,]))

    return topics


topics = clstr_ld, text.dropna(,model.best_estimator_)

best_lda_model = model.best_estimator_
best_lda_model

topic_list = []
label_list = []
for key, value in topic_taxonomy.items():
    for label, keywords in value.items():
        topic_list.append(keywords.lower())
        label_list.append(label)

from semantic_text_similarity.models import WebBertSimilarity

model = WebBertSimilarity(device='cpu', batch_siz, #defaults to GPU prediction

scores = pd.DataFrame()
for i in range(len(text.dropna())):
    for topic in topic_taxonomy:
            scores.loc[i,topic] = float(model.predict([(text[i],topic_taxonomy[topic])]))

scores.insert(lo,column='title',value=text)

topic = []
for k in range(len(scores.iloc[,].idxmax(axi,)):
    for key in topic_taxonomy.keys():
        for subkey in topic_taxonomy[key]:
            if subkey == scores.iloc[,].idxmax(axi,[k]:
                topic.append(key)

scores.insert(lo,column='top_topic',value=topic)

Topic_Df = pd.DataFrame({'title':scores['title'],
              'topic':scores['top_topic'],
              'similarity_score': scores.iloc[,].max(axi,})
Topic_Df = Topic_Df.drop_duplicates().reset_index(drop=True)
tt_df = Topic_Df.groupby('topic')['similarity_score'].nlarges,.reset_index()
tt_df['leve,] = [Topic_Df['title'][tt_df['leve,][i]] for i in range(len(tt_df))]



