

import pandas as pd
import numpy  as np
from AdvancedAnalytics.Text import text_analysis, text_plot

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition           import LatentDirichletAllocation
from sklearn.decomposition           import TruncatedSVD
from sklearn.decomposition           import NMF
 
#from collections import Counter
#from PIL         import Image
 
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
 
df   = pd.read_excel("Reviews-Complete.xlsx")
text = df['Text']

Rep_words= ["art","artist","houston"]
ta   = text_analysis(synonyms=None, stop_words=Rep_words , pos=True, stem=True)
cv   = CountVectorizer(max_df=0.95, min_df=0.05, max_features=None, # If we have a word that is in 95% or less than 5% of documnets, we exclude them.
                       binary=False, analyzer=ta.analyzer) # There is no limit in Maximum number of features that we pull in match the maximum number of words.
#"Binary=False" means the frequency of each word and each document will ve counted, if "Binary=Ture" then we have 1 and 0 for the frequency of each word.
tf    = cv.fit_transform(text) # This will apply this Text Analysis model to the data
terms = cv.get_feature_names() # this gives you a long list of names 

# Constants
m_features      = None  # default is None. number of features we are working
n_topics        = 5   # number of topics
max_iter        = 10   # maximum number of iterations
max_df          = 0.95  # max proportion of docs/reviews allowed for a term
learning_offset = 10.      # default is 10
learning_method = 'online' # alternative is 'batch' for large files
tfidf = True  # Use TF-IDF Weighting if True.  meaning those frequency will be weighted
svd   = False # Use SVD topic factorization if True Single Value Decomposition. It usually operates poorly in Python

def sortSecond(e):
    return e[1]
# Show Word Cloud based on TFIDF weighting
if tfidf == True:
    # Construct the TF/IDF matrix from the data
    print("\nConducting Term/Frequency Matrix using TF-IDF")
    # Default for norm is 'l2', use norm=None to supress
    tfidf_vect = TfidfTransformer(norm=None, use_idf=True)
    # tf matrix is (n_reviews)x(m_features)
    tf = tfidf_vect.fit_transform(tf) 
    
    term_idf_sums = tf.sum(axis=0)
    term_idf_scores = []
    for i in range(len(terms)):
        term_idf_scores.append([terms[i], term_idf_sums[0,i]])
    print("The Term/Frequency matrix has", tf.shape[0],\
          " rows, and", tf.shape[1], " columns.")
    print("The Term list has", len(terms), " terms.")
    term_idf_scores.sort(key=sortSecond, reverse=True)
    print("\nTerms with Highest TF-IDF Scores:")
    term_cloud= {}
    n_terms = len(terms)
    for i in range(n_terms):
        term_cloud[term_idf_scores[i][0]] = term_idf_scores[i][1]
        if i < 10:
            print('{:<15s}{:>8.2f}'.format(term_idf_scores[i][0], 
              term_idf_scores[i][1]))
            
dcomp='lda'
if dcomp == 'lda':
    # LDA Analysis
    uv = LatentDirichletAllocation(n_components=n_topics, 
                                   max_iter=max_iter,
                            learning_method=learning_method, 
                            learning_offset=learning_offset, 
                            random_state=12345)
if dcomp == 'svd':
    # In sklearn, SVD is synonymous with LSA (Latent Semantic Analysis)
    uv = TruncatedSVD(n_components=n_topics, algorithm='arpack',
                                    tol=0, random_state=12345)
   
if dcomp == 'nmf':
    uv = NMF(n_components=n_topics, random_state=12345, alpha=0.1, 
             l1_ratio=0.5)

if dcomp == 'kld':
    uv = NMF(n_components=n_topics, random_state=12345, alpha=0.1,    
             l1_ratio=0.5, beta_loss='kullback-leibler', solver='mu', 
             max_iter=1000)
    
U = uv.fit_transform(tf)
# Display the topic selections
print("\n********** GENERATED TOPICS **********")
text_analysis.display_topics(uv, terms, n_terms=10, word_cloud=True)

# Store predicted topic and its probability in array <topics>
n_reviews = df.shape[0]
# Initialize <topics> to all zeros
topics    = np.array([0]*n_reviews, dtype=float)
# Assign topics to reviews based on largest row value in U
df_topics = text_analysis.score_topics(U, display=True)
df        = df.join(df_topics)

### Creating a table #####
df1       = df.groupby('topic')['Text'].count()
df_topics = df.groupby('topic')[['A', 'B']].mean()
df_topics = df_topics.join(df1)
df_topics['percent'] = \
                  100*df_topics['Text']/df_topics['Text'].sum()
print("\nTopic  Reviews  Percent")
print("-------------------------")
for i in range(n_topics):
    print("{:>3d}{:>9d}{:>9.1f}%".format(i, 

                                            df_topics['Text'].loc[i],
                                            df_topics['percent'].loc[i]))
print("-------------------------\n")


