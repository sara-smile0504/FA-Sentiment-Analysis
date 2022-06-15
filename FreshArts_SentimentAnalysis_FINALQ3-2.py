# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 20:31:25 2022

@author: moreno-m
"""

#importing necessary libraries
from afinn import Afinn
import pandas as pd
import numpy  as np
import sys
import pandas as pd
import google.auth

""" The following package may not be installed, if so, use the following """
""" conda install -c conda-forge google-cloud-language                   """
from   google.cloud       import language
from   tkinter.messagebox import showinfo, showwarning, askquestion

#establish dataframe / import data
excel_file = "Q3-Sara-AFINNmanual.xlsx"
Fresh_df = pd.read_excel(excel_file)
Fresh_df.fillna('', inplace=True)
Fresh_df.head()

#AFINN Analysis on Response
""" *********************************************************************** """
"""               Initiate AFINN                                            """
#instantiate afinn
afn = Afinn()
  
print("**********AFINN Sentiment Analysis Running***********")

Fresh_df['AFINN_py_sentiment']=Fresh_df['Response'].apply(afn.score)

print(Fresh_df)

""" *********************************************************************** """
"""               Define Google Sentiment                                   """

def get_sentimentApply(row):
    client = language.LanguageServiceClient(credentials=credentials)
    file_type="Response"
    verbose=False
    
    if type(row[1]) != str:
        return ''
    
    if file_type=="Response":
        document = language.Document(content=row[1], language='en',
                                 type_=language.Document.Type.PLAIN_TEXT)
    else:
        document = language.Document(content=row[1], language='en',
                                 type_=language.Document.Type.HTML)
    response = client.analyze_sentiment(document=document, encoding_type = 'UTF32')
    sentiment = response.document_sentiment
    return pd.Series([sentiment.score, sentiment.magnitude])

""" *********************************************************************** """
"""               Verify Google Cloud Credentials                           """
google_project_file = "GoogleTranslateCredentials.json"
credentials, project_id = google.auth.\
                          load_credentials_from_file(google_project_file)
                          
client = language.LanguageServiceClient(credentials=credentials)

""" *********************************************************************** """
"""              Apply Google Sentiment to Data Frame                       """

print("**********Google Sentiment Analysis Running***********")
Fresh_df[['GoogleSentimentScore', 'GoogleMagnitudeScore']] = Fresh_df.apply(get_sentimentApply, axis=1)
print(Fresh_df)


""" *********************************************************************** """
"""              Print Average Results for each Score                       """

print("**********Q3 AVERAGE RESULTS******")
print("------AFINN RESULTS------")
print('AFINN Score Manual:', np.round(Fresh_df['AFINN_Score'].mean(),decimals=3))
print('AFINN Sentiment Words Manual:', np.round(Fresh_df['AFINN_Sentiment_Words'].mean(),decimals=3))
print('AFINN Sentiment Manual:', np.round(Fresh_df['AFINN_Sentiment'].mean(),decimals=3))
print('AFINN Sentiment Python:', np.round(Fresh_df['AFINN_py_sentiment'].mean(),decimals=3))


print("------GOOGLE RESULTS------")
print('Google Sentiment Score:', np.round(Fresh_df['GoogleSentimentScore'].mean(),decimals=3))
print('Google Magnitude Score:', np.round(Fresh_df['GoogleMagnitudeScore'].mean(),decimals=3))

""" *********************************************************************** """
"""              Save DataFrame to csv                       """

Fresh_df.to_csv('FreshArts_SentimentAnalyses_q3.csv')