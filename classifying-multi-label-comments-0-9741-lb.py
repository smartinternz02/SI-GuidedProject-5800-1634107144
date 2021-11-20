#importing some of the required libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re
import pickle


# We've seperate data available for training and testing.
# Load training and test data
train_df = pd.read_csv('Data/train.csv')
test_df = pd.read_csv('Data/test.csv')

#DATA DESCRIPTION

# In the training data, the comments are labelled as one or more of the 
# six categories; toxic, severe toxic, obscene, threat, insult and identity hate. 
# This is essentially a multi-label classification problem.
cols_target = ['insult','toxic','severe_toxic','identity_hate','threat','obscene']

# check for null comments in test_df
print(test_df.isnull().any())
# no null values in test_df


# All rows in the training and test data contain comments, so there's no need to clean 
# up null fields.

# Next, let's examine the correlations among the target variables.
data = train_df[cols_target]

colormap = plt.cm.plasma
plt.figure(figsize=(7,7))
plt.title('Correlation of features & targets',y=1.05,size=14)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,
           linecolor='white',annot=True)

# Indeed, it looks like some of the labels are higher correlated, e.g.
# insult-obscene has the highest at 0.74, followed by toxic-obscene and toxic-insult.
# This co-relation can be taken in use for other kind of algorithms, such as multi-chain labelling 



#DATA Pre-Processing

# Define a function to clean up the comment text, basic NLP
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

# clean the comment_text in both the datasets.
train_df['comment_text'] = train_df['comment_text'].map(lambda com : clean_text(com))
test_df['comment_text'] = test_df['comment_text'].map(lambda com : clean_text(com))

# Define all_text from entire train & test data for use in tokenization by Vectorizer
train_text = train_df['comment_text']
test_text = test_df['comment_text']
all_text = pd.concat([train_text, test_text])

# Vectorize the data
# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
word_vect = CountVectorizer(
                            strip_accents='unicode',
                            analyzer='word',
                            token_pattern=r'\w{1,}',
                            stop_words='english',
                            ngram_range=(1, 1)
                            )
    


# learn the vocabulary in the training data, then use it to create a document-term matrix
word_vect.fit(all_text)

# transform the data using the earlier fitted vocabulary, into a document-term matrix
train_features = word_vect.transform(train_text)
test_features = word_vect.transform(test_text)


#saving word vectorizer vocab as pkl file to be loaded afterwards
pickle.dump(word_vect.vocabulary_,open('word_feats.pkl','wb'))


# Solving a multi-label classification problem
# One way to approach a multi-label classification problem is to transform the problem into 
# separate single-class classifier problems. This is known as 'problem transformation'. 
# There are three methods:
    
# Binary Relevance : This is probably the simplest which treats each label as a separate 
# single classification problems. The key assumption here though, is that there are no correlation
# among the various labels.

# Classifier Chains : In this method, the first classifier is trained on the input X.
# Then the subsequent classifiers are trained on the input X and all previous classifiers' 
# predictions in the chain. This method attempts to draw the signals from the correlation among 
# preceding target variables.

# Label Powerset:  This method transforms the problem into a multi-class problem  where the
# multi-class labels are essentially all the unique label combinations. In our case here, where
# there are six labels, Label Powerset would in effect turn this into a 2^6 or 64-class problem. 


# Binary Relevance - build a multi-label classifier using Logistic Regression
# Model Building and Fitting

# import and instantiate the Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression(C=12.0)

# create submission file
submission_binary = pd.read_csv('sample_submission.csv')

mapper = {}
for label in cols_target:
    mapper[label] = logreg
    filename = str(label+'_model.sav')
    print(filename)
    print('... Processing {}'.format(label))
    y = train_df[label]
    # train the model using train_features & y
    mapper[label].fit(train_features, y)

    #saving the fitted model for class "label"
    pickle.dump(mapper[label], open(filename, 'wb'))

    # compute the training accuracy
    y_pred_X = mapper[label].predict(train_features)
    print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
    # compute the predicted probabilities for X_test_dtm
    test_y_prob = mapper[label].predict_proba(test_features)[:,1]
    submission_binary[label] = test_y_prob


# generate submission file
submission_binary.to_csv('submission_binary.csv',index=False)

