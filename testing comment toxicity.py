import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import pickle

loaded=CountVectorizer(decode_error='replace',vocabulary=pickle.load(open('word_features.pkl','rb')))

#test_comment='Food was really boring'
#test_comment=test_comment.split('delimiter')
#result=loaded.transform(test_comment)
#print(result)

#train_df = pd.read_csv('train.csv')
#test_df = pd.read_csv('test.csv')

user_df = pd.DataFrame(columns = ['comment_text'])
user_df

inp_comment =  str(input("enter a comment : "))
inp_comment

new_row = {'comment_text':inp_comment}
#append a new row to the dataframe formed by user inputs
user_df = user_df.append(new_row,ignore_index = True)


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

# clean the comment_text in train_df [Thanks to Pulkit Jha for the useful pointer.]
#train_df['comment_text'] = train_df['comment_text'].map(lambda com : clean_text(com))

# clean the comment_text in test_df [Thanks, Pulkit Jha.]
#test_df['comment_text'] = test_df['comment_text'].map(lambda com : clean_text(com))

# clean the comment_text in user_df [Thanks, Pulkit Jha.]
user_df['comment_text'] = user_df['comment_text'].map(lambda com : clean_text(com))
user_df


#train_text = train_df['comment_text']
#test_text = test_df['comment_text']
user_text = user_df['comment_text']


#train_features = loaded.transform(train_text)
# transform the test data using the earlier fitted vocabulary, into a document-term matrix
#test_features = loaded.transform(test_text)

user_features = loaded.transform(user_text)
print(user_features)

cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# create submission file
#submission_binary = pd.read_csv('sample_submission.csv')
#submission_binary.columns

lst= []
mapper = {}
for label in cols_target:
    filename = str(label+'_model.sav')
    filename
    model = pickle.load(open(filename, 'rb'))

    print('... Processing {}'.format(label))
    user_y_prob = model.predict_proba(user_features)[:,1]
    print(label,":",user_y_prob[0])
    lst.append([label,user_y_prob])
    #submission_binary[label] = test_y_prob
print(lst)
#[['obscene', array([0.05674509])],
# ['insult', array([0.07231278])], ['toxic', array([0.10434529])], ['severe_toxic', array([0.01385649])], ['identity_hate', array([0.01463043])], ['threat', array([0.00484142])]]

final=[]
for i in lst:
    if i[1]>0.5:
        final.append(i[0])

if not len(final):
    text = "The comment is clean"
else:
    text=final

print(text)










 
