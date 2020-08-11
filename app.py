from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv('train.csv')
    df.head()
    X=df.drop('label',axis=1)
    X.head()
    y=df['label']
    y.head()
    df.shape
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    df=df.dropna()
    news=df.copy()
    news.reset_index(inplace=True)
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    ps=PorterStemmer()
    corpus=[]
    for i in range(0, len(news)):
        detect=re.sub('[^a-zA-Z]',' ', news['title'][i])
        detect=detect.lower()
        detect=detect.split()
        detect=[ps.stem(word) for word in detect if not word in stopwords.words('english')]
        detect=' '.join(detect)
        corpus.append(detect)
        corpus
        cv=CountVectorizer(max_features=5000,ngram_range=(1,3))
        X=cv.fit_transform(corpus).toarray()
        X.shape
        y=news['label']
        y.head()
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        cv.get_feature_names()[:20]
        cv.get_params()
        count_df= pd.DataFrame(X_train, columns=cv.get_feature_names())
        count_df.head()
        from sklearn.naive_bayes import MultinomialNB
        MNB=MultinomialNB()
        MNB.fit(X_train, y_train)

        
    if request.method == 'POST':
        message=request.form['message']
        data=[message]
        vect=cv.transform(data).toarray()
        pred=MNB.predict(vect)
    
    return render_template('index.html',prediction_text='This news is{}'.format(pred))
        
    if __name__=='__main__':
        app.run(debug=True)    
  

