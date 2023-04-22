from flask import Flask, render_template, request,send_file
import spacy
import pandas as pd
import pickle
import tweepy
import matplotlib.pyplot as plt
from textblob import TextBlob
import string
model = pickle.load(open('model.pickle', 'rb'))
vect = pickle.load(open('vec.pickle', 'rb'))


consumer_key = "10yh2JisP7UC1yrJZc124IYqH"
consumer_secret =  "WIwZeogjvWJe0vQLfaNRQ2eosfwZgiE53rE8IREabFpOzMPMKS"
access_key ="1437038540391649284-mrXjVY9Xw92V5TsymunFkiiIzV1iXg"
access_secret = "wynX8FjopgU33joYxedQXhErvhmdQaEZMyMkqIZyrpPm4"
db = pd.DataFrame(columns=['username','text',])

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

def rem_pun(tweet):

    return ''.join(ch for ch in tweet if ch not in set(string.punctuation))

def rem_sw(tweet):

    sp=spacy.load('en_core_web_sm')
    sw=sp.Defaults.stop_words
    tweet=tweet.lower()
    tweet = ' '.join([w for w in tweet.split(' ') if w not in sw])
    return tweet


def scrape(ht, ds, numtweet):
    global db
    tweets = api.search_tweets(ht, lang="en",
                        since_id=ds,
                        tweet_mode='extended',count=numtweet)

    list_tweets = [tweet for tweet in tweets]
    for tweet in list_tweets:
        username = tweet.user.screen_name
        hashtags = tweet.entities['hashtags']

        try:
            text = tweet.retweeted_status.full_text
        except AttributeError:
            text = tweet.full_text
        hashtext = list()
        for j in range(0, len(hashtags)):
            hashtext.append(hashtags[j]['text'])

        ith_tweet = [username, text]
        db.loc[len(db)] = ith_tweet
        


    db['clean']=db['text'].apply(rem_pun)
    db['stop_word']=db['clean'].apply(rem_sw)
    # data=db['stop_word']
    count = vect.transform(db['stop_word'])
    predictions = model.predict(count)
    db['sentiment'] = predictions
    db = db.replace([0,1], ["Negative","Positive"])
    db.drop(['clean','stop_word'],axis=1, inplace=True)
    



		
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        hashtag = request.form['hashtag']
        date=request.form['date']
        tweets=request.form['tweets']
        
        
        scrape(hashtag, date, tweets)
        db.to_csv('Res.csv',index=False)
        return send_file('Res.csv',as_attachment=True)
    
    return render_template('home.html')

if __name__ == '__main__':
	app.run(debug=True)



