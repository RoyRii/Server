from flask import Flask, request, jsonify
app = Flask(__name__)

import json
import nltk
import pandas as pd
from matplotlib.patches import Rectangle
from textblob import TextBlob
import sys
import re
import html.parser
import itertools
from textblob import TextBlob
from textblob import Word
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# root
@app.route("/")
def index():
    import json
    d = json.dumps("This is root")
    return d

# Pre-process data
# GET
@app.route("/process")
def process():
    common_data_names = open("C:/Users/Roy/PycharmProjects/proj/common_data_names_friday.txt", "w")
    common_data_count = open("C:/Users/Roy/PycharmProjects/proj/common_data_count_friday.txt", "w")
    occur_data_names = open("C:/Users/Roy/PycharmProjects/proj/occur_data_names_friday.txt", "w")
    occur_data_count = open("C:/Users/Roy/PycharmProjects/proj/occur_data_count_friday.txt", "w")
    common_data_sentiment = open("C:/Users/Roy/PycharmProjects/proj/common_data_sentiment_friday.txt", "w")
    occur_sentiment = open("C:/Users/Roy/PycharmProjects/proj/occur_data_sentiment_friday.txt", "w")
    # cleaned_common_data = open("C:/Users/Roy/PycharmProjects/proj/cleaned_common_data.txt", "w")
    # cleaned_data_path = open("C:/Users/Roy/PycharmProjects/proj/cleaned_data.txt", "w")
    tweets_data_path = 'C:/Users/Roy/PycharmProjects/proj/twitter_data_friday.txt'
    tweets_data = []

    tweets_file = open(tweets_data_path, "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except:
            continue

    # number of tweets
    print(len(tweets_data))


    # structure the data into DataFrame
    tweets = pd.DataFrame()


    # adding columns to Data
    tweets['text'] = list([tweet.get('text', '')
                           for tweet in tweets_data])

    # simple NLP technique, shows the polarity of all tweets
    non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
    from collections import defaultdict
    import operator
    matrix = defaultdict(lambda: defaultdict(int))
    terms_only = []
    from collections import Counter
    count_all = Counter()
    cleaned_tweets_data = []
    for data in tweets['text']:
        data.translate(non_bmp_map)  # print tweet's text
        # 0. clean from URL
        data = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?\S', '', data,
                      flags=re.MULTILINE)
        data = re.sub(r'(http?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?\S', '', data, flags=re.MULTILINE)
        # print('0.  ', data)
        # 1. escaping HTML characters
        html_parser = html.parser.HTMLParser()
        tweet = html_parser.unescape(data)

        # # 3. Apostrophe Lookup
        # APPOSTOPHES = {"n't" : 'not'} ## Need a huge dictionary
        # words = tweet.split()
        # print(words)
        # reformed = [APPOSTOPHES[word] if word in APPOSTOPHES else word for word in words]
        # reformed = " ".join(reformed)
        # print('3.  ', reformed)
        # 4. Split Attached Words
        # cleaned = " ".join(re.findall('[A-Z][^A-Z]*', tweet))
        # print('4.  ', cleaned)

        # 6. standardizing words (happpppy = happy)
        tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))
        # print('5.  ', tweet)
        tblob = TextBlob(tweet)
        # convert to lower case
        tblob = tblob.lower()
        # print('lower : ', tblob)
        # spelling correction
        # tblob = tblob.correct()
        # print("Corrected : ", tblob)
        string = ''
        new_data = TextBlob(string)
        for word in tblob.words:
            w = Word(word)
            w = w.lemmatize("v")
            new_data = new_data + ' ' + w;
        # print('new : ', new_data)
        # new_data = new_data.words.singularize()  # [Words List] created
        # print("singular : ", new_data)
        data = '\t' + str(new_data)
        # split into words
        tokens = word_tokenize(data)
        # print("-----", tokens)
        # # convert to lower case
        # tokens = [w.lower() for w in tokens]
        # remove punctuation from each word
        import string
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        # remove remaining tokens that are not alphabetic
        # words = [word for word in stripped if word.isalpha()]
        words = stripped

        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        # print(words)
        # stemming of words
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in words]
        # print(stemmed)
        filtered = [f for f in stemmed if f != 'nt' and f != 'wo' and len(f) != 1 and len(f) != 2 and not f.startswith('/')]
        # print(filtered)
        count_all.update(filtered)
        cleaned_tweets_data.append(' '.join(filtered))
        # Build co-occurrence matrix
        for i in range(len(filtered) - 1):
            for j in range(i + 1, len(filtered)):
                w1, w2 = sorted([filtered[i], filtered[j]])
                if w1 != w2:
                    matrix[w1][w2] += 1


    print("stage 1")
    com_max = []
    # For each term, look for the most common co-occurrent terms
    for t1 in matrix:
        t1_max_terms = sorted(matrix[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
        for t2, t2_count in t1_max_terms:
            com_max.append(((t1, t2), t2_count))

    # Get the most frequent co-occurrences
    terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
    # print('Co-occurent terms: ', terms_max[:10])
    occur_trends = []
    nums = []
    for a, b in terms_max[:20]:
        occur_trends.append(a)
        nums.append(b)
    # print(occur_trends)
    # print(nums)
    # print('----------\n')
    print('Most common: ', count_all.most_common(20))
    # print('-------\n')
    single_trends = []
    trends_num = []
    for a, b in count_all.most_common(20):
        single_trends.append(a)
        trends_num.append(b)
    # print(single_trends)
    # print(trends_num)
    # print("------")

    str1 = str(occur_trends)
    str1 = str1.encode('ascii', 'ignore')
    occur_data_names.write(str(str1))
    str2 = str(nums)
    str2 = str2.encode('ascii', 'ignore')
    occur_data_count.write(str(str2))
    str3 = str(single_trends)
    str3 = str3.encode('ascii', 'ignore')
    common_data_names.write(str(str3))
    str4 = str(trends_num)
    str4 = str4.encode('ascii', 'ignore')
    common_data_count.write(str(str4))

    # single trends sentiment
    single_trends_sentiment = []
    for trend in single_trends:
        # Mining the tweets
        trendy_tweets = []

        # checks if the word in a text
        def word_in_text(word, text):
            word1 = word.lower()
            text1 = text.lower()
            match = re.search(word1, text1)
            if match:
                trendy_tweets.append(text)
                return True
            return False

        # add columns to tweets DataFrame
        tweets['keyword'] = tweets['text'].apply(lambda tweet: word_in_text(trend, tweet), '')
        # print(trendy_tweets)

        positive = []
        negative = []
        neutral = []
        # # simple NLP technique, shows the polarity of all tweets
        # non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
        if len(trendy_tweets) != 0:
            for single_tweet in trendy_tweets:
                # print(single_tweet.translate(non_bmp_map))  # print tweet's text
                analysis = TextBlob(single_tweet)
                # print(analysis.sentiment[0])  # print tweet's polarity
                if analysis.sentiment[0] > 0.0:
                    positive.append(analysis.sentiment[0])
                elif analysis.sentiment[0] < 0.0:
                    negative.append(analysis.sentiment[0])
                else:
                    neutral.append(analysis.sentiment[0])

            pos = (len(positive) / len(trendy_tweets)) * 100
            neu = (len(neutral) / len(trendy_tweets)) * 100
            neg = (len(negative) / len(trendy_tweets)) * 100
            # print(pos, '%')
            # print(neu, '%')
            # print(neg, '%')
            single_trends_sentiment.append(pos)
            single_trends_sentiment.append(neu)
            single_trends_sentiment.append(neg)
        else:
            single_trends_sentiment.append('ng')
            single_trends_sentiment.append('ng')
            single_trends_sentiment.append('ng')

    str_sent = str(single_trends_sentiment)
    str_sent = str_sent.encode('ascii', 'ignore')
    common_data_sentiment.write(str(str_sent))

    # occurent trends sentiment
    occur_trends_sentiment = []
    s = str(occur_trends)
    s = s[2:]
    blob = TextBlob(s)
    data1 = blob.words
    print(data1)
    for trend in data1:
        # Mining the tweets
        trendy_tweets = []

        # checks if the word in a text
        def word_in_text(word, text):
            word1 = word.lower()
            text1 = text.lower()
            match = re.search(word1, text1)
            if match:
                trendy_tweets.append(text)
                return True
            return False

        # add columns to tweets DataFrame
        tweets['keyword'] = tweets['text'].apply(lambda tweet: word_in_text(trend[1:], tweet), '')

        positive = []
        negative = []
        neutral = []
        # # simple NLP technique, shows the polarity of all tweets
        # non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
        if len(trendy_tweets) != 0:
            for single_tweet in trendy_tweets:
                # print(single_tweet.translate(non_bmp_map))  # print tweet's text
                analysis = TextBlob(single_tweet)
                # print(analysis.sentiment[0])  # print tweet's polarity
                if analysis.sentiment[0] > 0.0:
                    positive.append(analysis.sentiment[0])
                elif analysis.sentiment[0] < 0.0:
                    negative.append(analysis.sentiment[0])
                else:
                    neutral.append(analysis.sentiment[0])

            pos = (len(positive) / len(trendy_tweets)) * 100
            neu = (len(neutral) / len(trendy_tweets)) * 100
            neg = (len(negative) / len(trendy_tweets)) * 100
            # print(pos, '%')
            # print(neu, '%')
            # print(neg, '%')
            occur_trends_sentiment.append(pos)
            occur_trends_sentiment.append(neu)
            occur_trends_sentiment.append(neg)
        else:
            occur_trends_sentiment.append('ng')
            occur_trends_sentiment.append('ng')
            occur_trends_sentiment.append('ng')

    str_sent = str(occur_trends_sentiment)
    str_sent = str_sent.encode('ascii', 'ignore')
    occur_sentiment.write(str(str_sent))

    return "done"



# GET
@app.route("/trends")
def hello_user():
    """
    this serves as a demo purpose
    :param user:
    :return: str
    """
    # return "Hello %s!" % user

    from textblob import TextBlob
    path1 = 'C:/Users/Roy/PycharmProjects/proj/common_data_names_friday.txt'
    path2 = 'C:/Users/Roy/PycharmProjects/proj/common_data_count_friday.txt'
    path3 = 'C:/Users/Roy/PycharmProjects/proj/occur_data_names_friday.txt'
    path4 = 'C:/Users/Roy/PycharmProjects/proj/occur_data_count_friday.txt'
    import json
    tweets_data1 = []
    tweets_data2 = []
    tweets_data3 = []
    tweets_data4 = []

    tweets1 = open(path1, "r")
    for line in tweets1:
        try:
            tweets_data1.append(line)
        except:
            continue

    tweets2 = open(path2, "r")
    for line in tweets2:
        try:
            tweets_data2.append(line)
        except:
            continue

    tweets3 = open(path3, "r")
    for line in tweets3:
        try:
            tweets_data3.append(line)
        except:
            continue

    tweets4 = open(path4, "r")
    for line in tweets4:
        try:
            tweets_data4.append(line)
        except:
            continue

    str1 = tweets_data1[0]
    str1 = str1[2:]
    blob = TextBlob(str1)
    data1 = blob.words
    arr1 = []
    for word in data1:
        w = word[1:]
        arr1.append(w)

    str2 = tweets_data2[0]
    str2 = str2[2:]
    blob = TextBlob(str2)
    data2 = blob.words
    arr2 = []
    for word in data2:
        w = word
        arr2.append(w)

    str3 = tweets_data3[0]
    str3 = str3[2:]
    blob = TextBlob(str3)
    data3 = blob.words
    arr3 = []
    for word in data3:
        w = word[1:]
        arr3.append(w)

    str4 = tweets_data4[0]
    str4 = str4[2:]
    blob = TextBlob(str4)
    data4 = blob.words
    arr4 = []
    for word in data4:
        w = word
        arr4.append(w)

    row_json = json.dumps({'response': [{'res': arr1}, {'res': arr2}, {'res': arr3}, {'res': arr4}]})
    return row_json

# GET
@app.route("/info")
def info():
    path = 'C:/Users/Roy/PycharmProjects/proj/twitter_data_friday.txt'
    import json
    tweets_data = []
    tweets_file = open(path, "r")
    for line in tweets_file:
        try:
            tweets_data.append(line)
        except:
            continue
    info = len(tweets_data)
    info_json = json.dumps(info)
    return info_json

# GET
@app.route("/retweets")
def retweets():
    retweets = []
    urls = []
    usernames = []
    numbers = []
    path = 'C:/Users/Roy/PycharmProjects/proj/twitter_data_friday.txt'
    import json
    tweets_data = {}
    tweets_file = open(path, "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)
        except:
            continue
        if 'retweeted_status' not in tweet:
            rt = tweet['retweet_count']
            if rt < 25:
                continue
            tweets_data[rt['id_str']] = rt
        else:
            rt = tweet['retweeted_status']
            # rt1 = tweet['favorite_count']
            if rt['retweet_count'] < 25:
                continue
            tweets_data[rt['id_str']] = rt
    # convert to list
    tweets = [tweets_data[w] for w in tweets_data.keys()]
    # sort by retweet count
    tweets.sort(key=lambda x: -x['retweet_count'])
    # display top k retweets
    for t in tweets[:10]:
        s1 = str('@' + t['user']['screen_name'])
        usernames.append(s1)
        s2 = str(t['retweet_count'])
        numbers.append(s2)
        s = str(t['text'])
        a = TextBlob(t['text'])
        a1 = 'https://' + str(a.words[-1])
        urls.append(a1)
        print(s)
        retweets.append(s)
    retweets = json.dumps({'res': retweets, 'urls': urls, 'numbers': numbers, 'users': usernames}, ensure_ascii=False).encode('utf8')
    return retweets


# GET
@app.route("/likes")
def likes():
    retweets = []
    urls = []
    usernames = []
    numbers = []
    path = 'C:/Users/Roy/PycharmProjects/proj/twitter_data_friday.txt'
    import json
    tweets_data = {}
    tweets_file = open(path, "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)
        except:
            continue
        if 'retweeted_status' not in tweet:
            rt = tweet['favorite_count']
            if rt < 100:
                continue
            tweets_data[rt['id_str']] = rt
        else:
            rt = tweet['retweeted_status']
            # rt1 = tweet['favorite_count']
            if rt['favorite_count'] < 100:
                continue
            tweets_data[rt['id_str']] = rt
    # convert to list
    tweets = [tweets_data[w] for w in tweets_data.keys()]
    # sort by retweet count
    tweets.sort(key=lambda x: -x['favorite_count'])
    # display top k retweets
    for t in tweets[:10]:
        s1 = str('@' + t['user']['screen_name'])
        usernames.append(s1)
        s2 = str(t['favorite_count'])
        numbers.append(s2)
        s = str(t['text'])
        a = TextBlob(t['text'])
        a1 = 'https://' + str(a.words[-1])
        urls.append(a1)
        print(s)
        retweets.append(s)
    retweets = json.dumps({'res': retweets, 'urls': urls, 'numbers': numbers, 'users': usernames},
                          ensure_ascii=False).encode('utf8')
    return retweets


# GET
@app.route('/sentiment/<trend_type>/<position>')
def sentiment(trend_type, position):
    path1 = 'C:/Users/Roy/PycharmProjects/proj/common_data_sentiment_friday.txt'
    path2 = 'C:/Users/Roy/PycharmProjects/proj/occur_data_sentiment_friday.txt'
    import json
    sent_data1 = []
    if trend_type == 'single':
        sentiment_1 = open(path1, "r")
    elif trend_type == 'occur':
        sentiment_1 = open(path2, "r")

    for line in sentiment_1:
        try:
            sent_data1.append(line)
        except:
            continue

    str1 = sent_data1[0]
    str1 = str1[2:]
    blob = TextBlob(str1)
    data1 = blob.words
    arr1 = []
    for word in data1:
        w = word
        arr1.append(str(w))

    if trend_type == 'single':
        json_data = json.dumps({"res": arr1[int(position)*3 : int(position)*3 + 3]})
    elif trend_type == 'occur':
        json_data = json.dumps({"res": arr1[int(position) * 6: int(position) * 6 + 6]})

    return json_data


# running web app in local machine
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
