from flask import Flask, request, jsonify
app = Flask(__name__)
#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API
access_token = "959859922618986497-kOhEI70ixsqQTayC8WnpXaoDp6Qt1xk"
access_token_secret = "FrGgupbyHKY1zc4sZIunT9I5r5T21UDZzXElc4U5az2IY"
consumer_key = "zB1qyz6wJhx1e1tXt971Q52qX"
consumer_secret = "hHlvA85Oeog7DAS2SlmlVtAliWqjldQmIsyLDmzV1YXFdiydDF"
text_file = open("C:/Users/Roy/PycharmProjects/proj/twitter_data_friday.txt", "w")
#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):
    def on_data(self, data):
        text_file.write(data)
        return True

    def on_error(self, status):
        print(status)


@app.route('/tweets/<keyword>')
def tweets(keyword):

    track = keyword
    # This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    # This line filter Twitter Streams to capture data by the keywords
    stream.filter(track=[track])
    return "Collecting tweets..."

@app.route('/stop')
def stop():
    exit(1)

# running web app in local machine
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)