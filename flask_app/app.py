from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Headers to mimic browser request
headers = {
    'authority': 'www.amazon.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
    'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
}


def get_product_name(url):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return e
    
    soup = BeautifulSoup(response.text, 'html.parser')
    product_title_element = soup.select_one('span#productTitle')
    if product_title_element:
        product_name = product_title_element.get_text(strip=True)
    else:
        product_name = "Product Name Not Found"
    return product_name

def analyze_sentiment_vader(reviews):
    if not reviews:
        return 0, 0, 0

    analyzer = SentimentIntensityAnalyzer()
    positive_count = sum(1 for review in reviews if analyzer.polarity_scores(review)['compound'] > 0.05)
    negative_count = sum(1 for review in reviews if analyzer.polarity_scores(review)['compound'] < -0.05)
    neutral_count = len(reviews) - positive_count - negative_count

    positive_percentage = (positive_count / len(reviews)) * 100
    negative_percentage = (negative_count / len(reviews)) * 100
    neutral_percentage = (neutral_count / len(reviews)) * 100

    return positive_percentage, negative_percentage, neutral_percentage

# Load and preprocess data for ML model
stop_words = set(stopwords.words('english'))

def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in text_data:
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(token.lower() for token in nltk.word_tokenize(sentence) if token.lower() not in stop_words))
    return preprocessed_text

# Load dataset and train ML model
data = pd.read_csv("C:\\Users\\vanam\\Downloads\\amazonreviews.csv")
data['label'] = data['rating'].apply(lambda x: 1 if x >= 5 else 0)
data['review'] = preprocess_text(data['review'].values)

cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['review']).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.33, stratify=data['label'], random_state=42)

ml_model = DecisionTreeClassifier(random_state=0)
ml_model.fit(X_train, y_train)
  
def analyze_sentiment_ml(reviews):
    if not reviews:
        return 0, 0, 0

    reviews_preprocessed = preprocess_text(reviews)
    X_reviews = cv.transform(reviews_preprocessed).toarray()
    predictions = ml_model.predict(X_reviews)

    positive_percentage = (sum(predictions) / len(predictions)) * 100
    negative_percentage = 100 - positive_percentage
    neutral_percentage = 0  # As the ML model is binary, neutral percentage is not applicable

    return positive_percentage, negative_percentage, neutral_percentage

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return img_str

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        product_name = get_product_name(url)

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        amazon_reviews = [review.get_text(strip=True) for review in soup.select('span[data-hook="review-body"]')]

        if not amazon_reviews:
            return jsonify({
                'product_name': product_name,
                'positive_percentage': 0,
                'negative_percentage': 0,
                'neutral_percentage': 0,
                'word_cloud': ''
            })

        # Perform sentiment analysis using VADER
        vader_positive_percentage, vader_negative_percentage, vader_neutral_percentage = analyze_sentiment_vader(amazon_reviews)

        # Decide if VADER results are sufficient
        if vader_neutral_percentage > 40:  # Arbitrary threshold for refinement
            # Perform sentiment analysis using the ML model
            ml_positive_percentage, ml_negative_percentage, ml_neutral_percentage = analyze_sentiment_ml(amazon_reviews)
            positive_percentage = ml_positive_percentage
            negative_percentage = ml_negative_percentage
            neutral_percentage = ml_neutral_percentage
        else:
            positive_percentage = vader_positive_percentage
            negative_percentage = vader_negative_percentage
            neutral_percentage = vader_neutral_percentage

        # Generate word cloud
        word_cloud_text = ' '.join(amazon_reviews)
        word_cloud_image = generate_word_cloud(word_cloud_text)

        # Prepare response data with overall sentiment and percentages
        response_data = {
            'product_name': product_name,
            'positive_percentage': round(positive_percentage, 2),
            'negative_percentage': round(negative_percentage, 2),
            'neutral_percentage': round(neutral_percentage, 2),
            'word_cloud': word_cloud_image
        }

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)