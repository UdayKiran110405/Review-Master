# Review-Master: Amazon Review Sentiment Analysis Chrome Extension

## Description
Review-Master is an innovative Chrome extension that provides comprehensive sentiment analysis of Amazon product reviews. The tool helps consumers and businesses quickly understand customer opinions by:

- Automatically scraping and analyzing product reviews
- Classifying sentiments as Positive, Neutral, or Negative
- Tracking sentiment trends over time
- Highlighting key aspects like quality and value
- Generating intuitive visualizations including pie charts, trend graphs, and word clouds

Built with a robust technology stack combining NLP (VADER and machine learning models) with Flask backend and Chrome Extension frontend, this solution offers accurate, real-time insights into customer feedback.

![Extension Demo](flask_app/templates/Screenshot%202025-05-23%20014513.png)

## Features
- Real-time Amazon review scraping
- Sentiment classification (Positive/Neutral/Negative)
- Aspect-based sentiment analysis
- Interactive data visualizations
- Multiple ML model comparison
- Exportable reports

## Technology Stack
### UI Layer
- HTML5, CSS3, JavaScript
- Chart.js, jsPDF

### Backend Layer
- Flask (Python 3.8+)
- BeautifulSoup, Selenium

### NLP Layer
- VADER Sentiment Analyzer
- Scikit-learn (Logistic Regression, Random Forest, etc.)
- NLTK/spaCy, TF-IDF

## Architecture Overview
![Technology Stack](flask_app/templates/Screenshot%202025-05-23%20015910.png)
![Class Diagram](flask_app/templates/Screenshot%202025-05-23%20192215.png)

## Installation
1. Clone this repository: `git clone https://github.com/UdayKiran110405/Review-Master.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Load the extension in Chrome:
   - Go to `chrome://extensions/`
   - Enable Developer mode
   - Click "Load unpacked" and select the extension folder

## Usage
1. Navigate to any Amazon product page
2. Click the Review-Master extension icon
3. View comprehensive sentiment analysis including:

### Sentiment Distribution
![Pie Chart](flask_app/templates/Screenshot%202025-05-23%20190255.png)

### Monthly Sentiment Trends
![Trend Chart](flask_app/templates/Screenshot%202025-05-23%20190309.png)

### Key Review Aspects
![Aspect Analysis](flask_app/templates/Screenshot%202025-05-23%20190423.png)

### Word Frequency Insights
![Word Frequency](flask_app/templates/Screenshot%202025-05-23%20190349.png)

### Visual Word Cloud
![Word Cloud](flask_app/templates/Screenshot%202025-05-23%20190404.png)

### Model Performance
![Model Comparison](flask_app/templates/Screenshot%202025-05-23%20190046.png)

## Configuration Modules
- **URLScrapingModule**: Handles review scraping
- **VADERSentimentAnalyzer**: Rule-based sentiment analysis
- **MLSentimentAnalyzer**: Machine learning classification
- **DataVisualizationModule**: Generates all charts and visualizations
- **AssetAnalysisModule**: Performs aspect-based analysis

## Contributing
We welcome contributions! Please:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or support, please open an issue on GitHub or contact the maintainers.
