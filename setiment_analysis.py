from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt




# Load IMDb dataset
data = pd.read_csv("D:\\IMDB Dataset.csv")


# Display a few rows
print(data.head())




# Analyze sentiment polarity using TextBlob
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply sentiment analysis to reviews
data['Sentiment_Score'] = data['review'].apply(analyze_sentiment)

# Categorize into Positive, Negative, and Neutral (consistent thresholds)
data['Sentiment_Label'] = data['Sentiment_Score'].apply(
    lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
)

# Print first 10 rows with Sentiment Scores and Labels
print(data[['review', 'Sentiment_Score', 'Sentiment_Label']].head(10))


# Plot the sentiment distribution
sentiment_counts = data['Sentiment_Label'].value_counts()
colors = ['green', 'red', 'blue']

plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


# Save processed data to a CSV file
data.to_csv("imdb_sentiment_results.csv", index=False)
print("Processed data saved to 'imdb_sentiment_results.csv'")



