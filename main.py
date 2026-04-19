import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load dataset (giảm lag)
df = pd.read_csv(
    'C:/Users/ha lam/Downloads/archive/Reviews.csv',
    engine='python',
    on_bad_lines='skip',
    nrows=10000
)

print(df.head())

# ===== BIỂU ĐỒ RATING =====
ratings = df["Score"].value_counts()
numbers = ratings.index
quantity = ratings.values

plt.figure(figsize=(10, 8))
plt.pie(quantity, labels=numbers)
plt.title("Distribution of Amazon Product Ratings")
plt.show()

# ===== SENTIMENT ANALYSIS =====
sentiments = SentimentIntensityAnalyzer()

df["Positive"] = [sentiments.polarity_scores(str(i))["pos"] for i in df["Text"]]
df["Negative"] = [sentiments.polarity_scores(str(i))["neg"] for i in df["Text"]]
df["Neutral"] = [sentiments.polarity_scores(str(i))["neu"] for i in df["Text"]]

# Tổng
x = df["Positive"].mean()
y = df["Negative"].mean()
z = df["Neutral"].mean()

# Kết luận
def sentiment_score(a, b, c):
    if (a > b) and (a > c):
        print("Overall Sentiment: Positive 😊")
    elif (b > a) and (b > c):
        print("Overall Sentiment: Negative 😠")
    else:
        print("Overall Sentiment: Neutral 🙂")

sentiment_score(x, y, z)