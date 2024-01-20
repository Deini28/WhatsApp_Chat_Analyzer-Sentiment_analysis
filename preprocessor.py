import re
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob  # Import TextBlob for sentiment analysis
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(data):
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[apAP][mM])'
    messages = re.split(pattern, data)[1:]
    messages = [message.replace('\u202f', ' ') for message in messages]
    dates = re.findall(pattern, data)
    dates = [date.replace('\u202f', ' ') for date in dates]

    # Ensure that the lists have the same length
    min_length = min(len(messages), len(dates))
    messages = messages[:min_length]
    dates = dates[:min_length]
    # Create DataFrame
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Convert 'message_date' type
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %I:%M %p')

    # Rename the column
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Display the DataFrame
    df.head()
    # Extract user names and messages
    users = []
    messages = []

    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    # Add user and message columns to the DataFrame
    df['user'] = users
    df['message'] = messages

    # Drop the 'user_message' column
    df.drop(columns=['user_message'], inplace=True)

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['month_num'] = df['date'].dt.month
    df['only_date'] = df['date'].dt.date
    df['day_name'] = df['date'].dt.day_name()
    df['period'] = df['date'].dt.hour // 6  # Assuming you want to divide the day into 4 periods

    # Sentiment Analysis
    df[['positive', 'negative', 'neutral']] = df['message'].apply(lambda x: pd.Series(TextBlob(x).sentiment.polarity, index=['positive', 'negative', 'neutral']))


    return df