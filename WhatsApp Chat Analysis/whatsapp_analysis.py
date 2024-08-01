import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import emoji
from datetime import datetime
from nltk.corpus import stopwords
from textblob import TextBlob
import re
import collections

# Ensure NLTK stopwords are downloaded
import nltk
nltk.download('stopwords')

# Function to preprocess the chat data
def preprocess(data):
    processed_data = []
    date_formats = [
        '%d/%m/%y, %I:%M %p',
        '%d/%m/%Y, %I:%M %p',
        '%d/%m/%y, %H:%M %p',
        '%d/%m/%Y, %H:%M %p'
    ]

    for line in data:
        print(f"Processing line: {line}")  # Debugging: Show current line being processed
        try:
            if ' - ' in line:
                parts = line.split(' - ', 1)
                if len(parts) != 2:
                    print(f"Skipping line (unexpected format): {line}")
                    continue

                date_str, message = parts

                date = None
                for fmt in date_formats:
                    try:
                        date = datetime.strptime(date_str.strip(), fmt)
                        break
                    except ValueError:
                        continue

                if not date:
                    print(f"Date parsing failed for line: {line}")
                    continue

                if ": " in message:
                    user, message = message.split(": ", 1)
                else:
                    user, message = "System", message

                processed_data.append([date, user, message])
            else:
                if processed_data:
                    processed_data[-1][2] += f" {line}"
                else:
                    processed_data.append([None, "Unknown", line])
        except Exception as e:
            print(f"Error processing line: {line}")
            print(f"Exception: {e}")

    chat_df = pd.DataFrame(processed_data, columns=['Date', 'User', 'Message'])
    chat_df['Date'] = pd.to_datetime(chat_df['Date'], errors='coerce')
    chat_df = chat_df.dropna(subset=['Date'])
    chat_df['Message'] = chat_df['Message'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    print(f"Processed DataFrame: {chat_df.head()}")

    return chat_df

# Function to clean text for NLP tasks
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in emoji.EMOJI_DATA and c.isalpha() or c.isspace()])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Function to extract emojis and count their occurrences
def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

def count_emojis(df):
    emoji_counts = {}
    for message in df['Message']:
        emojis = extract_emojis(message)
        for e in emojis:
            if e in emoji_counts:
                emoji_counts[e] += 1
            else:
                emoji_counts[e] = 1
    return emoji_counts

# Function to perform sentiment analysis
def analyze_sentiment(message):
    analysis = TextBlob(message)
    return analysis.sentiment.polarity

# Function to get the most positive and negative speakers
def get_sentiment_scores(df):
    df['Sentiment'] = df['Message'].apply(analyze_sentiment)
    sentiment_scores = df.groupby('User')['Sentiment'].mean()
    most_positive = sentiment_scores.idxmax() if not sentiment_scores.empty else "N/A"
    most_negative = sentiment_scores.idxmin() if not sentiment_scores.empty else "N/A"
    return sentiment_scores, most_positive, most_negative

# CSS to expand the main content width
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 90%;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Upload the WhatsApp chat file
uploaded_file = st.file_uploader("Choose a WhatsApp chat file", type="txt")

if uploaded_file is not None:
    data = uploaded_file.read().decode("utf-8").splitlines()
    chat_data = preprocess(data)
    print(chat_data)

    # Ensure Date column is in datetime format
    try:
        chat_data['Date'] = pd.to_datetime(chat_data['Date'])
    except Exception as e:
        st.error(f"Error parsing dates: {e}")

    st.title("WhatsApp Chat Analysis")

    # Calculate key metrics
    total_contributions = chat_data['User'].value_counts()
    most_talkative = total_contributions.idxmax() if not total_contributions.empty else "N/A"
    least_talkative = total_contributions.idxmin() if not total_contributions.empty else "N/A"
    most_active_day = chat_data['Date'].dt.day_name().mode()[0] if not chat_data['Date'].empty else "N/A"
    most_active_time = chat_data['Date'].dt.hour.mode()[0] if not chat_data['Date'].empty else "N/A"

    # Create a table of key insights
    st.header("Key Insights")
    insights_df = pd.DataFrame({
        "Metric": ["Most Talkative", "Least Talkative", "Most Active Day", "Most Active Time"],
        "Value": [most_talkative, least_talkative, most_active_day, most_active_time]
    })
    st.table(insights_df)

    # Sentiment Analysis
    st.header("Sentiment Analysis")
    sentiment_scores, most_positive, most_negative = get_sentiment_scores(chat_data)
    
    # One row, three columns layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Sentiment Scores by User")
        st.write(sentiment_scores)
        
    with col2:
        st.subheader("Most Positive & Negative Speakers")
        st.write(f"Most Positive Speaker: {most_positive}")
        st.write(f"Most Negative Speaker: {most_negative}")

    
    st.header("Total Contribution of Chat Participants")
    total_contributions = chat_data['User'].value_counts()
    if not total_contributions.empty:
        st.write(total_contributions)
        st.bar_chart(total_contributions)
        most_talkative = total_contributions.idxmax()
        least_talkative = total_contributions.idxmin()
        st.write(f"Most Talkative: {most_talkative}")
        st.write(f"Least Talkative: {least_talkative}")
    else:
        st.write("No data available to show chat contributions.")

    # Most Active Day & Time
    st.header("Most Active Day & Time")
    if not chat_data.empty and 'Date' in chat_data.columns:
        most_active_day = chat_data['Date'].dt.day_name().mode()[0]
        most_active_time = chat_data['Date'].dt.hour.mode()[0]
        st.write(f"Most Active Day: {most_active_day}")
        st.write(f"Most Active Time: {most_active_time}")
    else:
        st.write("Chat data is empty or Date column is not available.")
    
    # Data Visualization of Group Activity
    st.header("Data Visualization of Group Activity")

    # Set up a figure with 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Most Busy Days
    st.write("Most Busy Days")
    if not chat_data.empty and 'Date' in chat_data.columns:
        busy_days = chat_data['Date'].dt.date.value_counts().head(10)
        if not busy_days.empty:
            axs[0].pie(busy_days, labels=busy_days.index, autopct='%1.1f%%', startangle=140)
            axs[0].set_title('Most Busy Days')
            
        else:
            st.write("No data available for most busy days.")
    else:
        st.write("Chat data is empty or Date column is not available.")

    # Most Busy Months
    st.write("Most Busy Months")
    if not chat_data.empty and 'Date' in chat_data.columns:
        busy_months = chat_data['Date'].dt.to_period('M').value_counts().sort_index()
        if not busy_months.empty:
            axs[1].pie(busy_months, labels=busy_months.index.astype(str), autopct='%1.1f%%', startangle=140)
            axs[1].set_title('Most Busy Months')
           
        else:
            st.write("No data available for most busy months.")
    else:
        st.write("Chat data is empty or Date column is not available.")

    # Most Busy Years
    st.write("Most Busy Years")
    if not chat_data.empty and 'Date' in chat_data.columns:
        busy_years = chat_data['Date'].dt.year.value_counts().sort_index()
        if not busy_years.empty:
            axs[2].pie(busy_years, labels=busy_years.index, autopct='%1.1f%%', startangle=140)
            axs[2].set_title('Most Busy Years')
           
        else:
            st.write("No data available for most busy years.")
    else:
        st.write("Chat data is empty or Date column is not available.")

    st.pyplot(fig)

    # Media Count
    st.header("Media Count Sent by Each Person")
    media_msgs = chat_data[chat_data['Message'].str.contains('<Media omitted>', na=False)]
    media_counts = media_msgs['User'].value_counts()
    if not media_counts.empty:
        st.write(media_counts)
    else:
        st.write("No media messages found.")

    # Missed Calls
    st.header("Missed Calls")
    missed_calls = chat_data[chat_data['Message'].str.contains('missed a call', na=False)]
    missed_call_counts = missed_calls['User'].value_counts()
    if not missed_call_counts.empty:
        st.write(missed_call_counts)
    else:
        st.write("No missed calls found.")

    # Hourly Contribution of Chat Participants
    st.header("Hourly Contribution of Chat Participants")
    if not chat_data.empty and 'Date' in chat_data.columns:
        chat_data['Hour'] = chat_data['Date'].dt.hour
        hourly_contributions = chat_data.groupby('Hour')['Message'].count()
        if not hourly_contributions.empty:
            st.line_chart(hourly_contributions)
        else:
            st.write("No hourly data available.")
    else:
        st.write("Chat data is empty or Date column is not available.")

    # Activity of Chat Participants on Every Minute of the Day
    st.header("Activity of Chat Participants on Every Minute of the Day")
    if not chat_data.empty and 'Date' in chat_data.columns:
        chat_data['Minute'] = chat_data['Date'].dt.minute
        minute_activity = chat_data.groupby(['Hour', 'Minute'])['Message'].count().unstack()
        if not minute_activity.empty:
            fig, ax = plt.subplots()
            sns.heatmap(minute_activity.fillna(0), ax=ax)
            st.pyplot(fig)
        else:
            st.write("No minute-level data available.")
    else:
        st.write("Chat data is empty or Date column is not available.")

    # Chat Activity HeatMap
     # Chat Activity HeatMap
    st.header("Chat Activity HeatMap")
    if not chat_data.empty and 'Date' in chat_data.columns:
        chat_data['Day'] = chat_data['Date'].dt.day_name()
        chat_data['Hour'] = chat_data['Date'].dt.hour
        heatmap_data = chat_data.groupby(['Day', 'Hour'])['Message'].count().unstack().fillna(0)
        
        fig, ax = plt.subplots()
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.0f', ax=ax)
        st.pyplot(fig)
        
        # Summary of HeatMap
        most_active_day = heatmap_data.sum(axis=1).idxmax()
        most_active_hour = heatmap_data.sum(axis=0).idxmax()
        least_active_day = heatmap_data.sum(axis=1).idxmin()
        least_active_hour = heatmap_data.sum(axis=0).idxmin()

        st.write(f"**Most Active Day:** {most_active_day}")
        st.write(f"**Most Active Hour:** {most_active_hour}")
        st.write(f"**Least Active Day:** {least_active_day}")
        st.write(f"**Least Active Hour:** {least_active_hour}")
    else:
        st.write("Chat data is empty or Date column is not available.")

    # Word Cloud
    st.header("Word Cloud")
    if not chat_data.empty and 'Message' in chat_data.columns:
        text = ' '.join(chat_data['Message'])
        cleaned_text = clean_text(text)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
        st.image(wordcloud.to_image())
    else:
        st.write("No messages available for word cloud.")

 # Data Visualization of States
   
