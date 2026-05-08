import warnings
from pathlib import Path

import pandas as pd
from prophet import Prophet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

DATA_DIR = Path("StackOverflow Data")
PROCESSED_DIR = DATA_DIR / "processed"
CHATGPT_LAUNCH = pd.Timestamp("2022-11-01")


def add_date_and_category(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df["category"] = "post-ChatGPT"
    df.loc[df["date"] < CHATGPT_LAUNCH, "category"] = "pre-ChatGPT"
    return df


def load_raw_data():
    questions = pd.read_csv(DATA_DIR / "question_volume.csv")
    users = pd.read_csv(DATA_DIR / "user_registration.csv")
    votes = pd.read_csv(DATA_DIR / "votes_over_time.csv")
    comments = pd.read_csv(DATA_DIR / "comments.csv")

    questions = add_date_and_category(questions)
    users = add_date_and_category(users)
    votes = add_date_and_category(votes)
    comments = add_date_and_category(comments)

    votes["downvote_ratio"] = votes["downvotes"] / votes["total_votes"].replace(0, pd.NA)
    votes["upvote_ratio"] = votes["upvotes"] / votes["total_votes"].replace(0, pd.NA)

    comments["Text"] = comments["Text"].fillna("").astype(str)
    return questions, users, votes, comments


def score_comments(comments):
    comments = comments.copy()
    analyzer = SentimentIntensityAnalyzer()
    comments["sentiment_score"] = comments["Text"].map(
        lambda text: analyzer.polarity_scores(text)["compound"]
    )

    comments["sentiment_label"] = pd.cut(
        comments["sentiment_score"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"],
    ).astype(str)
    return comments


def build_monthly_sentiment(comments):
    return (
        comments.groupby("date", as_index=False)
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            std_sentiment=("sentiment_score", "std"),
            comment_count=("sentiment_score", "count"),
        )
        .sort_values("date")
    )


def prophet_forecast(df, date_col, value_col):
    prophet_df = (
        df[[date_col, value_col]]
        .rename(columns={date_col: "ds", value_col: "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )

    train = prophet_df[prophet_df["ds"] < CHATGPT_LAUNCH].copy()
    test = prophet_df[prophet_df["ds"] >= CHATGPT_LAUNCH].copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
    )
    model.fit(train)

    future = model.make_future_dataframe(periods=len(test), freq="MS")
    forecast = model.predict(future)
    results = prophet_df.merge(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="left",
    )
    results["effect"] = results["y"] - results["yhat"]
    return results


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw CSV files...")
    questions, users, votes, comments = load_raw_data()

    print("Running VADER sentiment scoring...")
    comments = score_comments(comments)

    print("Building monthly sentiment summary...")
    monthly_sentiment = build_monthly_sentiment(comments)

    print("Training Prophet model for question volume...")
    question_forecast = prophet_forecast(questions, "date", "question_count")

    print("Training Prophet model for sentiment...")
    sentiment_forecast = prophet_forecast(monthly_sentiment, "date", "avg_sentiment")

    questions.to_csv(PROCESSED_DIR / "questions.csv", index=False)
    users.to_csv(PROCESSED_DIR / "users.csv", index=False)
    votes.to_csv(PROCESSED_DIR / "votes.csv", index=False)
    comments.to_csv(PROCESSED_DIR / "comments_with_sentiment.csv", index=False)
    monthly_sentiment.to_csv(PROCESSED_DIR / "monthly_sentiment.csv", index=False)
    question_forecast.to_csv(PROCESSED_DIR / "question_forecast.csv", index=False)
    sentiment_forecast.to_csv(PROCESSED_DIR / "sentiment_forecast.csv", index=False)

    print(f"Done. Processed files saved in: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
