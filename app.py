from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

CHATGPT_LAUNCH = pd.Timestamp("2022-11-01")
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"


st.set_page_config(
    page_title="Impact of ChatGPT on Stack Overflow",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_csv(path, date_columns=None):
    date_columns = date_columns or []
    return pd.read_csv(path, parse_dates=date_columns)


@st.cache_data(show_spinner=False)
def load_processed_data():
    required_files = {
        "questions": PROCESSED_DIR / "questions.csv",
        "users": PROCESSED_DIR / "users.csv",
        "votes": PROCESSED_DIR / "votes.csv",
        "comments": PROCESSED_DIR / "comments_with_sentiment.csv",
        "monthly_sentiment": PROCESSED_DIR / "monthly_sentiment.csv",
        "question_forecast": PROCESSED_DIR / "question_forecast.csv",
        "sentiment_forecast": PROCESSED_DIR / "sentiment_forecast.csv",
    }

    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        return None, missing

    data = {
        "questions": read_csv(required_files["questions"], ["date"]),
        "users": read_csv(required_files["users"], ["date"]),
        "votes": read_csv(required_files["votes"], ["date"]),
        "comments": read_csv(required_files["comments"], ["date"]),
        "monthly_sentiment": read_csv(required_files["monthly_sentiment"], ["date"]),
        "question_forecast": read_csv(required_files["question_forecast"], ["ds"]),
        "sentiment_forecast": read_csv(required_files["sentiment_forecast"], ["ds"]),
    }
    return data, []


def pct_change(pre, post):
    if pre == 0:
        return 0
    return (post - pre) / pre * 100


def rolling(series, window):
    return series.rolling(window=window, min_periods=1).mean()


def style_axis(ax):
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18)


def show_missing_data_message(missing):
    st.title("Impact of ChatGPT on Stack Overflow")
    st.error("Processed data files were not found yet.")
    st.write(
        "Run this command once before starting the app. It will create fast-loading "
        "processed CSV files with sentiment scores and Prophet forecasts."
    )
    st.code("python prepare_data.py", language="bash")
    st.write("Missing files:")
    st.code("\n".join(missing), language="text")


data, missing_files = load_processed_data()
if missing_files:
    show_missing_data_message(missing_files)
    st.stop()

questions = data["questions"]
users = data["users"]
votes = data["votes"]
comments = data["comments"]
monthly_sentiment = data["monthly_sentiment"]
results_questions = data["question_forecast"]
results_sentiment = data["sentiment_forecast"]


with st.sidebar:
    st.title("Navigation")
    section = st.radio(
        "Go to",
        options=[
            "Overview",
            "Activity Trends",
            "Sentiment Analysis",
            "Forecast",
            "Conclusion",
        ],
    )

    st.divider()
    st.title("Controls")
    window = st.slider(
        "Smoothing window (months)",
        min_value=1,
        max_value=12,
        value=3,
        help="Higher = smoother line. Lower = more detail.",
    )

    st.divider()
    st.caption("Python · pandas · matplotlib · plotly · VADER · Prophet")


st.title("Impact of ChatGPT on Stack Overflow")
st.markdown("*How ChatGPT reshaped a tech community: an analysis of user activity and sentiment*")
st.divider()


pre_ques = questions.loc[questions["category"] == "pre-ChatGPT", "question_count"].mean()
post_ques = questions.loc[questions["category"] == "post-ChatGPT", "question_count"].mean()
pre_user = users.loc[users["category"] == "pre-ChatGPT", "new_users"].mean()
post_user = users.loc[users["category"] == "post-ChatGPT", "new_users"].mean()
pct_positive = (comments["sentiment_label"] == "positive").mean() * 100
avg_sentiment = comments["sentiment_score"].mean()


if section == "Overview":
    st.header("Project Overview")
    st.write(
        """
        This dashboard explores how activity on Stack Overflow has changed after
        the launch of ChatGPT in November 2022. It focuses on four key areas:
        **question volume**, **new user registrations**, **community voting**,
        and **comment sentiment**.

        The analysis is based on data from 2019 to April 2026.

        Data source: [Stack Exchange Data Explorer](https://data.stackexchange.com/stackoverflow/queries)
        """
    )

    st.divider()
    st.subheader("Key Numbers at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Monthly Questions", f"{post_ques:,.0f}", f"{post_ques - pre_ques:+,.0f} vs pre-ChatGPT")
    col2.metric("Avg Monthly New Users", f"{post_user:,.0f}", f"{post_user - pre_user:+,.0f} vs pre-ChatGPT")
    col3.metric("Positive Comments", f"{pct_positive:.1f}%", "of all 50k comments")
    col4.metric("Avg Sentiment Score", f"{avg_sentiment:+.3f}", "VADER compound score")

    st.divider()
    st.subheader("Major Highlights")
    col_a, col_b, col_c = st.columns(3)
    col_a.success("New user registrations **increased** after ChatGPT launch")
    col_b.info("Community sentiment remains **stable** overall")
    col_c.warning("Question volume **declining** below forecast")

    st.info(
        """
        Stack Overflow appears to be used differently following the launch of
        ChatGPT. This may indicate a shift away from simple, common questions
        rather than a reduction in engagement.
        """
    )


elif section == "Activity Trends":
    st.header("Activity Trends Over Time")
    st.write(
        "Use the **smoothing window** slider in the sidebar to adjust how much "
        "the lines are smoothed. A smaller window makes fluctuations visible; "
        "a larger window highlights long-term trends."
    )

    st.subheader("Question Volume Over Time")
    question_rolling = rolling(questions["question_count"], window)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(questions["date"], questions["question_count"], color="steelblue", alpha=0.4, width=20, label="Monthly Questions")
    ax.plot(questions["date"], question_rolling, color="steelblue", linewidth=2.5, label=f"{window}-Month Rolling Average")
    ax.axvline(CHATGPT_LAUNCH, color="red", linestyle="--", linewidth=1.5, label="ChatGPT Launch (Nov 2022)")
    ax.set_title("Stack Overflow Question Volume Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Questions")
    ax.legend(fontsize=10)
    style_axis(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write("Question volume shows a decreasing trend, with a sharper decline observed after the ChatGPT launch.")

    st.subheader("New Users and Voting Behaviour")
    col1, col2 = st.columns(2)

    user_rolling = rolling(users["new_users"], window)
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(users["date"], users["new_users"], color="green", alpha=0.4, width=20, label="Monthly New Users")
    ax1.plot(users["date"], user_rolling, color="green", linewidth=2.5, label=f"{window}-Month Rolling Avg")
    ax1.axvline(CHATGPT_LAUNCH, color="red", linestyle="--", linewidth=1.5, label="ChatGPT Launch")
    ax1.set_title("New User Registrations", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Number of New Users")
    ax1.legend(fontsize=9)
    style_axis(ax1)
    plt.tight_layout()
    col1.pyplot(fig1)
    plt.close(fig1)
    col1.write("New registrations surged after 2024, indicating that the community is growing with newcomers.")

    upvotes_rolling = rolling(votes["upvotes"], window)
    downvotes_rolling = rolling(votes["downvotes"], window)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(votes["date"], upvotes_rolling, color="green", linewidth=2.5, label="Upvotes")
    ax2.plot(votes["date"], downvotes_rolling, color="red", linewidth=2.5, label="Downvotes")
    ax2.axvline(CHATGPT_LAUNCH, color="black", linestyle="--", linewidth=1.5, label="ChatGPT Launch")
    ax2.set_title("Community Votes Over Time", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Number of Votes")
    ax2.legend(fontsize=9)
    style_axis(ax2)
    plt.tight_layout()
    col2.pyplot(fig2)
    plt.close(fig2)
    col2.write("Overall votes have reduced with a significant drop in upvotes, suggesting decreasing contribution and engagement.")

    st.divider()
    st.subheader("Pre vs Post ChatGPT — Activity Summary")
    pre_dv = votes.loc[votes["category"] == "pre-ChatGPT", "downvote_ratio"].mean()
    post_dv = votes.loc[votes["category"] == "post-ChatGPT", "downvote_ratio"].mean()
    pre_uv = votes.loc[votes["category"] == "pre-ChatGPT", "upvote_ratio"].mean()
    post_uv = votes.loc[votes["category"] == "post-ChatGPT", "upvote_ratio"].mean()

    summary_df = pd.DataFrame(
        {
            "Metric": ["Avg Monthly Questions", "Avg Monthly New Users", "Avg Downvote Ratio", "Avg Upvote Ratio"],
            "Pre-ChatGPT": [f"{pre_ques:,.0f}", f"{pre_user:,.0f}", f"{pre_dv:.4f}", f"{pre_uv:.4f}"],
            "Post-ChatGPT": [f"{post_ques:,.0f}", f"{post_user:,.0f}", f"{post_dv:.4f}", f"{post_uv:.4f}"],
            "Change": [
                f"{pct_change(pre_ques, post_ques):+.1f}%",
                f"{pct_change(pre_user, post_user):+.1f}%",
                f"{pct_change(pre_dv, post_dv):+.1f}%",
                f"{pct_change(pre_uv, post_uv):+.1f}%",
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


elif section == "Sentiment Analysis":
    st.header("Sentiment Analysis Using VADER")
    st.write(
        """
        50,000 Stack Overflow comments were analyzed and scored using VADER.
        Each comment was assigned a compound score from -1.0 to +1.0, then
        labeled as positive, neutral, or negative.
        """
    )

    st.divider()
    st.subheader("Distribution of Comment Sentiment Labels")
    label_counts = comments["sentiment_label"].value_counts().reset_index()
    label_counts.columns = ["sentiment_label", "count"]
    label_counts["percent"] = label_counts["count"] / label_counts["count"].sum()

    fig = px.bar(
        label_counts,
        x="sentiment_label",
        y="percent",
        color="sentiment_label",
        color_discrete_map={"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"},
        text=label_counts["percent"].map(lambda value: f"{value:.1%}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Sentiment Label", yaxis_title="Percentage", yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Half of all comments having positive sentiment indicates that the community tone remains stable.")

    st.divider()
    st.subheader("Comment Sentiment Over Time")
    sentiment_rolling = rolling(monthly_sentiment["avg_sentiment"], window)

    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(monthly_sentiment["date"], monthly_sentiment["avg_sentiment"], alpha=0.2, color="steelblue", label="Monthly average")
    ax.plot(monthly_sentiment["date"], sentiment_rolling, color="steelblue", linewidth=2.5, label=f"{window}-month rolling average")
    ax.axvline(CHATGPT_LAUNCH, color="red", linestyle="--", linewidth=1.5, label="ChatGPT launch (Nov 2022)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_ylim(-0.05, 0.25)
    ax.set_title("Stack Overflow Comment Sentiment Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg VADER Compound Score")
    ax.legend(fontsize=11)
    style_axis(ax)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.write("Sentiment stays largely positive over time, with minor fluctuations and no noticeable shift after the ChatGPT launch.")

    st.divider()
    st.subheader("Pre vs Post ChatGPT — Sentiment Label Mix")
    label_mix = (
        comments.groupby(["category", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .apply(lambda row: row / row.sum() * 100, axis=1)
        .round(1)
    )
    label_mix = label_mix.reindex(["pre-ChatGPT", "post-ChatGPT"])
    label_mix_long = label_mix.reset_index().melt(id_vars="category", var_name="sentiment_label", value_name="percentage")

    fig3 = px.bar(
        label_mix_long,
        x="sentiment_label",
        y="percentage",
        color="sentiment_label",
        facet_col="category",
        barmode="group",
        color_discrete_map={"positive": "#90ee90", "neutral": "grey", "negative": "#ff9999"},
    )
    fig3.update_layout(yaxis_title="Percentage (%)", showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.write("Despite small changes, the overall community tone remains stable after the launch of ChatGPT.")

    st.divider()
    st.subheader("Sentiment Score Statistics")
    sentiment_comparison = (
        comments.groupby("category")["sentiment_score"]
        .agg(mean_score="mean", median_score="median", std_score="std", count="count")
        .round(4)
        .reindex(["pre-ChatGPT", "post-ChatGPT"])
    )
    st.dataframe(sentiment_comparison, use_container_width=True)
    st.write("There is a slight increase in positivity and variation after ChatGPT, but the overall change remains minimal.")


elif section == "Forecast":
    st.header("Prophet Forecasting — ML Predictive Modelling")
    st.write(
        """
        A Prophet model is trained on pre-ChatGPT data and used to forecast forward.
    The gap between the forecast (expected trend without ChatGPT) and the
    actual values provides an estimate of how patterns changed after its launch.
        """
    )

    show_ci = st.checkbox("Show 95% confidence interval", value=True)
    st.divider()

    st.subheader("Part 1 — Question Volume: Actual vs Forecast")
    results_post_q = results_questions[results_questions["ds"] >= CHATGPT_LAUNCH]

    fig, ax = plt.subplots(figsize=(12, 5))
    if show_ci:
        ax.fill_between(results_post_q["ds"], results_post_q["yhat_lower"], results_post_q["yhat_upper"], alpha=0.15, color="steelblue", label="95% confidence interval")
    ax.plot(results_post_q["ds"], results_post_q["yhat"], color="green", linewidth=2, linestyle="--", label="Prophet Questions Forecast")
    ax.plot(results_questions["ds"], results_questions["y"], color="red", linewidth=2.5, label="Actual Question Volume")
    ax.axvline(CHATGPT_LAUNCH, color="black", linestyle="--", linewidth=1.5, label="ChatGPT launch (Nov 2022)")
    ax.set_title("Stack Overflow Question Volume: Actual vs Forecast", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Questions")
    ax.legend(fontsize=10)
    style_axis(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    post_q = results_questions[results_questions["ds"] >= CHATGPT_LAUNCH]
    avg_gap_q = post_q["effect"].mean()
    pct_eff_q = avg_gap_q / post_q["yhat"].mean() * 100
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Forecast (expected)", f"{post_q['yhat'].mean():,.0f} / month")
    col2.metric("Avg Actual", f"{post_q['y'].mean():,.0f} / month")
    col3.metric("Monthly Gap", f"{avg_gap_q:+,.0f}", f"{pct_eff_q:+.1f}% vs forecast")

    st.write("Question volume declines after November 2022; forecast values indicate higher expected values without ChatGPT's launch.")
    st.divider()

    st.subheader("Part 2 — Comment Sentiment: Actual vs Forecast")
    results_sentiment_post = results_sentiment[results_sentiment["ds"] >= CHATGPT_LAUNCH]

    fig2, ax = plt.subplots(figsize=(12, 5))
    if show_ci:
        ax.fill_between(results_sentiment_post["ds"], results_sentiment_post["yhat_lower"], results_sentiment_post["yhat_upper"], alpha=0.15, color="steelblue", label="95% confidence interval")
    ax.plot(results_sentiment_post["ds"], results_sentiment_post["yhat"], color="green", linewidth=2, linestyle="--", label="Prophet Sentiment Forecast")
    ax.plot(results_sentiment["ds"], results_sentiment["y"], color="red", linewidth=2.5, label="Actual Sentiment")
    ax.axvline(CHATGPT_LAUNCH, color="black", linestyle="--", linewidth=1.5, label="ChatGPT launch (Nov 2022)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("Stack Overflow Comment Sentiment: Actual vs Forecast", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Sentiment Score (-1 to +1)")
    ax.set_ylim(-0.05, 0.25)
    ax.legend(fontsize=10, loc="lower right")
    style_axis(ax)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    post_s = results_sentiment[results_sentiment["ds"] >= CHATGPT_LAUNCH]
    avg_gap_s = post_s["effect"].mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Forecast Sentiment", f"{post_s['yhat'].mean():+.4f}")
    col2.metric("Avg Actual Sentiment", f"{post_s['y'].mean():+.4f}")
    col3.metric("Avg Monthly Effect", f"{avg_gap_s:+.4f}")
    st.write("Sentiment remained above forecast levels, with stable community mood.")


elif section == "Conclusion":
    st.header("Results, Insights and Conclusion")
    st.success(
        """
        Stack Overflow is being used differently since ChatGPT launched:
        **not abandoned, but evolving**. User interaction is changing, from
        asking routine questions to discussing problems that AI cannot yet
        reliably solve.
        """
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Activity Findings")
        st.markdown(
            """
            - Question volume **declined** and fell below Prophet forecast
            - New user registrations **surged**, especially after 2024
            - Both upvotes and downvotes decreased, indicating **lower overall engagement**
            - The platform is attracting users who interact less with existing content
            """
        )

    with col2:
        st.subheader("Sentiment Findings")
        st.markdown(
            """
            - Sentiment distribution across comments: ~50% positive, ~30% neutral, ~20% negative
            - **No noticeable shift** in sentiment after ChatGPT launched
            - Actual sentiment remained **slightly above** Prophet forecast
            - Overall community sentiment remained stable
            """
        )

    st.divider()
    st.subheader("Summary Table")
    conclusion_df = pd.DataFrame(
        {
            "Metric": [
                "Monthly question volume",
                "New user registrations",
                "Upvotes & downvotes",
                "Comment sentiment",
                "Forecast vs actual (questions)",
                "Forecast vs actual (sentiment)",
            ],
            "Direction": [
                "Decreased",
                "Increased",
                "Both decreased",
                "Stable",
                "Gap widens post-launch",
                "Actual >= forecast",
            ],
            "Interpretation": [
                "Users redirecting simpler questions to AI",
                "Platform still attracting newcomers",
                "Lower overall engagement and quality signals",
                "Community mood largely unaffected by ChatGPT",
                "ChatGPT appears to have influenced question volume",
                "Sentiment held up better than expected",
            ],
        }
    )
    st.dataframe(conclusion_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Limitations")
    st.warning(
        """
        - **VADER** struggles with sarcasm and complex domain-specific language
        - **Correlation does not imply causation**; other factors may explain the trends
        - Sentiment analysis is based on a sample of 50,000 comments, not the full dataset
        """
    )

    st.divider()
    st.subheader("Final Takeaway")
    st.info(
        """
        **Stack Overflow is evolving.** With AI tools such as ChatGPT becoming
        more popular, users appear to be shifting quick, routine questions to AI
        while leaving more complex questions and community-based support to the
        platform. Community sentiment has remained stable throughout this transition.
        """
    )
