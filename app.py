# import libraries
import streamlit as st          
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# page config
st.set_page_config(
    page_title = "Impact of ChatGPT on Stack Overflow",                                 
    layout     = "wide"                            
)

CHATGPT_LAUNCH = pd.Timestamp('2022-11-01')

# Data Loading with Caching
@st.cache_data
def load_data():
    '''load and clean data'''
    questions = pd.read_csv("data/question_volume.csv")
    users = pd.read_csv("data/user_registration.csv")
    votes = pd.read_csv("data/votes_over_time.csv")
    comments = pd.read_csv("data/comments.csv")

    # add date column
    for df in [questions, users, votes, comments]:
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

    # convert comments to string type
    comments['Text'] = comments['Text'].fillna('').astype(str)

    # add pre/post category
    for df in [questions, users, votes, comments]:
        df['category'] = 'post-ChatGPT'
        df.loc[df['date'] < '2022-11-01', 'category'] = 'pre-ChatGPT'
    
    # calculate upvote, downvote ratios
    votes['downvote_ratio'] = votes['downvotes']/votes['total_votes']
    votes['upvote_ratio']   = votes['upvotes']/votes['total_votes']

    return questions, users, votes, comments

@st.cache_data
def run_vader(comments):
    analyzer = SentimentIntensityAnalyzer()
    comments['sentiment_score'] = comments['Text'].apply(
        lambda a: analyzer.polarity_scores(a)['compound']
    )

    def label_sentiment(score):
        if score > 0.05: return 'positive'
        if score < -0.05: return 'negative'
        return 'neutral'

    comments['sentiment_label'] = comments['sentiment_score'].apply(label_sentiment)
    return comments

@st.cache_data
def build_monthly_sentiment(comments):
    monthly_sentiment = (
        comments
        .groupby('date')['sentiment_score']
        .agg(avg_sentiment='mean', std_sentiment='std', comment_count='count')
        .reset_index()
        .sort_values('date')
    )
    return monthly_sentiment

@st.cache_data
def run_prophet_questions(questions):
    prophet_questions = (
        questions[['date', 'question_count']]
        .rename(columns={'date': 'ds', 'question_count': 'y'})
        .sort_values('ds').reset_index(drop=True)
    )
    
    train = prophet_questions[prophet_questions['ds'] < CHATGPT_LAUNCH].copy()
    test = prophet_questions[prophet_questions['ds'] >= CHATGPT_LAUNCH].copy()

    model_questions = Prophet(
        yearly_seasonality=True, weekly_seasonality=False,
        daily_seasonality=False, interval_width=0.95
    )

    model_questions.fit(train)

    future = model_questions.make_future_dataframe(periods=len(test), freq='MS')

    forecast_questions = model_questions.predict(future)
    results_questions = prophet_questions.merge(
        forecast_questions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left'
    )
    results_questions['effect'] = results_questions['y'] - results_questions['yhat']

    return results_questions

@st.cache_data
def run_prophet_sentiment(monthly_sentiment):

    prophet_sentiment = (
        monthly_sentiment[['date', 'avg_sentiment']]
        .rename(columns={'date': 'ds', 'avg_sentiment': 'y'})
        .sort_values('ds').reset_index(drop=True)
    )
    
    train_s = prophet_sentiment[prophet_sentiment['ds'] < CHATGPT_LAUNCH].copy()
    test_s = prophet_sentiment[prophet_sentiment['ds'] >= CHATGPT_LAUNCH].copy()

    model_sentiment = Prophet(
        yearly_seasonality=True, weekly_seasonality=False,
        daily_seasonality=False, interval_width=0.95
    )
    model_sentiment.fit(train_s)

    future_s = model_sentiment.make_future_dataframe(periods=len(test_s), freq='MS')
    forecast_sentiment = model_sentiment.predict(future_s)
    results_sentiment = prophet_sentiment.merge(
        forecast_sentiment[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left'
    )
    results_sentiment['effect'] = results_sentiment['y'] - results_sentiment['yhat']

    return results_sentiment

# st.spinner() shows a loading message while the code inside the "with" block is running
with st.spinner("Loading data and running models: Please wait"):
    questions, users, votes, comments = load_data()
    comments = run_vader(comments)
    monthly_sentiment = build_monthly_sentiment(comments)
    results_questions = run_prophet_questions(questions)
    results_sentiment = run_prophet_sentiment(monthly_sentiment)

# SIDEBAR

# st.radio(): Creates a list of options for user to pick one
# st.slider(): Creates a draggable slider
# st.divider(): Draws a horizontal separator line

with st.sidebar:

    st.title("Navigation")
    # user chooses a section, which  is stored in 'section'
    section = st.radio(
        "Go to",
        options = [
            "Overview",
            "Activity Trends",
            "Sentiment Analysis",
            "Forecast",
            "Conclusion"
        ]
    )
    st.divider()

    st.title("Controls")
    # User drags the slider, which  is stored in 'window'
    # Used in calculating rolling averages for charts
    window = st.slider(
        label = "Smoothing window (months)",
        min_value = 1,
        max_value = 12,
        value = 3,
        help = "Higher = smoother line. Lower = more detail."
    )
    st.divider()
    st.caption("*Analysis: Python · pandas · matplotlib · plotly · vaderSentiment · prophet*")
    
# PAGE TITLE
st.title("Impact of ChatGPT on Stack Overflow")
st.markdown("*How ChatGPT reshaped a tech community: an analysis of user activity and sentiment*")
st.divider()

# PAGE NAVIGATION

# ========================
#  PAGE 1 — OVERVIEW
# ========================

# Calculate values 
pre_ques  = questions[questions['category'] == 'pre-ChatGPT']['question_count'].mean()
post_ques = questions[questions['category'] == 'post-ChatGPT']['question_count'].mean()
pre_user = users[users['category'] == 'pre-ChatGPT']['new_users'].mean()
post_user = users[users['category'] == 'post-ChatGPT']['new_users'].mean()
pct_positive = (comments['sentiment_label'] == 'positive').mean() * 100
avg_sentiment = comments['sentiment_score'].mean()

if section == "Overview":

    st.header("Project Overview")
    st.write("""
    This dashboard explores how activity on Stack Overflow has changed after the launch of ChatGPT in November 2022.
    It focuses on four key areas: **question volume**, **new user registrations**,
    **community voting**, and **comment sentiment**.
    
    The analysis is based on data from 2019 to April 2026.
             
    Data source: [Stack Exchange Data Explorer](https://data.stackexchange.com/stackoverflow/queries)
    """)
    st.divider()

    st.subheader("Key Numbers at a Glance")
    col1, col2, col3, col4 = st.columns(4)

    # st.metric() shows a styled number card
    col1.metric(
        label = "Avg Monthly Questions",
        value = f"{post_ques:,.0f}",
        delta = f"{post_ques - pre_ques:+,.0f} vs pre-ChatGPT"
    )
    col2.metric(
        label = "Avg Monthly New Users",
        value = f"{post_user:,.0f}",
        delta = f"{post_user - pre_user :+,.0f} vs pre-ChatGPT"
    )
    col3.metric(
        label = "Positive Comments",
        value = f"{pct_positive:.1f}%",
        delta = "of all 50k comments"
    )
    col4.metric(
        label = "Avg Sentiment Score",
        value = f"{avg_sentiment:+.3f}",
        delta = "VADER compound score"
    )
    st.divider()

    st.subheader("Major Highlights")

    # Coloured callout boxes:
    #   st.info() --> blue box (neutral information)
    #   st.success() --> green box (positive finding)
    #   st.warning() --> yellow box (caution or limitation)
    #   st.error() --> red box (error or critical warning)

    col_a, col_b, col_c = st.columns(3)
    col_a.success("New user registrations **increased** after ChatGPT launch")
    col_b.info("Community sentiment remains **stable** overall")
    col_c.warning("Question volume **declining** below forecast")

    st.info("""
    Stack Overflow appears to be used differently following the launch of ChatGPT.
    This may indicate a shift away from simple, common questions rather than a reduction in engagement.""")

# ===============================
#  PAGE 2 — ACTIVITY TRENDS
# ===============================

elif section == "Activity Trends":
    st.header("Activity Trends Over Time")
    st.write("Use the **smoothing window** slider in the sidebar to adjust how much the lines are smoothed." \
    " A smaller window makes sudden fluctuations visible, whereas a larger window displays long-term trends.")

    st.subheader("Question Volume Over Time")
    questions['rolling_avg'] = questions['question_count'].rolling(window=window).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(questions['date'], questions['question_count'],
           color='steelblue', alpha=0.4, width=20, label='Monthly Questions')
    ax.plot(questions['date'], questions['rolling_avg'],
            color='steelblue', linewidth=2.5, label=f'{window}-Month Rolling Average')
    ax.axvline(pd.Timestamp('2022-11-01'), color='red', linestyle='--',
               linewidth=1.5, label='ChatGPT Launch (Nov 2022)')
    ax.set_title('Stack Overflow Question Volume Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Questions', fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    st.pyplot(fig)   
    plt.close()      

    st.caption("Question volume shows a decreasing trend, with a sharper decline observed after the ChatGPT launch.")

    # new user plot as left column and voting plot as right column
    st.subheader("New Users and Voting Behaviour")
    col1, col2 = st.columns(2)

    # Left column: New Users 
    users['rolling_avg'] = users['new_users'].rolling(window=window).mean()

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(users['date'], users['new_users'],
            color='green', alpha=0.4, width=20, label='Monthly New Users')
    ax1.plot(users['date'], users['rolling_avg'],
             color='green', linewidth=2.5, label=f'{window}-Month Rolling Avg')
    ax1.axvline(pd.Timestamp('2022-11-01'), color='red', linestyle='--',
                linewidth=1.5, label='ChatGPT Launch')
    ax1.set_title('New User Registrations', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Number of New Users', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.tick_params(axis='x', rotation=45)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.tight_layout()

    col1.pyplot(fig1)   
    plt.close()
    col1.caption("New registrations surged after 2024, indicating that the community is growing with newcomers.")

    # Right column: Votes 
    votes['upvotes_rolling'] = votes['upvotes'].rolling(window=window).mean()
    votes['downvotes_rolling'] = votes['downvotes'].rolling(window=window).mean()

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(votes['date'], votes['upvotes_rolling'],
             color='green', linewidth=2.5, label='Upvotes')
    ax2.plot(votes['date'], votes['downvotes_rolling'],
             color='red', linewidth=2.5, label='Downvotes')
    ax2.axvline(pd.Timestamp('2022-11-01'), color='black', linestyle='--',
                linewidth=1.5, label='ChatGPT Launch')
    ax2.set_title('Community Votes Over Time', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Number of Votes', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.tick_params(axis='x', rotation=45)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()

    col2.pyplot(fig2)   
    plt.close()
    col2.caption("Overall votes have reduced with significant drop in upvotes, suggesting decreasing user contribution and engagement.")
    st.divider()

    st.subheader("Pre vs Post ChatGPT — Activity Summary")
    def pct_change(pre, post):
        return (post - pre) / pre * 100
    
    pre_dv  = votes[votes['category']=='pre-ChatGPT']['downvote_ratio'].mean()
    post_dv = votes[votes['category']=='post-ChatGPT']['downvote_ratio'].mean()
    pre_uv  = votes[votes['category']=='pre-ChatGPT']['upvote_ratio'].mean()
    post_uv = votes[votes['category']=='post-ChatGPT']['upvote_ratio'].mean()

    summary_df = pd.DataFrame({
        'Metric': ['Avg Monthly Questions', 'Avg Monthly New Users',
                   'Avg Downvote Ratio', 'Avg Upvote Ratio'],
        'Pre-ChatGPT': [f"{pre_ques:,.0f}", f"{pre_user:,.0f}",
                        f"{pre_dv:.4f}", f"{pre_uv:.4f}"],
        'Post-ChatGPT': [f"{post_ques:,.0f}", f"{post_user:,.0f}",
                         f"{post_dv:.4f}", f"{post_uv:.4f}"],
        'Change': [f"{pct_change(pre_ques, post_ques):+.1f} %",
                   f"{pct_change(pre_user, post_user):+.1f} %",
                   f"{pct_change(pre_dv, post_dv):+.1f} %",
                   f"{pct_change(pre_uv, post_uv):+.1f} %"]
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ===================================
#  PAGE 3 — SENTIMENT ANALYSIS
# ===================================

elif section == "Sentiment Analysis":

    st.header("Sentiment Analysis Using VADER")
    st.write("""
    50,000 Stack Overflow comments were analyzed and scored using VADER. 
    Each comment was assigned a compound score from −1.0 (very negative) to +1.0 (very positive), and then labeled as positive, neutral, or negative.
    """)
    st.divider()

    st.subheader("Distribution of Comment Sentiment Labels")
    label_counts = comments['sentiment_label'].value_counts().reset_index()
    label_counts.columns = ['sentiment_label', 'count']
    label_counts['percent'] = label_counts['count'] / label_counts['count'].sum()

    fig = px.bar(
        label_counts,
        x='sentiment_label', y='percent',
        color='sentiment_label',
        color_discrete_map={'positive':'#2ecc71','neutral':'#95a5a6','negative':'#e74c3c'},
        text=label_counts['percent'].apply(lambda x: f"{x:.1%}")
        # title='Distribution of Comment Sentiment Labels'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title='Sentiment Label', yaxis_title='Percentage',
                      yaxis_tickformat='.0%', showlegend=False)

    st.plotly_chart(fig, use_container_width=True)  
    st.caption("Half of all the comments having positive sentiment indicates that overall tone of the community remains stable. Only a smaller share of users express dissatisfaction or concerns.")
    st.divider()

    st.subheader("Comment Sentiment Over Time")
    monthly_sentiment['rolling_avg'] = monthly_sentiment['avg_sentiment'].rolling(window=window).mean()

    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(monthly_sentiment['date'], monthly_sentiment['avg_sentiment'],
                    alpha=0.2, color='steelblue', label='Monthly average')
    ax.plot(monthly_sentiment['date'], monthly_sentiment['rolling_avg'],
            color='steelblue', linewidth=2.5, label=f'{window}-month rolling average')
    ax.axvline(CHATGPT_LAUNCH, color='red', linestyle='--',
               linewidth=1.5, label='ChatGPT launch (Nov 2022)')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    ax.set_ylim(-0.05, 0.25)
    ax.set_title('Stack Overflow Comment Sentiment Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Avg VADER Compound Score', fontsize=12)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    st.pyplot(fig2)
    plt.close()

    st.caption("Sentiment stays largely positive over time, with minor fluctuations and no noticeable shift after the ChatGPT launch. Monthly variability reflects the natural changes in community tone.")
    st.divider()

    st.subheader("Pre vs Post ChatGPT — Sentiment Label Mix")
    label_mix = (
        comments.groupby(['category', 'sentiment_label']).size()
        .unstack(fill_value=0)
        .apply(lambda row: row / row.sum() * 100, axis=1).round(1)
    )
    label_mix = label_mix.loc[['pre-ChatGPT', 'post-ChatGPT']]

    label_mix_long = (
        label_mix.reset_index()
        .melt(id_vars='category', var_name='sentiment_label', value_name='percentage')
    )

    fig3 = px.bar(
        label_mix_long, x='sentiment_label', y='percentage',
        color='sentiment_label', facet_col='category', barmode='group',
        color_discrete_map={'positive':'#90ee90','neutral':'grey','negative':'#ff9999'}
        # title='Sentiment Label Mix: Pre vs Post ChatGPT'
    )
    fig3.update_layout(yaxis_title='Percentage (%)', showlegend=False)

    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Despite small changes, the overall tone of the community remain stable, indicating no meaningful shift following the launch of ChatGPT.") 
    st.divider()

    st.subheader("Sentiment Score Statistics")
    sentiment_comparison = (
        comments.groupby('category')['sentiment_score']
        .agg(mean_score='mean', median_score='median', std_score='std', count='count')
        .round(4)
    )
    sentiment_comparison = sentiment_comparison.loc[['pre-ChatGPT', 'post-ChatGPT']]
    st.dataframe(sentiment_comparison, use_container_width=True)
    st.caption("There is a slight increase in both positivity and variation in comments after the launch of ChatGPT, but the overall magnitude of change remains minimal.") 

# ==========================
#  PAGE 4 — FORECAST
# ==========================

elif section == "Forecast":

    st.header("Prophet Forecasting — ML Predictive Modelling")
    st.write("""
    A Prophet model is trained on pre-ChatGPT data and used to forecast forward.
    The gap between the forecast (expected trend without ChatGPT) and the
    actual values provides an estimate of how patterns changed after its launch.
    """)

    # st.checkbox() : allows user to select whether to show the confidence interval band or not
    show_ci = st.checkbox("Show 95% confidence interval", value=True)
    st.divider()

    # --- Part 1: Question Volume Forecast ------------------
    st.subheader("Part 1 — Question Volume: Actual vs Forecast")
    results_post_q = results_questions[results_questions['ds'] >= CHATGPT_LAUNCH]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Only show the confidence interval band if the checkbox is ticked
    if show_ci:
        ax.fill_between(results_post_q['ds'],
                        results_post_q['yhat_lower'], results_post_q['yhat_upper'],
                        alpha=0.15, color='steelblue', label='95% confidence interval')
        
    ax.plot(results_post_q['ds'], results_post_q['yhat'],
            color='green', linewidth=2, linestyle='--',
            label='Prophet Questions Forecast (Post-ChatGPT)')
    ax.plot(results_questions['ds'], results_questions['y'],
            color='red', linewidth=2.5, label='Actual Question Volume')
    ax.axvline(CHATGPT_LAUNCH, color='black', linestyle='--',
               linewidth=1.5, label='ChatGPT launch (Nov 2022)')
    ax.set_title('Stack Overflow Question Volume: Actual vs Forecast', fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('Monthly Questions')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close()

    # Display effect (change in values) as metric cards
    post_q = results_questions[results_questions['ds'] >= CHATGPT_LAUNCH]
    avg_gap_q = post_q['effect'].mean()
    pct_eff_q = (avg_gap_q / post_q['yhat'].mean()) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Forecast (expected)", f"{post_q['yhat'].mean():,.0f} / month")
    col2.metric("Avg Actual", f"{post_q['y'].mean():,.0f} / month")
    col3.metric("Monthly Gap", f"{avg_gap_q:+,.0f}", f"{pct_eff_q:+.1f}% vs forecast")

    st.caption("Question volume declines after November 2022, forecast values indicate higher expected values without ChatGPT's launch.")
    st.divider()

    # --- Part 2: Sentiment Forecast ------------------
    st.subheader("Part 2 — Comment Sentiment: Actual vs Forecast")
    results_sentiment_post = results_sentiment[results_sentiment['ds'] >= CHATGPT_LAUNCH]

    fig2, ax = plt.subplots(figsize=(12, 5))

    if show_ci:
        ax.fill_between(results_sentiment_post['ds'],
                        results_sentiment_post['yhat_lower'],
                        results_sentiment_post['yhat_upper'],
                        alpha=0.15, color='steelblue', label='95% confidence interval')
    
    ax.plot(results_sentiment_post['ds'], results_sentiment_post['yhat'],
            color='green', linewidth=2, linestyle='--',
            label='Prophet Sentiment Forecast (Post-ChatGPT)')
    ax.plot(results_sentiment['ds'], results_sentiment['y'],
            color='red', linewidth=2.5, label='Actual Sentiments')
    ax.axvline(CHATGPT_LAUNCH, color='black', linestyle='--', linewidth=1.5,
               label='ChatGPT launch (Nov 2022)')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    ax.set_title('Stack Overflow Comment Sentiments: Actual vs Forecast', fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('Monthly Sentiment Score (−1 to +1)')
    # ax.legend(fontsize=10)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(-0.05, 0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    st.pyplot(fig2)
    plt.close()

    post_s    = results_sentiment[results_sentiment['ds'] >= CHATGPT_LAUNCH]
    avg_gap_s = post_s['effect'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Forecast Sentiment", f"{post_s['yhat'].mean():+.4f}")
    col2.metric("Avg Actual Sentiment", f"{post_s['y'].mean():+.4f}")
    col3.metric("Avg Monthly Effect", f"{avg_gap_s:+.4f}")

    st.caption("Sentiment remained above forecast levels, with stable community mood.")

# ===============================
#  PAGE 5 — CONCLUSION
# ===============================

elif section == "Conclusion":

    st.header("Results, Insights and Conclusion")
    st.success("""
    Stack Overflow is being used differently since ChatGPT launched: **not abandoned, but evolving**.
    User interaction is changing, from asking routine questions to discussing problems that AI cannot yet reliably solve.
    """)
    st.divider()

    # create 2 columns to write findings
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Activity Findings")
        st.markdown("""
        - Question volume **declined** and fell below Prophet forecast
        - New user registrations **surged**, especially after 2024
        - Both upvotes and downvotes decreased, indicating **lower overall engagement**
        - The platform is attracting users who interact less with existing content
        """)

    with col2:
        st.subheader("Sentiment Findings")
        st.markdown("""
        - Sentiment distribution across all comments: ~50% positive, ~30% neutral, ~20% negative
        - **No noticeable shift** in sentiment after ChatGPT launched
        - Actual sentiment remained **slightly above** Prophet forecast
        - Overall community sentiment remained stable
        """)
    
    st.divider()
    st.subheader("Summary Table")

    conclusion_df = pd.DataFrame({
        'Metric': [
            'Monthly question volume', 'New user registrations', 'Upvotes & downvotes', 
            'Comment sentiment', 'Forecast vs actual (questions)', 'Forecast vs actual (sentiment)'
        ],
        'Direction': [
            '↓ Decreased', '↑ Increased', '↓ Both decreased',
            '→ Stable', 'Gap widens post-launch', 'Actual ≥ forecast'
        ],
        'Interpretation': [
            'Users redirecting simpler questions to AI',
            'Platform still attracting newcomers',
            'Lower overall engagement as well quality',
            'Community mood largely unaffected by ChatGPT',
            'ChatGPT appears to have influenced question volume',
            'Sentiment held up better than expected'
        ]
    })
    st.dataframe(conclusion_df, use_container_width=True, hide_index=True)
    st.divider()

    st.subheader("Limitations")
    st.warning("""
    - **VADER** struggles with sarcasm and complex sentences with domain-specific terms
    - **Correlation does not imply causation**, other factors may also explain the trends
    - Sentiment analysis is based on a sample of 50,000 comments, not the full dataset
    """)
    st.divider()

    st.subheader("Final Takeaway")
    st.info("""
    **Stack Overflow is evolving.** 
    With AI tools such as ChatGPT becoming more popularized, users appear to be shifting asking quick, routine questions to AI, 
    while leaving more complex questions and community-based support to the platform. 
    Community sentiment has remained stable throughout this transition, suggesting users are not frustrated with the platform, 
    but are instead changing how they interact with it and where they seek different types of answers.
    """)
