# Impact of ChatGPT on Stack Overflow
How ChatGPT reshaped a tech community: an analysis of user activity and sentiment

## Project Overview
This dashboard explores how activity on Stack Overflow has changed after the launch of ChatGPT in November 2022. It focuses on four key areas: 
- question volume: monthly question counts
- new user registrations: monthly new user registrations
- community voting: monthly upvote and downvote counts
- comment sentiment: 50,000 comments for sentiment analysis

The analysis is based on data from 2019 to April 2026.

Data source: [Stack Exchange Data Explorer](https://data.stackexchange.com/stackoverflow/queries)

## Libraries Used
- pandas
- matplotlib
- plotly
- vaderSentiment
- prophet

## Analysis
**Activity Findings**
- Question volume declined and fell below Prophet forecast
- New user registrations surged, especially after 2024
- Both upvotes and downvotes decreased, indicating lower overall engagement
- The platform is attracting users who interact less with existing content

**Sentiment Findings**
- Sentiment distribution across all comments: ~50% positive, ~30% neutral, ~20% negative
- No noticeable shift in sentiment after ChatGPT launched
- Actual sentiment remained slightly above Prophet forecast
- Overall community sentiment remained stable

## Limitations
- VADER struggles with sarcasm and complex sentences with domain-specific terms
- Correlation does not imply causation, other factors may also explain the trends
- Sentiment analysis is based on a sample of 50,000 comments, not the full dataset

## **Final Takeaway**

Stack Overflow is evolving. With AI tools such as ChatGPT becoming more popularized, users appear to be shifting asking quick, routine questions to AI, while leaving more complex questions and community-based support to the platform. Community sentiment has remained stable throughout this transition, suggesting users are not frustrated with the platform, but are instead changing how they interact with it and where they seek different types of answers.
