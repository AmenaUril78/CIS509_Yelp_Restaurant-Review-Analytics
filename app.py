import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Independent Restaurant Diagnostic Tool",
    layout="wide"
)

st.title("🍽 Independent Restaurant Operational Diagnostic Tool")

st.markdown("""
This dashboard operationalizes our ProjectEDA framework:

- Independent sit-down restaurants (< 5 locations)
- At-Risk (≤ 2.5 stars)
- Excellent (≥ 4.5 stars)
- Reviews filtered through 2018 (pre-pandemic baseline)
""")

st.divider()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Upload Yelp Dataset")

business_file = st.sidebar.file_uploader("Upload Business JSON", type=["json"])
review_file = st.sidebar.file_uploader("Upload Review JSON", type=["json"])

sample_size = st.sidebar.slider(
    "Review Sample Size (for speed)",
    2000, 15000, 5000, 1000
)

if business_file is None or review_file is None:
    st.info("Please upload both Business and Review JSON files.")
    st.stop()

# ---------------------------------------------------
# JSON LOADER
# ---------------------------------------------------

@st.cache_data(show_spinner=True)
def load_json(file):
    file.seek(0)
    content = file.read().decode("utf-8").strip()

    if content.startswith("["):
        return pd.DataFrame(json.loads(content))

    rows = []
    for line in content.splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except:
                continue

    return pd.DataFrame(rows)

business_df = load_json(business_file)

# ---------------------------------------------------
# FILTER LOGIC
# ---------------------------------------------------

business_df = business_df[
    business_df["categories"].fillna("").str.contains("Restaurants", case=False)
]

name_counts = business_df["name"].value_counts()
small_names = name_counts[name_counts < 5].index
business_df = business_df[business_df["name"].isin(small_names)]

business_df = business_df[
    (business_df["stars"] <= 2.5) |
    (business_df["stars"] >= 4.5)
].copy()

business_df["segment"] = business_df["stars"].apply(
    lambda x: "At-Risk (≤2.5)" if x <= 2.5 else "Excellent (≥4.5)"
)

st.success(f"Filtered Independent Sit-Down Restaurants: {business_df.shape[0]}")

st.divider()

# ---------------------------------------------------
# LOAD REVIEWS
# ---------------------------------------------------

review_df = load_json(review_file)

review_df["date"] = pd.to_datetime(review_df["date"], errors="coerce")
review_df = review_df[review_df["date"] <= "2018-12-31"]

valid_ids = set(business_df["business_id"])
review_df = review_df[review_df["business_id"].isin(valid_ids)].copy()

review_df = review_df.head(sample_size)

merged_df = review_df.merge(
    business_df[["business_id", "segment", "name"]],
    on="business_id",
    how="left"
)

sia = SentimentIntensityAnalyzer()

merged_df["sentiment_score"] = merged_df["text"].apply(
    lambda x: sia.polarity_scores(str(x))["compound"]
)

# =====================================================
# EXECUTIVE PORTFOLIO OVERVIEW
# =====================================================

st.header("📊 Executive-Level Portfolio Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", len(merged_df))
col2.metric("Unique Restaurants", merged_df["business_id"].nunique())
col3.metric("Average Sentiment (All)", f"{merged_df['sentiment_score'].mean():.3f}")

colA, colB = st.columns(2)

with colA:
    st.markdown("### Review Count by Segment")
    st.bar_chart(merged_df["segment"].value_counts())

with colB:
    st.markdown("### Sentiment Distribution by Segment")
    fig, ax = plt.subplots()
    sns.boxplot(data=merged_df, x="segment", y="sentiment_score", ax=ax)
    st.pyplot(fig)

st.divider()

# =====================================================
# INDIVIDUAL RESTAURANT DIAGNOSTIC
# =====================================================

st.header("🏢 Individual Restaurant Diagnostic")

restaurants_with_reviews = merged_df["name"].unique()

if len(restaurants_with_reviews) == 0:
    st.warning("No restaurants with usable reviews available.")
    st.stop()

selected_restaurant = st.selectbox(
    "Select an Independent Restaurant:",
    sorted(restaurants_with_reviews)
)

if selected_restaurant:

    selected_bid = business_df[
        business_df["name"] == selected_restaurant
    ]["business_id"].iloc[0]

    restaurant_reviews = merged_df[
        merged_df["business_id"] == selected_bid
    ].copy()

    avg_sent = restaurant_reviews["sentiment_score"].mean()
    segment_label = restaurant_reviews["segment"].iloc[0]

    # ---------------------------------------------------
    # PERFORMANCE OVERVIEW
    # ---------------------------------------------------

    st.subheader(f"Performance Overview: {selected_restaurant}")

    colA, colB, colC = st.columns(3)

    colA.metric("Segment", segment_label)

    colB.metric(
        label="Average Sentiment",
        value=f"{avg_sent:.3f}",
        help="""
VADER sentiment ranges from -1 (very negative) to +1 (very positive).

• Above +0.5 → Strong positive customer language  
• Near 0 → Mixed experiences  
• Below 0 → Negative customer sentiment  

This reflects the overall tone of customer reviews.
"""
    )

    colC.metric("Total Reviews", len(restaurant_reviews))

    if len(restaurant_reviews) < 5:
        st.warning("⚠ Small review sample — interpret results cautiously.")

    st.divider()

    # ---------------------------------------------------
    # ISSUE DETECTION
    # ---------------------------------------------------

    combined_text = " ".join(
        restaurant_reviews["text"].astype(str)
    ).lower()

    issue_buckets = {
        "Service & Staff Attitude": ["rude", "server", "staff", "attitude"],
        "Wait Times & Speed": ["wait", "slow", "minutes", "hour"],
        "Food Quality & Consistency": ["cold", "bland", "overcooked", "undercooked"],
        "Cleanliness & Environment": ["dirty", "bathroom"],
        "Price & Value Perception": ["expensive", "overpriced", "worth"]
    }

    detected_issues = [
        bucket for bucket, keywords in issue_buckets.items()
        if any(word in combined_text for word in keywords)
    ]

    if not detected_issues:
        detected_issues = ["Service & Operational Consistency"]

    # ---------------------------------------------------
    # RISK + SENTIMENT SECTION
    # ---------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚩 Operational Risk Signals")
        st.markdown("**Primary Risk Areas Identified:**")
        for issue in detected_issues[:2]:
            st.markdown(f"**{issue}**")

    with col2:
        st.markdown("### 📊 Customer Sentiment Breakdown")

        def categorize_sentiment(score):
            if score >= 0.05:
                return "Positive"
            elif score <= -0.05:
                return "Negative"
            else:
                return "Neutral"

        restaurant_reviews = restaurant_reviews.copy()
        restaurant_reviews["sentiment_category"] = restaurant_reviews[
            "sentiment_score"
        ].apply(categorize_sentiment)

        sentiment_counts = restaurant_reviews["sentiment_category"].value_counts()
        total = len(restaurant_reviews)

        pos_pct = round(100 * sentiment_counts.get("Positive", 0) / total, 1)
        neu_pct = round(100 * sentiment_counts.get("Neutral", 0) / total, 1)
        neg_pct = round(100 * sentiment_counts.get("Negative", 0) / total, 1)

        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 Negative", f"{neg_pct}%")
        m2.metric("🟡 Neutral", f"{neu_pct}%")
        m3.metric("🟢 Positive", f"{pos_pct}%")

        st.bar_chart(sentiment_counts)

        if neg_pct >= 30:
            st.error("⚠ High proportion of negative reviews detected.")
        elif neg_pct >= 15:
            st.warning("Moderate level of negative customer experiences.")
        else:
            st.success("Customer sentiment is largely positive.")

    st.divider()

    # ---------------------------------------------------
    # FULL-WIDTH IMPROVEMENT PLAN
    # ---------------------------------------------------

    st.markdown("## 📈 Customized Improvement Plan")

    st.markdown("""
**Core Operational Stabilization**
- Audit end-to-end service flow  
- Reinforce hospitality standards  
- Monitor peak-hour staffing  
""")

    for issue in detected_issues[:2]:

        if issue == "Service & Staff Attitude":
            st.markdown("""
**Service Culture Enhancement**
- Weekly hospitality coaching  
- Manager floor accountability  
""")

        elif issue == "Wait Times & Speed":
            st.markdown("""
**Speed Optimization**
- Track kitchen ticket times  
- Pre-batch high-demand items  
""")

        elif issue == "Food Quality & Consistency":
            st.markdown("""
**Food Quality Controls**
- Temperature audits  
- Standardized plating procedures  
""")

        elif issue == "Cleanliness & Environment":
            st.markdown("""
**Cleanliness Initiative**
- Hourly restroom checks  
- Visible sanitation procedures  
""")

        elif issue == "Price & Value Perception":
            st.markdown("""
**Value Perception Strategy**
- Bundle high-margin combos  
- Reinforce value messaging  
""")

    st.success("Improvement plan derived dynamically from Yelp review patterns.")
