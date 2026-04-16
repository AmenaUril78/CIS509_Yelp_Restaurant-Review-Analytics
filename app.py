import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import json
import re
from collections import Counter
from textblob import TextBlob

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Independent Restaurant Diagnostic Tool",
    layout="wide",
    page_icon="🍽️",
)

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
STOP_WORDS = {
    "the","and","was","for","that","with","this","but","they","have","from",
    "our","not","were","all","had","been","their","there","which","when",
    "would","will","are","has","its","more","than","about","just","your",
    "what","get","also","very","good","great","place","food","here","went",
    "got","back","said","came","time","even","like","really","order","then",
    "them","some","over","could","only","well","nice","first","other",
    "after","because","never","again","didn","service","restaurant","ordered"
}

def get_sentiment(text: str) -> float:
    return TextBlob(str(text)).sentiment.polarity

def categorize_sentiment(score: float) -> str:
    if score > 0.05:  return "Positive"
    if score < -0.05: return "Negative"
    return "Neutral"

def top_keywords(text: str, n: int = 15) -> list:
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    words = [w for w in words if w not in STOP_WORDS]
    return Counter(words).most_common(n)

@st.cache_data(show_spinner="Loading data…")
def load_json(file_bytes: bytes) -> pd.DataFrame:
    content = file_bytes.decode("utf-8").strip()
    if content.startswith("["):
        return pd.DataFrame(json.loads(content))
    rows = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)

@st.cache_data(show_spinner="Filtering restaurants…")
def build_business(biz_bytes: bytes) -> pd.DataFrame:
    df = load_json(biz_bytes)
    required = {"categories", "name", "stars", "business_id"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Business file missing columns: {missing}")
        st.stop()
    df = df[df["categories"].fillna("").str.contains("Restaurants", case=False)].copy()
    name_counts = df["name"].value_counts()
    df = df[df["name"].isin(name_counts[name_counts < 5].index)].copy()
    df = df[(df["stars"] <= 2.5) | (df["stars"] >= 4.5)].copy()
    df["segment"] = df["stars"].apply(
        lambda x: "At-Risk (≤2.5⭐)" if x <= 2.5 else "Excellent (≥4.5⭐)"
    )
    return df

@st.cache_data(show_spinner="Analyzing reviews… (this may take a moment)")
def build_reviews(rev_bytes: bytes, valid_ids: frozenset, sample_size: int) -> pd.DataFrame:
    df = load_json(rev_bytes)
    required = {"date", "business_id", "text"}
    if required - set(df.columns):
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] <= "2018-12-31"]
    df = df[df["business_id"].isin(valid_ids)].copy()
    df = df.sample(min(sample_size, len(df)), random_state=42)
    df["sentiment_score"] = df["text"].apply(get_sentiment)
    df["sentiment_category"] = df["sentiment_score"].apply(categorize_sentiment)
    return df

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🍽️ Independent Restaurant Operational Diagnostic Tool")
st.markdown("""
Transforms raw Yelp review data into **actionable operational insights** for independent restaurants.

| Filter | Criteria |
|---|---|
| Chain size | < 5 locations (by name) |
| At-Risk | ≤ 2.5 ⭐ |
| Excellent | ≥ 4.5 ⭐ |
| Review window | Through end of 2018 (pre-pandemic baseline) |
""")
st.divider()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("⚙️ Configuration")
st.sidebar.subheader("Upload Yelp Dataset")

business_file = st.sidebar.file_uploader("📁 Business JSON", type=["json"])
review_file   = st.sidebar.file_uploader("📁 Review JSON",   type=["json"])

sample_size = st.sidebar.slider(
    "Review Sample Size",
    min_value=1000, max_value=16000, value=5000, step=1000,
    help="Higher = more accurate but slower. This dataset has ~16,000 matched reviews."
)

st.sidebar.markdown("---")
st.sidebar.caption("CIS 509 – Unstructured Data Analytics | Team 8")

if business_file is None or review_file is None:
    st.info("👈 Upload both **Business JSON** and **Review JSON** files in the sidebar.")
    st.stop()

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
biz_bytes = business_file.read()
rev_bytes = review_file.read()

business_df = build_business(biz_bytes)

if business_df.empty:
    st.error("No restaurants matched the filter criteria.")
    st.stop()

valid_ids = frozenset(business_df["business_id"])
review_df = build_reviews(rev_bytes, valid_ids, sample_size)

if review_df.empty:
    st.error("No matching reviews found. Ensure both files are from the same Yelp dataset.")
    st.stop()

merged_df = review_df.merge(
    business_df[["business_id", "segment", "name", "stars", "city", "state"]],
    on="business_id", how="left"
)

at_risk_df   = merged_df[merged_df["segment"] == "At-Risk (≤2.5⭐)"]
excellent_df = merged_df[merged_df["segment"] == "Excellent (≥4.5⭐)"]

# =====================================================
# EXECUTIVE PORTFOLIO OVERVIEW
# =====================================================
st.header("📊 Executive-Level Portfolio Overview")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Reviews",         f"{len(merged_df):,}")
c2.metric("Unique Restaurants",    f"{merged_df['business_id'].nunique():,}")
c3.metric("At-Risk Restaurants",   f"{(business_df['segment']=='At-Risk (≤2.5⭐)').sum():,}")
c4.metric("Excellent Restaurants", f"{(business_df['segment']=='Excellent (≥4.5⭐)').sum():,}")
c5.metric("Avg Sentiment (All)",   f"{merged_df['sentiment_score'].mean():.3f}")

st.markdown("---")

colA, colB = st.columns(2)

with colA:
    st.markdown("### Review Count by Segment")
    seg_counts = merged_df["segment"].value_counts()
    colors = ["#e74c3c" if "At-Risk" in s else "#2ecc71" for s in seg_counts.index]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(seg_counts.index, seg_counts.values, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("Number of Reviews")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_title("Reviews per Segment")
    for i, v in enumerate(seg_counts.values):
        ax.text(i, v + 30, f"{v:,}", ha="center", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

with colB:
    st.markdown("### Sentiment Distribution by Segment")
    fig, ax = plt.subplots(figsize=(5, 3))
    palette = {"At-Risk (≤2.5⭐)": "#e74c3c", "Excellent (≥4.5⭐)": "#2ecc71"}
    sns.boxplot(data=merged_df, x="segment", y="sentiment_score", ax=ax, palette=palette)
    ax.set_xlabel("")
    ax.set_ylabel("TextBlob Polarity Score")
    ax.set_title("Sentiment Score by Segment")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

st.markdown("### Segment Sentiment Summary")
summary = (
    merged_df.groupby("segment")["sentiment_score"]
    .agg(Mean="mean", Median="median", Std="std", Reviews="count")
    .round(3).reset_index()
)
st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown("### Top Cities in Dataset")
city_counts = business_df.groupby(["city", "segment"]).size().unstack(fill_value=0)
city_counts["Total"] = city_counts.sum(axis=1)
st.dataframe(city_counts.sort_values("Total", ascending=False).head(10), use_container_width=True)

st.divider()

# =====================================================
# SEGMENT DEEP DIVE
# =====================================================
st.header("🔍 Segment Deep Dive")

tab1, tab2 = st.tabs(["🔴 At-Risk Restaurants", "🟢 Excellent Restaurants"])

def segment_tab(seg_df, label, color):
    top = (
        seg_df.groupby(["name", "city"])["sentiment_score"]
        .agg(Avg_Sentiment="mean", Review_Count="count")
        .sort_values("Review_Count", ascending=False)
        .head(10).reset_index().round(3)
    )
    st.markdown(f"#### Top 10 Most-Reviewed {label} Restaurants")
    st.dataframe(top, use_container_width=True, hide_index=True)

    st.markdown(f"#### Monthly Sentiment Trend — {label}")
    seg_df = seg_df.copy()
    seg_df["year_month"] = seg_df["date"].dt.to_period("M").astype(str)
    trend = seg_df.groupby("year_month")["sentiment_score"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(trend["year_month"], trend["sentiment_score"], color=color, marker="o", markersize=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_ylabel("Avg Sentiment")
    ax.set_title(f"Monthly Avg Sentiment — {label}")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

with tab1:
    segment_tab(at_risk_df, "At-Risk", "#e74c3c")
with tab2:
    segment_tab(excellent_df, "Excellent", "#2ecc71")

st.divider()

# =====================================================
# INDIVIDUAL RESTAURANT DIAGNOSTIC
# =====================================================
st.header("🏢 Individual Restaurant Diagnostic")

merged_df["display_name"] = (
    merged_df["name"] + " — " +
    merged_df["city"].fillna("") + " (" +
    merged_df["stars"].astype(str) + "⭐)"
)
business_df["display_name"] = (
    business_df["name"] + " — " +
    business_df["city"].fillna("") + " (" +
    business_df["stars"].astype(str) + "⭐)"
)

seg_filter = st.radio(
    "Filter by segment:",
    ["All", "At-Risk (≤2.5⭐)", "Excellent (≥4.5⭐)"],
    horizontal=True
)

if seg_filter != "All":
    filtered_names = sorted(
        merged_df[merged_df["segment"] == seg_filter]["display_name"].dropna().unique()
    )
else:
    filtered_names = sorted(merged_df["display_name"].dropna().unique())

selected_display = st.selectbox("Select a Restaurant:", filtered_names)

if selected_display:
    selected_bid = business_df[
        business_df["display_name"] == selected_display
    ]["business_id"].iloc[0]

    restaurant_reviews = merged_df[merged_df["business_id"] == selected_bid].copy()

    if restaurant_reviews.empty:
        st.warning("No reviews found for this restaurant.")
        st.stop()

    avg_sent      = restaurant_reviews["sentiment_score"].mean()
    segment_label = restaurant_reviews["segment"].iloc[0]
    star_rating   = business_df[business_df["business_id"] == selected_bid]["stars"].iloc[0]

    # KPI row
    st.subheader(f"📍 {selected_display}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segment",       segment_label)
    c2.metric("Star Rating",   f"{star_rating} ⭐")
    c3.metric("Avg Sentiment", f"{avg_sent:.3f}",
              help="TextBlob polarity: -1 (very negative) → +1 (very positive)")
    c4.metric("Total Reviews", len(restaurant_reviews))

    if len(restaurant_reviews) < 5:
        st.warning("⚠️ Small review sample — interpret results cautiously.")

    seg_avg = merged_df[merged_df["segment"] == segment_label]["sentiment_score"].mean()
    delta   = avg_sent - seg_avg
    direction = "↑ above" if delta >= 0 else "↓ below"
    st.caption(f"Segment avg: **{seg_avg:.3f}** | This restaurant is **{abs(delta):.3f} {direction}** segment average")

    st.divider()

    # Sentiment over time
    st.markdown("### 📅 Sentiment Trend Over Time")
    time_df = restaurant_reviews.copy()
    time_df["year_month"] = time_df["date"].dt.to_period("M").astype(str)
    trend = time_df.groupby("year_month")["sentiment_score"].mean().reset_index()

    if len(trend) >= 2:
        trend_color = "#e74c3c" if "At-Risk" in segment_label else "#2ecc71"
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(trend["year_month"], trend["sentiment_score"],
                marker="o", color=trend_color, linewidth=2, markersize=5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(seg_avg, color="blue", linestyle=":", linewidth=1,
                   label=f"Segment avg ({seg_avg:.2f})")
        ax.set_xticks(range(len(trend)))
        ax.set_xticklabels(trend["year_month"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Avg Sentiment")
        ax.set_title("Monthly Average Sentiment Score")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("Not enough data points to plot a trend.")

    st.divider()

    # Issue detection
    combined_text = " ".join(restaurant_reviews["text"].astype(str)).lower()

    issue_buckets = {
        "Service & Staff Attitude":   ["rude", "server", "staff", "attitude", "unfriendly",
                                       "unprofessional", "ignored", "manager"],
        "Wait Times & Speed":         ["wait", "slow", "minutes", "hour", "forever",
                                       "long time", "took too long", "rushed"],
        "Food Quality & Consistency": ["cold", "bland", "overcooked", "undercooked", "raw",
                                       "stale", "tasteless", "dry", "soggy"],
        "Cleanliness & Environment":  ["dirty", "bathroom", "filthy", "unclean", "roach",
                                       "pest", "gross", "sticky"],
        "Price & Value Perception":   ["expensive", "overpriced", "worth", "cheap",
                                       "value", "not worth", "pricey"],
        "Order Accuracy":             ["wrong order", "incorrect", "missing item", "forgot",
                                       "not what i ordered", "messed up"],
        "Noise & Atmosphere":         ["loud", "noisy", "crowded", "cramped", "atmosphere",
                                       "ambiance", "dark"],
    }

    issue_scores = {}
    for bucket, keywords in issue_buckets.items():
        count = sum(combined_text.count(kw) for kw in keywords)
        if count > 0:
            issue_scores[bucket] = count

    detected_issues = sorted(issue_scores, key=issue_scores.get, reverse=True)
    if not detected_issues:
        detected_issues = ["Service & Operational Consistency"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚩 Operational Risk Signals")
        st.markdown("Ranked by keyword frequency in reviews:")
        for i, issue in enumerate(detected_issues[:5], 1):
            freq = issue_scores.get(issue, 0)
            bar  = "█" * min(freq // 2, 20)
            st.markdown(f"**{i}. {issue}**")
            st.caption(f"~{freq} mentions  {bar}")

    with col2:
        st.markdown("### 🔑 Top Review Keywords")
        kw = top_keywords(combined_text, 15)
        if kw:
            kw_df = pd.DataFrame(kw, columns=["Word", "Count"])
            bar_color = "#e74c3c" if "At-Risk" in segment_label else "#2ecc71"
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.barh(kw_df["Word"][::-1], kw_df["Count"][::-1], color=bar_color)
            ax.set_xlabel("Frequency")
            ax.set_title("Most Common Review Words")
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

    st.divider()

    # Sentiment breakdown
    st.markdown("### 📊 Customer Sentiment Breakdown")
    sentiment_counts = restaurant_reviews["sentiment_category"].value_counts()
    total = len(restaurant_reviews)
    pos_pct = round(100 * sentiment_counts.get("Positive", 0) / total, 1)
    neu_pct = round(100 * sentiment_counts.get("Neutral",  0) / total, 1)
    neg_pct = round(100 * sentiment_counts.get("Negative", 0) / total, 1)

    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 Negative", f"{neg_pct}%")
    m2.metric("🟡 Neutral",  f"{neu_pct}%")
    m3.metric("🟢 Positive", f"{pos_pct}%")

    sizes  = [s for s in [neg_pct, neu_pct, pos_pct] if s > 0]
    labels = [l for l, s in zip(["Negative","Neutral","Positive"],
                                  [neg_pct, neu_pct, pos_pct]) if s > 0]
    pie_colors = ["#e74c3c", "#f39c12", "#2ecc71"][:len(sizes)]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%",
           startangle=140, wedgeprops=dict(edgecolor="white"))
    ax.set_title("Sentiment Split")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    if neg_pct >= 30:
        st.error("⚠️ High proportion of negative reviews detected.")
    elif neg_pct >= 15:
        st.warning("Moderate level of negative customer experiences.")
    else:
        st.success("Customer sentiment is largely positive.")

    st.markdown("#### 📝 Sample Negative Reviews")
    neg_reviews = restaurant_reviews[
        restaurant_reviews["sentiment_category"] == "Negative"
    ]["text"].head(3)
    if neg_reviews.empty:
        st.info("No negative reviews in the sample.")
    else:
        for rev in neg_reviews:
            st.markdown(f"> _{str(rev)[:350]}{'…' if len(str(rev)) > 350 else ''}_")

    st.divider()

    # Improvement plan
    st.markdown("## 📈 Customized Improvement Plan")

    improvement_map = {
        "Service & Staff Attitude": (
            "Service Culture Enhancement",
            ["Weekly hospitality coaching for all front-of-house staff",
             "Manager accountability walks every peak-hour service",
             "Empathy & de-escalation training for customer-facing roles",
             "Post-shift debrief: review any complaints from that day"]
        ),
        "Wait Times & Speed": (
            "Speed & Throughput Optimization",
            ["Track kitchen ticket times and display on visible board",
             "Pre-batch high-demand items before rush hours",
             "Implement table-ready notification system",
             "Weekly ticket-time analysis — flag and address outliers"]
        ),
        "Food Quality & Consistency": (
            "Food Quality Controls",
            ["Daily temperature audits at the expo/pass station",
             "Standardized portioning cards posted at every station",
             "Blind taste-test QA at start of each shift",
             "Track comps and returns — identify recurring problem dishes"]
        ),
        "Cleanliness & Environment": (
            "Cleanliness Initiative",
            ["Hourly restroom inspection checklist with signature log",
             "Visible hand-sanitizer stations at entrance and tables",
             "Weekly deep-clean schedule posted in kitchen",
             "Monthly health inspection readiness audit"]
        ),
        "Price & Value Perception": (
            "Value Perception Strategy",
            ["Bundle high-margin items into clearly priced combos",
             "Train staff to communicate freshness and quality cues",
             "Introduce a loyalty/punch-card program for repeat guests",
             "Test menu price anchoring by adding premium option"]
        ),
        "Order Accuracy": (
            "Order Accuracy Program",
            ["Read back every order verbally before submitting to kitchen",
             "Color-coded ticket system for dietary restrictions",
             "Track error rate weekly — celebrate lowest-error teams",
             "Double-check at expo before plates leave the kitchen"]
        ),
        "Noise & Atmosphere": (
            "Atmosphere Improvement",
            ["Add acoustic panels or soft furnishings to dampen noise",
             "Adjust background music volume by daypart",
             "Optimize lighting — warmer tones at evening service",
             "Add post-visit survey to capture atmosphere feedback"]
        ),
    }

    with st.expander("🔧 Core Operational Stabilization (Always Applied)", expanded=True):
        st.markdown("""
- Audit end-to-end service flow: host → server → kitchen → expo → table  
- Weekly team huddles (15 min) to reinforce hospitality standards  
- Respond to all online reviews within 48 hours  
- Monitor peak-hour staffing and adjust scheduling proactively  
- Set monthly sentiment improvement targets and track them  
""")

    for issue in detected_issues[:5]:
        if issue in improvement_map:
            title, actions = improvement_map[issue]
            with st.expander(f"🎯 {title}", expanded=False):
                for action in actions:
                    st.markdown(f"- {action}")

    st.success("✅ Improvement plan generated dynamically from Yelp review patterns.")

    st.divider()
    st.markdown("### 🔍 Raw Review Explorer")
    if st.checkbox("Show all reviews for this restaurant"):
        cols_to_show = [c for c in ["date", "stars", "sentiment_score",
                                     "sentiment_category", "text"]
                        if c in restaurant_reviews.columns]
        st.dataframe(
            restaurant_reviews[cols_to_show].sort_values("date", ascending=False),
            use_container_width=True
        )
