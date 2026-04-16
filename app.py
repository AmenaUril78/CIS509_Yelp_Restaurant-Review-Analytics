import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import json
import re
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Independent Restaurant Diagnostic Tool",
    layout="wide",
    page_icon="🍽️",
)

st.title("🍽️ Independent Restaurant Operational Diagnostic Tool")
st.markdown("""
Analyzing **independent sit-down restaurants** from the Yelp Academic Dataset (pre-2019 baseline).

| Filter | Criteria |
|---|---|
| Chain size | < 5 locations (same name) |
| At-Risk segment | ≤ 2.5 ⭐ |
| Excellent segment | ≥ 4.5 ⭐ |
| Review window | Through 2018 (pre-pandemic) |
""")
st.divider()

# ---------------------------------------------------
# STOP WORDS
# ---------------------------------------------------
STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","was","are","were","be","been","have","has","had","do","does","did",
    "will","would","could","should","may","might","not","no","this","that",
    "it","its","we","they","their","our","my","your","he","she","i","you",
    "from","by","as","up","out","so","if","then","than","there","here",
    "just","also","more","some","all","about","over","after","very","really",
    "place","food","got","get","good","great","time","restaurant","went",
    "back","even","one","like","ordered","came","said","much","make","made",
    "come","dont","cant","didnt","wasnt","into","only","well","still","also",
    "never","always","every","each","went","tried","going","being","same",
}

# ---------------------------------------------------
# ISSUE BUCKETS
# ---------------------------------------------------
ISSUE_BUCKETS = {
    "Service & Staff Attitude":   ["rude","server","staff","attitude","unfriendly","unprofessional","ignored","waiter","waitress","manager"],
    "Wait Times & Speed":         ["wait","slow","minutes","hour","forever","long time","took too long","busy","rushed"],
    "Food Quality & Consistency": ["cold","bland","overcooked","undercooked","raw","stale","tasteless","dry","bad","terrible","awful"],
    "Cleanliness & Environment":  ["dirty","bathroom","filthy","unclean","roach","pest","gross","smell","sticky"],
    "Price & Value Perception":   ["expensive","overpriced","worth","cheap","value","price","cost","pricey","charged"],
    "Order Accuracy":             ["wrong","incorrect","missing","forgot","mistake","not what"],
    "Noise & Atmosphere":         ["loud","noisy","crowded","cramped","atmosphere","ambiance","music","parking"],
}

IMPROVEMENT_MAP = {
    "Service & Staff Attitude": (
        "Service Culture Enhancement",
        ["Weekly hospitality coaching sessions",
         "Manager accountability walks during peak hours",
         "Customer-facing staff empathy training",
         "Empower staff to resolve complaints on the spot"]
    ),
    "Wait Times & Speed": (
        "Speed & Throughput Optimization",
        ["Track and display kitchen ticket times",
         "Pre-batch high-demand items before rush hours",
         "Implement table-ready notification system",
         "Review table turn time and seating flow"]
    ),
    "Food Quality & Consistency": (
        "Food Quality Controls",
        ["Daily temperature audits at expo station",
         "Standardized plating & portioning procedures",
         "Blind taste-test QA each shift",
         "Refresh menu items with low satisfaction scores"]
    ),
    "Cleanliness & Environment": (
        "Cleanliness Initiative",
        ["Hourly restroom inspection checklist",
         "Visible sanitation station at entrance",
         "Deep-clean schedule posted publicly",
         "Assign cleanliness owner per shift"]
    ),
    "Price & Value Perception": (
        "Value Perception Strategy",
        ["Bundle high-margin combos with clear savings callout",
         "Train staff to highlight fresh/local ingredients",
         "Introduce a loyalty reward for repeat visits",
         "Audit menu pricing vs. local competitors"]
    ),
    "Order Accuracy": (
        "Order Accuracy Program",
        ["Read-back every order before submitting to kitchen",
         "Color-coded ticket system for dietary restrictions",
         "Track error rates weekly and reward accuracy",
         "Standardize modifier codes in POS system"]
    ),
    "Noise & Atmosphere": (
        "Atmosphere Improvement",
        ["Acoustic panels or soft furnishings to reduce noise",
         "Adjust background music volume by daypart",
         "Curate lighting for ambiance at dinner service",
         "Improve parking signage and accessibility"]
    ),
}

# ---------------------------------------------------
# SIDEBAR — file uploaders
# ---------------------------------------------------
st.sidebar.header("⚙️ Configuration")
st.sidebar.subheader("Upload Yelp Dataset")

business_file = st.sidebar.file_uploader("📁 Upload Business JSON", type=["json"])
review_file   = st.sidebar.file_uploader("📁 Upload Review JSON",   type=["json"])

sample_size = st.sidebar.slider(
    "Review Sample Size",
    min_value=1000, max_value=20000, value=5000, step=1000,
    help="Higher = more accurate but slower."
)

st.sidebar.markdown("---")
st.sidebar.caption("CIS 509 – Unstructured Data Analytics | Team 8")

if business_file is None or review_file is None:
    st.info("👈 Upload both **Business JSON** and **Review JSON** in the sidebar to begin.")
    st.stop()

# ---------------------------------------------------
# LOADERS
# ---------------------------------------------------
@st.cache_data(show_spinner="Loading business data…")
def load_businesses(data: bytes) -> pd.DataFrame:
    content = data.decode("utf-8").strip()
    if content.startswith("["):
        rows = json.loads(content)
    else:
        rows = [json.loads(l) for l in content.splitlines() if l.strip()]
    return pd.DataFrame(rows)

@st.cache_data(show_spinner="Loading & scoring reviews… (this may take a moment)")
def load_and_score_reviews(data: bytes, valid_ids: frozenset, n: int) -> pd.DataFrame:
    content = data.decode("utf-8").strip()
    if content.startswith("["):
        rows = json.loads(content)
    else:
        rows = [json.loads(l) for l in content.splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] <= "2018-12-31"]
    df = df[df["business_id"].isin(valid_ids)].copy()
    df = df.sample(min(n, len(df)), random_state=42)

    sia = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["text"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )
    df["sentiment_category"] = df["sentiment_score"].apply(
        lambda s: "Positive" if s >= 0.05 else ("Negative" if s <= -0.05 else "Neutral")
    )
    return df

# ---------------------------------------------------
# LOAD & FILTER BUSINESSES
# ---------------------------------------------------
biz_raw = load_businesses(business_file.read())

for col in ["categories","name","stars","business_id"]:
    if col not in biz_raw.columns:
        st.error(f"Business file missing column: `{col}`")
        st.stop()

biz_df = biz_raw[biz_raw["categories"].fillna("").str.contains("Restaurants", case=False)].copy()
name_counts = biz_df["name"].value_counts()
biz_df = biz_df[biz_df["name"].isin(name_counts[name_counts < 5].index)].copy()
biz_df = biz_df[(biz_df["stars"] <= 2.5) | (biz_df["stars"] >= 4.5)].copy()
biz_df["segment"] = biz_df["stars"].apply(
    lambda x: "At-Risk (≤2.5⭐)" if x <= 2.5 else "Excellent (≥4.5⭐)"
)

if biz_df.empty:
    st.error("No restaurants matched filter criteria.")
    st.stop()

# ---------------------------------------------------
# LOAD & SCORE REVIEWS
# ---------------------------------------------------
valid_ids  = frozenset(biz_df["business_id"])
review_raw = review_file.read()
rev_df     = load_and_score_reviews(review_raw, valid_ids, sample_size)

if rev_df.empty:
    st.error("No matching reviews found in the uploaded review file.")
    st.stop()

merged = rev_df.merge(
    biz_df[["business_id","name","stars","segment","city","state"]],
    on="business_id", how="left"
)

st.success(f"✅ **{biz_df.shape[0]:,}** independent restaurants | **{len(merged):,}** reviews loaded")
st.divider()

# =====================================================
# TAB LAYOUT
# =====================================================
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Overview", "🏢 Restaurant Diagnostic", "📋 Data Explorer"])

# =====================================================
# TAB 1 — PORTFOLIO OVERVIEW
# =====================================================
with tab1:
    st.header("Executive-Level Portfolio Overview")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Reviews",       f"{len(merged):,}")
    c2.metric("Unique Restaurants",  f"{merged['business_id'].nunique():,}")
    c3.metric("At-Risk Restaurants", f"{(biz_df['segment']=='At-Risk (≤2.5⭐)').sum():,}")
    c4.metric("Avg Sentiment",       f"{merged['sentiment_score'].mean():.3f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Review Count by Segment")
        seg_counts = merged["segment"].value_counts()
        colors = ["#e74c3c" if "At-Risk" in s else "#2ecc71" for s in seg_counts.index]
        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(seg_counts.index, seg_counts.values, color=colors, edgecolor="white", width=0.5)
        ax.set_ylabel("Reviews")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
        ax.set_title("Reviews per Segment")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown("### Sentiment Distribution by Segment")
        palette = {"At-Risk (≤2.5⭐)":"#e74c3c","Excellent (≥4.5⭐)":"#2ecc71"}
        fig, ax = plt.subplots(figsize=(5,3))
        sns.boxplot(data=merged, x="segment", y="sentiment_score", ax=ax, palette=palette)
        ax.set_xlabel(""); ax.set_ylabel("VADER Compound Score")
        ax.set_title("Sentiment Distribution")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    st.markdown("### Segment Sentiment Summary")
    summary = (
        merged.groupby("segment")["sentiment_score"]
        .agg(Mean="mean", Median="median", Std="std", Reviews="count")
        .round(3).reset_index()
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🏆 Top 10 Excellent Restaurants (by Avg Sentiment)")
    top10 = (
        merged[merged["segment"]=="Excellent (≥4.5⭐)"]
        .groupby(["name","city","state","stars"])["sentiment_score"]
        .agg(Avg_Sentiment="mean", Reviews="count")
        .reset_index()
        .query("Reviews >= 3")
        .sort_values("Avg_Sentiment", ascending=False)
        .head(10)
        .round(3)
    )
    top10.columns = ["Name","City","State","Stars","Avg Sentiment","Reviews"]
    st.dataframe(top10, use_container_width=True, hide_index=True)

    st.markdown("### ⚠️ Top 10 Most At-Risk Restaurants (by Avg Sentiment)")
    bottom10 = (
        merged[merged["segment"]=="At-Risk (≤2.5⭐)"]
        .groupby(["name","city","state","stars"])["sentiment_score"]
        .agg(Avg_Sentiment="mean", Reviews="count")
        .reset_index()
        .query("Reviews >= 2")
        .sort_values("Avg_Sentiment")
        .head(10)
        .round(3)
    )
    bottom10.columns = ["Name","City","State","Stars","Avg Sentiment","Reviews"]
    st.dataframe(bottom10, use_container_width=True, hide_index=True)

    st.markdown("### 🌍 Reviews by State")
    if "state" in merged.columns:
        state_counts = merged["state"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10,3))
        ax.bar(state_counts.index, state_counts.values, color="#3498db")
        ax.set_ylabel("Reviews"); ax.set_title("Top 15 States by Review Volume")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

# =====================================================
# TAB 2 — INDIVIDUAL DIAGNOSTIC
# =====================================================
with tab2:
    st.header("Individual Restaurant Diagnostic")

    # Build display label: Name + stars + city
    merged["display_label"] = (
        merged["name"].fillna("Unknown") + " — " +
        merged["city"].fillna("") + ", " +
        merged["state"].fillna("") + " (" +
        merged["stars"].astype(str) + "⭐)"
    )
    biz_df_labeled = biz_df.copy()
    biz_df_labeled["display_label"] = (
        biz_df_labeled["name"].fillna("Unknown") + " — " +
        biz_df_labeled.get("city", pd.Series("")).fillna("") + ", " +
        biz_df_labeled.get("state", pd.Series("")).fillna("") + " (" +
        biz_df_labeled["stars"].astype(str) + "⭐)"
    )

    col_filter, col_search = st.columns([1,2])
    with col_filter:
        seg_filter = st.selectbox("Filter by Segment", ["All", "At-Risk (≤2.5⭐)", "Excellent (≥4.5⭐)"])

    filtered_merged = merged if seg_filter == "All" else merged[merged["segment"] == seg_filter]
    available = sorted(filtered_merged["display_label"].dropna().unique())

    if not available:
        st.warning("No restaurants match the selected filter.")
        st.stop()

    selected_label = st.selectbox("Select a Restaurant:", available)

    if selected_label:
        # Get business_id from label
        match = merged[merged["display_label"] == selected_label]
        if match.empty:
            st.warning("Restaurant not found.")
            st.stop()

        selected_bid = match["business_id"].iloc[0]
        rest_reviews = merged[merged["business_id"] == selected_bid].copy()
        biz_info     = biz_df[biz_df["business_id"] == selected_bid].iloc[0]

        avg_sent      = rest_reviews["sentiment_score"].mean()
        segment_label = biz_info["segment"]
        star_rating   = biz_info["stars"]
        city_state    = f"{biz_info.get('city','')}, {biz_info.get('state','')}"

        # --- KPI Row ---
        st.subheader(f"📍 {biz_info['name']}  —  {city_state}")
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Segment",      segment_label)
        k2.metric("Star Rating",  f"{star_rating} ⭐")
        k3.metric("Avg Sentiment",f"{avg_sent:.3f}",
                  help="VADER: -1 (very negative) → +1 (very positive)")
        k4.metric("Reviews",      len(rest_reviews))

        if len(rest_reviews) < 5:
            st.warning("⚠️ Small sample — interpret cautiously.")

        st.divider()

        # --- Sentiment Trend ---
        st.markdown("### 📅 Sentiment Trend Over Time")
        trend = (
            rest_reviews.assign(ym=rest_reviews["date"].dt.to_period("M").astype(str))
            .groupby("ym")["sentiment_score"].mean().reset_index()
        )
        if len(trend) >= 2:
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(trend["ym"], trend["sentiment_score"], marker="o", color="#3498db", linewidth=2)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.fill_between(trend["ym"], trend["sentiment_score"], 0,
                            where=trend["sentiment_score"]>0, alpha=0.15, color="#2ecc71")
            ax.fill_between(trend["ym"], trend["sentiment_score"], 0,
                            where=trend["sentiment_score"]<0, alpha=0.15, color="#e74c3c")
            ax.set_ylabel("Avg Sentiment"); ax.set_title("Monthly Average Sentiment")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
        else:
            st.info("Not enough data points for a trend chart.")

        st.divider()

        # --- Issue Detection + Keywords ---
        combined_text = " ".join(rest_reviews["text"].astype(str)).lower()

        issue_scores = {}
        for bucket, kws in ISSUE_BUCKETS.items():
            cnt = sum(combined_text.count(kw) for kw in kws)
            if cnt > 0:
                issue_scores[bucket] = cnt
        detected_issues = sorted(issue_scores, key=issue_scores.get, reverse=True)
        if not detected_issues:
            detected_issues = ["Service & Operational Consistency"]

        words = re.findall(r'\b[a-z]{4,}\b', combined_text)
        words = [w for w in words if w not in STOP_WORDS]
        top_kw = Counter(words).most_common(12)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🚩 Operational Risk Signals")
            for i, issue in enumerate(detected_issues[:5], 1):
                freq = issue_scores.get(issue, 0)
                color = "🔴" if i <= 2 else "🟡"
                st.markdown(f"{color} **{i}. {issue}** — ~{freq} mentions")

        with col2:
            st.markdown("### 🔑 Top Review Keywords")
            if top_kw:
                kw_df = pd.DataFrame(top_kw, columns=["Word","Count"])
                fig, ax = plt.subplots(figsize=(5,4))
                colors = ["#e74c3c" if i < 3 else "#3498db" for i in range(len(kw_df))]
                ax.barh(kw_df["Word"][::-1], kw_df["Count"][::-1], color=colors[::-1])
                ax.set_xlabel("Frequency"); ax.set_title("Most Common Words in Reviews")
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

        st.divider()

        # --- Sentiment Breakdown ---
        st.markdown("### 📊 Customer Sentiment Breakdown")

        sent_counts = rest_reviews["sentiment_category"].value_counts()
        total = len(rest_reviews)
        pos_pct = round(100 * sent_counts.get("Positive",0)/total, 1)
        neu_pct = round(100 * sent_counts.get("Neutral", 0)/total, 1)
        neg_pct = round(100 * sent_counts.get("Negative",0)/total, 1)

        m1,m2,m3 = st.columns(3)
        m1.metric("🔴 Negative", f"{neg_pct}%")
        m2.metric("🟡 Neutral",  f"{neu_pct}%")
        m3.metric("🟢 Positive", f"{pos_pct}%")

        fig, ax = plt.subplots(figsize=(4,4))
        sizes  = [neg_pct, neu_pct, pos_pct]
        labels = ["Negative","Neutral","Positive"]
        colors = ["#e74c3c","#f39c12","#2ecc71"]
        non_zero = [(s,l,c) for s,l,c in zip(sizes,labels,colors) if s > 0]
        if non_zero:
            s,l,c = zip(*non_zero)
            ax.pie(s, labels=l, colors=c, autopct="%1.1f%%", startangle=140,
                   wedgeprops=dict(edgecolor="white"))
        ax.set_title("Sentiment Split")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        if neg_pct >= 30:
            st.error("⚠️ High proportion of negative reviews detected.")
        elif neg_pct >= 15:
            st.warning("Moderate level of negative customer experiences.")
        else:
            st.success("Customer sentiment is largely positive.")

        # --- Sample Reviews ---
        with st.expander("📝 Sample Negative Reviews"):
            neg = rest_reviews[rest_reviews["sentiment_category"]=="Negative"]["text"].head(3)
            if neg.empty:
                st.info("No negative reviews in this sample.")
            else:
                for rev in neg:
                    st.markdown(f"> _{str(rev)[:400]}{'…' if len(str(rev))>400 else ''}_")
                    st.markdown("---")

        with st.expander("📝 Sample Positive Reviews"):
            pos = rest_reviews[rest_reviews["sentiment_category"]=="Positive"]["text"].head(3)
            if pos.empty:
                st.info("No positive reviews in this sample.")
            else:
                for rev in pos:
                    st.markdown(f"> _{str(rev)[:400]}{'…' if len(str(rev))>400 else ''}_")
                    st.markdown("---")

        st.divider()

        # --- Improvement Plan ---
        st.markdown("## 📈 Customized Improvement Plan")

        with st.expander("🔧 Core Operational Stabilization (Always Applied)", expanded=True):
            st.markdown("""
- Audit end-to-end service flow from host stand to check delivery
- Reinforce hospitality standards with weekly team huddles
- Monitor peak-hour staffing levels and adjust schedule proactively
- Respond to all online reviews within 48 hours
- Track net promoter score (NPS) monthly
""")

        for issue in detected_issues[:4]:
            if issue in IMPROVEMENT_MAP:
                title, actions = IMPROVEMENT_MAP[issue]
                with st.expander(f"🎯 {title}"):
                    for action in actions:
                        st.markdown(f"- {action}")

        st.success("✅ Plan generated dynamically from Yelp review patterns.")

# =====================================================
# TAB 3 — DATA EXPLORER
# =====================================================
with tab3:
    st.header("📋 Data Explorer")

    st.markdown("### Filtered Business Dataset")
    biz_show = biz_df[["name","stars","segment","city","state","review_count"]].copy() if "city" in biz_df.columns else biz_df[["name","stars","segment"]].copy()
    st.dataframe(biz_show.sort_values("stars"), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Review Dataset Sample")
    rev_show_cols = [c for c in ["date","business_id","stars","sentiment_score","sentiment_category","text"] if c in merged.columns]
    st.dataframe(
        merged[rev_show_cols].sort_values("date", ascending=False).head(500),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    st.markdown("### Download Filtered Data")
    csv = merged.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Merged Reviews CSV",
        data=csv,
        file_name="restaurant_reviews_filtered.csv",
        mime="text/csv"
    )
