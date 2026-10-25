"""Streamlit app for the Scam Message Detection System.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html as html_lib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    MODEL_COMPARISON_CSV,
    THRESHOLD_JSON,
)
from src.predict import get_detector  # noqa: E402

st.set_page_config(
    page_title="Scam message detector",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Sample messages
# --------------------------------------------------------------------------- #
SCAM_SAMPLES: Dict[str, str] = {
    "UPI refund scam": (
        "URGENT: Your Rs.4,999 refund from Flipkart is pending. Claim now: "
        "flipkart-refund.in/claim Your UPI will be blocked in 24 hrs"
    ),
    "KYC / OTP phishing": (
        "Your KYC has expired. Update KYC immediately: sbi-kyc.top/update "
        "Share OTP 889912 with the agent to continue. Bank account will be closed."
    ),
    "Account blocked scare": (
        "WARNING! Your bank account XX4521 is BLOCKED due to suspicious txn. "
        "Verify now at onlinesbi-verify.click or your account will be frozen."
    ),
    "Fake job offer": (
        "Congratulations! You are selected for Data Entry Operator role. "
        "Earn Rs.25,000-40,000 per week sitting at home. Pay registration fee "
        "Rs.2,500 today: job-reg.xyz/signup. Limited slots!"
    ),
    "Lottery / prize": (
        "You have WON Rs.25,00,000 in KBC lucky draw!! Your number is selected. "
        "Claim prize now at kbc-winner.click. Send your Aadhaar + OTP to release funds."
    ),
    "Digital arrest threat": (
        "This is CBI Delhi. Your number is involved in cybercrime. Digital arrest "
        "declared. Do not contact anyone. Transfer Rs.1,50,000 for bail verification "
        "immediately or you will be arrested within 2 hours."
    ),
    "Courier / customs fraud": (
        "Customs department seized your courier with illegal cash. Pay Rs.5,000 "
        "customs duty at courier-clear.top/pay within 3 hrs or parcel will be destroyed."
    ),
}

LEGIT_SAMPLES: Dict[str, str] = {
    "Bank debit alert": (
        "Rs.2,500.00 debited from a/c XX4521 towards UPI to merchant@okhdfcbank "
        "on 14-10. Bal: Rs.18,203.55. Not you? Call 18002095566"
    ),
    "Genuine OTP": (
        "OTP 493821 for UPI txn of Rs.899 to swiggy@ybl. Do not share with anyone. "
        "Valid 10 mins. -HDFC Bank"
    ),
    "Salary credit": (
        "Rs.15,000 credited to your a/c XX7832 from EMP SALARY on 30-09. "
        "Avl Bal: Rs.62,410.00 -SBI"
    ),
    "Delivery update": "Your Amazon order has been delivered. Track it in the app",
    "Friend chat": "Hi, are we still meeting for coffee tomorrow at 5pm? Let me know",
    "Office message": "Meeting moved to 3pm. Please join via the usual link",
}

TIPS: Dict[str, str] = {
    "SCAM": (
        "Do not click the link or share any OTP. Banks, UPI apps and government "
        "agencies never ask for OTP/PIN over SMS or call. Verify independently by "
        "calling the official customer-care number printed on the back of your card "
        "or from the bank's official website — never from the message."
    ),
    "SAFE": (
        "Looks like a normal message. Still avoid clicking unknown links and never "
        "share OTPs or PINs with anyone, even if they claim to be from your bank."
    ),
}

VERDICT_COLORS = {"SCAM": "#C62828", "SAFE": "#1B7F3B"}


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model bundle...")
def load_detector():
    return get_detector()


@st.cache_data(show_spinner=False, ttl="1h")
def load_model_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"comparison": None, "threshold": None}
    try:
        import pandas as pd

        if MODEL_COMPARISON_CSV.exists():
            info["comparison"] = pd.read_csv(MODEL_COMPARISON_CSV)
    except Exception:
        pass
    try:
        if THRESHOLD_JSON.exists():
            info["threshold"] = json.loads(THRESHOLD_JSON.read_text(encoding="utf-8"))
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def highlight_message(message: str, reasons: List[Dict[str, Any]]) -> str:
    """Return HTML with suspicious words highlighted (red=SCAM, green=SAFE)."""
    flagged = {str(r["feature"]).lower(): float(r["weight"]) for r in reasons}
    tokens = re.findall(r"\w+|\W+", message)
    parts: List[str] = []
    for tok in tokens:
        key = tok.lower()
        if key in flagged:
            w = flagged[key]
            if w > 0:
                style = (
                    "background:#FDECEC;color:#8E1B1B;font-weight:600;"
                    "border-radius:4px;padding:1px 4px;box-decoration-break:clone;"
                    "-webkit-box-decoration-break:clone;"
                )
            else:
                style = (
                    "background:#E8F6ED;color:#0E5727;font-weight:600;"
                    "border-radius:4px;padding:1px 4px;box-decoration-break:clone;"
                    "-webkit-box-decoration-break:clone;"
                )
            parts.append(
                f'<span style="{style}" title="weight={w:+.3f}">'
                f"{html_lib.escape(tok)}</span>"
            )
        else:
            parts.append(html_lib.escape(tok))
    return "".join(parts)


def risk_gauge(prob: float, threshold: float):
    """Half-donut risk gauge (top semicircle) built with Altair.

    Angle 0 in Vega points up; the gauge sweeps from -pi/2 (left) to +pi/2 (right).
    """
    import math

    import altair as alt
    import pandas as pd

    START = -math.pi / 2
    END = math.pi / 2
    p = min(max(prob, 0.0), 1.0)

    bg = pd.DataFrame({"start": [START], "end": [END]})
    fg = pd.DataFrame({"start": [START], "end": [START + p * math.pi]})

    bg_arc = (
        alt.Chart(bg)
        .mark_arc(innerRadius=58, outerRadius=92, cornerRadius=8)
        .encode(theta=alt.Theta("start:Q"), theta2=alt.Theta2("end:Q"))
    )
    fg_arc = (
        alt.Chart(fg)
        .mark_arc(innerRadius=58, outerRadius=92, cornerRadius=8)
        .encode(
            theta=alt.Theta("start:Q"),
            theta2=alt.Theta2("end:Q"),
            color=alt.value(
                VERDICT_COLORS["SCAM"] if p >= threshold else VERDICT_COLORS["SAFE"]
            ),
        )
    )

    # Threshold tick: radial line at the tuned threshold angle
    cx, cy = 140.0, 95.0
    thr_angle = START + min(max(threshold, 0.0), 1.0) * math.pi
    tick = alt.Chart(
        pd.DataFrame(
            {
                "x": [cx + 52 * math.sin(thr_angle), cx + 100 * math.sin(thr_angle)],
                "y": [cy - 52 * math.cos(thr_angle), cy - 100 * math.cos(thr_angle)],
            }
        )
    ).mark_line(color="#555555", strokeWidth=2).encode(x="x:Q", y="y:Q")

    label = alt.Chart(
        pd.DataFrame({"y": [cy - 18], "text": [f"{p:.0%}"]})
    ).mark_text(fontSize=30, fontWeight=700).encode(
        y="y:Q", text="text:N", color=alt.value("#17171C")
    )

    caption = alt.Chart(
        pd.DataFrame({"y": [cy + 18], "text": ["scam risk"]})
    ).mark_text(fontSize=12).encode(y="y:Q", text="text:N", color=alt.value("#5B6470"))

    thr_label = alt.Chart(
        pd.DataFrame({"y": [8], "text": [f"threshold {threshold:.2f}"]})
    ).mark_text(fontSize=11).encode(y="y:Q", text="text:N", color=alt.value("#5B6470"))

    return (
        (bg_arc + fg_arc + tick + label + caption + thr_label)
        .properties(width=280, height=130)
        .configure_view(strokeOpacity=0)
    )


def reasons_chart(reasons: List[Dict[str, Any]]):
    """Horizontal bar chart of top contributing features."""
    import altair as alt
    import pandas as pd

    df = pd.DataFrame(
        [
            {"feature": str(r["feature"]), "weight": float(r["weight"])}
            for r in reasons
        ]
    )
    if df.empty:
        return None
    df["direction"] = df["weight"].apply(lambda w: "toward SCAM" if w > 0 else "toward SAFE")
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("weight:Q", title="weight"),
            y=alt.Y("feature:N", sort="-x", title=""),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["toward SCAM", "toward SAFE"],
                    range=["#C62828", "#1B7F3B"],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=["feature", alt.Tooltip("weight:Q", format="+.3f"), "direction"],
        )
        .properties(height=max(40 * len(df), 80))
    )
    return chart


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### :material/shield: Scam message detector")
    st.caption("SMS / WhatsApp classifier with per-message explanations")

    sample_tab, model_tab = st.tabs(["Samples", "Model info"])

    with sample_tab:
        st.markdown("#### Scam examples")
        scam_choice = st.selectbox(
            "Scam sample",
            options=list(SCAM_SAMPLES.keys()),
            key="scam_sample",
            label_visibility="collapsed",
            placeholder="Pick a scam sample",
        )
        st.button(
            "Load scam sample",
            icon=":material/warning:",
            key="load_scam",
            width="stretch",
        )
        if st.session_state.get("load_scam"):
            st.session_state["message_input"] = SCAM_SAMPLES[scam_choice]
            st.rerun()

        st.markdown("#### Legit examples")
        legit_choice = st.selectbox(
            "Legit sample",
            options=list(LEGIT_SAMPLES.keys()),
            key="legit_sample",
            label_visibility="collapsed",
            placeholder="Pick a legit sample",
        )
        st.button(
            "Load legit sample",
            icon=":material/check:",
            key="load_legit",
            width="stretch",
        )
        if st.session_state.get("load_legit"):
            st.session_state["message_input"] = LEGIT_SAMPLES[legit_choice]
            st.rerun()

    with model_tab:
        info = load_model_info()
        if info["threshold"]:
            t = info["threshold"]
            st.metric("Model", t.get("model", "?"))
            st.metric("Decision threshold", f"{t.get('threshold', 0.5):.3f}")
            st.metric(
                "Recall @ threshold (val)",
                f"{t.get('recall_at_threshold', 0):.1%}",
                help="Share of true scams caught at ~98-99% precision",
            )
        if info["comparison"] is not None:
            st.markdown("#### Test-set PR-AUC by model")
            st.bar_chart(
                info["comparison"].set_index("model")[["pr_auc"]],
                height=220,
            )
            with st.expander("Full comparison table"):
                st.dataframe(info["comparison"], width="stretch", hide_index=True)
        with st.expander("How it works"):
            st.markdown(
                "- Word TF-IDF (1-2 grams) + char TF-IDF (3-5 grams) + "
                "handcrafted features (URL count, urgency words, digit ratio, ...)\n"
                "- Best classical model selected by PR-AUC for the SCAM class\n"
                "- Decision threshold tuned on a validation split for ~98-99% "
                "precision\n"
                "- Coefficient-based and LIME explanations per message"
            )

    st.caption("Portfolio project · v1.0")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title(":material/search: Scam message analyzer")
st.caption(
    "Paste an SMS or WhatsApp message — the model scores its scam risk and "
    "highlights the exact words and links behind the verdict."
)


# --------------------------------------------------------------------------- #
# Input form
# --------------------------------------------------------------------------- #
if "message_input" not in st.session_state:
    st.session_state["message_input"] = SCAM_SAMPLES["UPI refund scam"]

with st.form("analyze_form", border=False):
    message = st.text_area(
        "Message",
        key="message_input",
        height=150,
        placeholder="Type or paste a message here...",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(
        "Analyze", type="primary", icon=":material/bolt:", width="stretch"
    )

if submitted and not message.strip():
    st.warning("Please enter a message first.")

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
if submitted and message.strip():
    with st.spinner("Analyzing message..."):
        det = load_detector()
        result = det.predict(message)

    verdict = result["label_str"]
    prob = float(result["probability"])
    thr = float(result["threshold"])
    is_scam = verdict == "SCAM"

    # --- Verdict cards ------------------------------------------------------
    vcol, gcol, mcol = st.columns([1.1, 1.4, 1.3], vertical_alignment="center")

    with vcol:
        st.html(
            f'<div style="text-align:center;padding:22px 12px;border-radius:14px;'
            f"background:{VERDICT_COLORS[verdict]};color:white;\">"
            f'<div style="font-size:0.8rem;letter-spacing:0.12em;opacity:0.85;">'
            f"VERDICT</div>"
            f'<div style="font-size:2.2rem;font-weight:800;line-height:1.2;">'
            f"{verdict}</div>"
            f'<div style="font-size:0.85rem;opacity:0.9;">'
            f"threshold {thr:.2f}</div></div>"
        )

    with gcol:
        st.altair_chart(risk_gauge(prob, thr), key="risk_gauge")

    with mcol:
        st.progress(min(max(prob, 0.0), 1.0), text=f"**Scam probability {prob:.1%}**")
        st.html("<div style='height:10px'></div>")
        c1, c2 = st.columns(2)
        c1.metric(
            "Confidence",
            f"{max(prob, 1 - prob):.1%}",
            delta="high" if max(prob, 1 - prob) > 0.9 else "moderate",
            delta_color="off",
        )
        c2.metric(
            "Verdict margin",
            f"{abs(prob - thr):.2f}",
            help="Distance from the decision threshold",
        )

    st.html("<div style='height:14px'></div>")

    # --- Tabs for details ---------------------------------------------------
    detail_tabs = st.tabs(
        [
            ":material/highlighter: Signals in message",
            ":material/lightbulb: Why flagged",
            ":material/link: Links",
            ":material/tips_and_updates: What to do",
        ]
    )

    # Signals
    with detail_tabs[0]:
        st.html(
            '<div style="line-height:2;font-size:1.05rem;padding:14px 16px;'
            'border:1px solid #E2E2EA;border-radius:10px;background:#FAFAFE;">'
            + highlight_message(message, result["reasons"])
            + "</div>"
        )
        with st.container(horizontal=True, horizontal_alignment="center"):
            st.html(
                '<span style="background:#FDECEC;color:#8E1B1B;border-radius:4px;'
                'padding:2px 8px;font-size:0.8rem;">pushes toward SCAM</span>'
            )
            st.html(
                '<span style="background:#E8F6ED;color:#0E5727;border-radius:4px;'
                'padding:2px 8px;font-size:0.8rem;">pushes toward SAFE</span>'
            )

    # Why flagged
    with detail_tabs[1]:
        reasons = result["reasons"]
        if reasons:
            cgraph, ctable = st.columns([3, 2], vertical_alignment="top")
            with cgraph:
                chart = reasons_chart(reasons)
                if chart is not None:
                    st.altair_chart(chart, key="reasons_chart")
            with ctable:
                st.dataframe(
                    [
                        {
                            "feature": str(r["feature"]),
                            "weight": round(float(r["weight"]), 4),
                            "signal": "SCAM" if r["weight"] > 0 else "SAFE",
                        }
                        for r in reasons
                    ],
                    width="stretch",
                    hide_index=True,
                )
            st.caption(
                "Weights come from a reference logistic-regression model "
                "(coefficient × TF-IDF weight for this message)."
            )
        else:
            st.info("No strong contributing features found.")

    # Links
    with detail_tabs[2]:
        urls = result["urls"]
        if urls:
            suspicious_urls = [u for u in urls if u["shortener"] or u["suspicious_tld"]]
            for u in urls:
                flags = []
                if u["shortener"]:
                    flags.append("URL shortener")
                if u["suspicious_tld"]:
                    flags.append("Suspicious TLD")
                badge = ""
                if flags:
                    badge = (
                        ' <span style="background:#C62828;color:white;'
                        'border-radius:4px;padding:2px 8px;font-size:0.75rem;'
                        'margin-left:8px;">'
                        + ", ".join(html_lib.escape(f) for f in flags)
                        + "</span>"
                    )
                border = "#C62828" if flags else "#1B7F3B"
                st.html(
                    '<div style="padding:10px 14px;border-left:4px solid '
                    + border
                    + ';background:#FAFAFE;border-radius:6px;margin:6px 0;">'
                    + f'<code style="font-size:0.95rem;">'
                    + html_lib.escape(u["url"])
                    + f"</code>{badge}</div>"
                )
            if suspicious_urls:
                st.warning(
                    f"{len(suspicious_urls)} of {len(urls)} link(s) look suspicious "
                    "(shortener or unusual TLD). Do not open them.",
                    icon=":material/warning:",
                )
        else:
            st.caption("No URLs detected in this message.")

    # What to do
    with detail_tabs[3]:
        if verdict == "SCAM":
            st.error(TIPS["SCAM"], icon=":material/error:")
        else:
            st.success(TIPS["SAFE"], icon=":material/check_circle:")

    # --- Advanced details ---------------------------------------------------
    with st.expander(":material/expand: Advanced details"):
        adv1, adv2 = st.columns(2)
        with adv1:
            st.markdown("**Handcrafted features**")
            st.json(
                {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in result["handcrafted"].items()
                }
            )
        with adv2:
            st.markdown("**LIME explanation**")
            st.caption("Model-agnostic local explanation (takes a few seconds).")
            if st.button("Run LIME", icon=":material/science:", key="run_lime"):
                with st.spinner("Computing LIME explanation..."):
                    try:
                        from src.explain import lime_explain_single

                        lime_res = lime_explain_single(message, num_features=10)
                        items = lime_res.get("explanation", [])
                        if items:
                            st.dataframe(items, width="stretch", hide_index=True)
                        else:
                            st.info(lime_res.get("error", "No explanation produced."))
                    except Exception as e:
                        st.warning(f"LIME failed: {e}")
