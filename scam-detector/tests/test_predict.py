"""Unit tests for src.preprocess and src.predict."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import (  # noqa: E402
    handcrafted_features,
    replace_entities,
)

SCAM_MESSAGES = [
    "URGENT: Your KYC has expired. Update now at bit.ly/kyc-scam to avoid account block!",
    "You have won Rs.25,00,000 in the KBC lottery! Claim at www.kbc-winner.xyz now",
    "Rs.4,999 refund pending on your UPI. Click paytm-refund.top/upi within 2 hrs",
    "Your bank account will be BLOCKED immediately. Verify OTP 889912 at sbi-kyc.click",
    "Customs dept seized your courier with cash. Pay Rs.5,000 fine at customs-pay.click",
]

LEGIT_MESSAGES = [
    "OTP 493821 for UPI txn of Rs.899 to swiggy@ybl. Do not share with anyone. Valid 10 mins. -HDFC Bank",
    "Rs.15,000 credited to your a/c XX7832 from EMP SALARY on 30-09. Avl Bal: Rs.62,410.00 -SBI",
    "Hi, are we still meeting for coffee tomorrow at 5pm? Let me know",
    "Your Amazon order has been delivered. Track it in the app",
    "Meeting moved to 3pm. Please join via the usual link",
]


@pytest.mark.parametrize("msg", SCAM_MESSAGES + LEGIT_MESSAGES)
def test_preprocess_keeps_placeholders(msg: str) -> None:
    import re

    cleaned = replace_entities(msg)
    # Should be lowercase
    assert cleaned == cleaned.lower()
    # All raw digits must be replaced by placeholder tokens (no digits remain)
    assert re.search(r"\d", cleaned) is None
    # Placeholders must be preserved (not deleted) when source had digits/urls/phones
    had_entities = bool(re.search(r"\d|https?://|www\.", msg))
    has_placeholder = any(t in cleaned for t in ("<num>", "<phone>", "<url>"))
    if had_entities:
        assert has_placeholder
    # entities are never deleted: content-bearing token always present for entity messages
    if not had_entities:
        assert len(cleaned) > 0


def test_preprocess_replaces_url() -> None:
    cleaned = replace_entities("visit https://example.com/offer now")
    assert "<url>" in cleaned
    assert "https://" not in cleaned


def test_preprocess_replaces_phone() -> None:
    cleaned = replace_entities("call me at +91 98765 43210")
    assert "<phone>" in cleaned
    assert "98765" not in cleaned


def test_handcrafted_features_keys() -> None:
    feats = handcrafted_features("URGENT!!! Win $1000 now at bit.ly/win.xyz")
    expected = {
        "msg_len",
        "word_count",
        "url_count",
        "digit_ratio",
        "upper_ratio",
        "exclam_count",
        "urgency_count",
        "currency_symbol",
        "has_shortener",
        "suspicious_tld",
        "avg_word_len",
        "special_char_ratio",
        "has_phone",
        "has_url",
    }
    assert expected.issubset(feats.keys())


def test_handcrafted_detects_shortener_and_tld() -> None:
    feats = handcrafted_features("check bit.ly/free-prize")
    assert feats["has_shortener"] == 1
    assert feats["url_count"] == 1
    feats2 = handcrafted_features("claim at www.kbc-winner.xyz/claim")
    assert feats2["suspicious_tld"] == 1
    assert feats2["has_shortener"] == 0


def test_handcrafted_currency_symbol() -> None:
    feats = handcrafted_features("You won Rs.5000")  # Rs. not symbol
    feats2 = handcrafted_features("You won $5000")
    feats3 = handcrafted_features("You won ₹5000")
    feats4 = handcrafted_features("You won £5000")
    assert feats2["currency_symbol"] == 1
    assert feats3["currency_symbol"] == 1
    assert feats4["currency_symbol"] == 1


# --------------------------------------------------------------------------- #
# predict tests
# --------------------------------------------------------------------------- #
def _detector():
    from src.predict import get_detector

    return get_detector()


def test_predict_returns_expected_keys() -> None:
    res = _detector().predict(SCAM_MESSAGES[0])
    assert set(res.keys()) >= {"label", "label_str", "probability", "reasons", "urls"}
    assert res["label"] in (0, 1)
    assert 0.0 <= res["probability"] <= 1.0


@pytest.mark.parametrize("msg", SCAM_MESSAGES)
def test_scam_messages_flagged(msg: str) -> None:
    res = _detector().predict(msg)
    assert res["label_str"] == "SCAM", f"Expected SCAM for: {msg} (p={res['probability']})"


@pytest.mark.parametrize("msg", LEGIT_MESSAGES)
def test_legit_messages_pass(msg: str) -> None:
    res = _detector().predict(msg)
    assert res["label_str"] == "SAFE", f"Expected SAFE for: {msg} (p={res['probability']})"


def test_predict_probability_monotonic_with_scam_keyword() -> None:
    det = _detector()
    p_scam = det.predict("URGENT: your account blocked, verify otp immediately at xyz.click")["probability"]
    p_legit = det.predict("ok see you tomorrow")["probability"]
    assert p_scam > p_legit
