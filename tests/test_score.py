"""The two-level macro. Getting this wrong is the easiest way to report a wrong number.

These tests deliberately avoid `pytrec_eval`: they feed pre-computed per-topic scores so the
*averaging logic* is tested in isolation, which is where the risk actually lives.
"""

from __future__ import annotations

import pytest

import score as scoring


def domain(score: float, num_topics: int) -> dict[str, float]:
    return {"score": score, "num_topics": num_topics}


# ------------------------------------------- the mistake this module exists to prevent --
def test_macro_weights_every_domain_equally_regardless_of_size() -> None:
    """The core rule (CLAUDE.md §4/§6). History has 801 topics, IOTA ~10; both count once.

    Hand-computed: a domain at 0.9 with 1000 topics and one at 0.1 with 10 topics macro to
    0.5. A pooled mean would give ~0.892, which is the wrong answer.
    """
    per_domain = {"history": domain(0.9, 1000), "iota": domain(0.1, 10)}
    result = scoring.macro_over_domains(per_domain)
    assert result["macro"] == pytest.approx(0.5)
    assert result["num_domains"] == 2
    assert result["num_topics"] == 1010  # reported, but not used for weighting

    pooled = (0.9 * 1000 + 0.1 * 10) / 1010
    assert result["macro"] != pytest.approx(pooled), "this is the pooled-mean bug"


def test_reproduces_the_organizers_published_macro() -> None:
    """The 13 published per-domain 1a-train values must average to the published 0.0879.

    This is the same arithmetic that established the averaging level in the first place, run
    through our own code path rather than by hand.
    """
    from gate import PUBLISHED, PUBLISHED_MACRO

    per_domain = {d: domain(v["1a_train"], 10) for d, v in PUBLISHED.items()}
    result = scoring.macro_over_domains(per_domain)
    # Each published value is rounded to 4 dp before averaging, so allow that rounding.
    assert result["macro"] == pytest.approx(PUBLISHED_MACRO["1a_train"], abs=1e-4)
    assert result["num_domains"] == 13


def test_equal_sized_domains_make_macro_and_pooled_agree() -> None:
    """Sanity check: the two averagings only differ when domain sizes differ."""
    per_domain = {"a": domain(0.2, 50), "b": domain(0.4, 50)}
    assert scoring.macro_over_domains(per_domain)["macro"] == pytest.approx(0.3)


# ------------------------------------------------------------------ edge handling --
def test_domains_with_no_topics_are_skipped_not_counted_as_zero() -> None:
    """A domain we failed to run is a bug to fix, not a 0.0 to average in."""
    per_domain = {"a": domain(0.8, 10), "broken": domain(0.0, 0)}
    result = scoring.macro_over_domains(per_domain)
    assert result["macro"] == pytest.approx(0.8)
    assert result["num_domains"] == 1
    assert result["skipped_domains"] == ["broken"]


def test_empty_input_is_zero_not_a_crash() -> None:
    result = scoring.macro_over_domains({})
    assert result["macro"] == 0.0
    assert result["num_domains"] == 0


def test_a_genuine_zero_scoring_domain_still_counts() -> None:
    """0.0 with topics is a real result; 0.0 with no topics is a failure. Different things."""
    per_domain = {"a": domain(1.0, 5), "b": domain(0.0, 5)}
    result = scoring.macro_over_domains(per_domain)
    assert result["macro"] == pytest.approx(0.5)
    assert result["num_domains"] == 2


# ------------------------------------------------------------- pooled, for the paper --
def test_pooled_is_available_but_distinct() -> None:
    """Kept only to quantify the gap in the paper. Never our reported score."""
    nested = {
        "big": {f"t{i}": {"ndcg_cut_10": 0.9} for i in range(100)},
        "small": {"t0": {"ndcg_cut_10": 0.1}},
    }
    pooled = scoring.pooled_over_topics(nested)
    macro = scoring.macro_over_domains(
        {"big": domain(0.9, 100), "small": domain(0.1, 1)})["macro"]

    assert pooled == pytest.approx((0.9 * 100 + 0.1) / 101)
    assert macro == pytest.approx(0.5)
    assert pooled > macro, "large domains dominate the pooled mean — why RETECO rejects it"


def test_pooled_accepts_the_flat_shape_used_by_cv_and_bootstrap() -> None:
    """eval.bootstrap and eval.cv pass {domain: {topic: float}}; confusing the two shapes
    silently would be worse than accepting both."""
    flat = {"big": {f"t{i}": 0.9 for i in range(100)}, "small": {"t0": 0.1}}
    nested = {"big": {f"t{i}": {"ndcg_cut_10": 0.9} for i in range(100)},
              "small": {"t0": {"ndcg_cut_10": 0.1}}}
    assert scoring.pooled_over_topics(flat) == pytest.approx(
        scoring.pooled_over_topics(nested))


def test_pooled_on_empty_is_zero() -> None:
    assert scoring.pooled_over_topics({}) == 0.0


# ------------------------------------------------------------------------ render --
def test_render_shows_per_domain_and_marks_equal_weight() -> None:
    per_domain = {"iota": domain(0.2083, 3), "history": domain(0.0877, 240)}
    text = scoring.render(per_domain, scoring.macro_over_domains(per_domain))
    assert "iota" in text and "history" in text
    assert "MACRO" in text
    assert "equal weight" in text


def test_render_reports_skipped_domains() -> None:
    per_domain = {"a": domain(0.5, 4), "gone": domain(0.0, 0)}
    text = scoring.render(per_domain, scoring.macro_over_domains(per_domain))
    assert "skipped" in text and "gone" in text
