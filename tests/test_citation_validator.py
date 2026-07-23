"""Citation validator tests.

The property under test: a marker the model invented must never survive into
the answer shown to the user. That is the whole anti-hallucination guarantee,
so it is tested against the messy formatting real local models produce, not
just the ideal case.
"""

from __future__ import annotations

from seclit.generate.cite import (
    CitationReport,
    extract_markers,
    validate_citations,
)

ALLOWED = {"c1", "c2", "c3"}


class TestExtractMarkers:
    def test_canonical_form(self):
        assert extract_markers("Attackers pivot laterally [^c1].") == ["c1"]

    def test_accepts_formatting_variants_models_actually_emit(self):
        # Local models are inconsistent formatters; identity is what matters.
        assert extract_markers("a [c1] b [^c2] c (c3)") == ["c1", "c2", "c3"]

    def test_multiple_ids_in_one_group(self):
        assert extract_markers("Claim [^c1, c2].") == ["c1", "c2"]
        assert extract_markers("Claim [^c1; ^c3].") == ["c1", "c3"]

    def test_adjacent_groups(self):
        assert extract_markers("Claim [^c1][^c2].") == ["c1", "c2"]

    def test_repeats_are_preserved(self):
        # Validity is measured over citations made, not distinct sources.
        assert extract_markers("A [^c1]. B [^c1].") == ["c1", "c1"]

    def test_bare_caret_digits_are_accepted(self):
        """Regression: gemma3:12b emits [^7] without the 'c'. Rejecting that
        form silently zeroed the grounding metric on real answers."""
        assert extract_markers("Segmentation limits movement [^7].") == ["c7"]
        assert extract_markers("Both agree [^7, 9].") == ["c7", "c9"]

    def test_bare_digits_require_a_caret(self):
        """Without the caret, '(2020)' would parse as marker c2020 and a year
        citation would be silently mangled."""
        assert extract_markers("Published (2020) by the team.") == []
        assert extract_markers("See reference [1] for detail.") == []
        assert extract_markers("Table (7) shows results.") == []

    def test_ignores_unrelated_brackets(self):
        assert extract_markers("See [1] and (Smith 2020) and [table 2].") == []

    def test_empty_text(self):
        assert extract_markers("") == []


class TestValidation:
    def test_valid_markers_pass_through(self):
        report = validate_citations("Firewalls filter traffic [^c1].", ALLOWED)

        assert report.valid_markers == ["c1"]
        assert report.invalid_markers == []
        assert report.validity_rate == 1.0
        assert "[^c1]" in report.text

    def test_fabricated_marker_is_stripped(self):
        """The core guarantee: a citation to a chunk that was never retrieved
        must not reach the user."""
        report = validate_citations("Claim about zero trust [^c9].", ALLOWED)

        assert report.invalid_markers == ["c9"]
        assert "c9" not in report.text
        assert report.validity_rate == 0.0

    def test_mixed_group_keeps_only_valid_ids(self):
        report = validate_citations("Both sources agree [^c1, c7].", ALLOWED)

        assert report.valid_markers == ["c1"]
        assert report.invalid_markers == ["c7"]
        assert "c1" in report.text
        assert "c7" not in report.text

    def test_off_by_one_beyond_retrieved_set_is_rejected(self):
        """A frequent real failure: the model cites c4 when only c1-c3 exist."""
        report = validate_citations("Detection improves [^c4].", ALLOWED)

        assert report.invalid_markers == ["c4"]
        assert report.validity_rate == 0.0

    def test_validity_rate_is_proportional(self):
        report = validate_citations("A [^c1]. B [^c2]. C [^c8]. D [^c9].", ALLOWED)

        assert len(report.valid_markers) == 2
        assert len(report.invalid_markers) == 2
        assert report.validity_rate == 0.5

    def test_strip_invalid_can_be_disabled_for_auditing(self):
        report = validate_citations("Claim [^c9].", ALLOWED, strip_invalid=False)

        assert report.invalid_markers == ["c9"]
        assert "c9" in report.text  # preserved for inspection

    def test_no_markers_yields_vacuous_validity(self):
        report = validate_citations("A sentence with no citation at all.", ALLOWED)

        assert report.total_markers == 0
        assert report.validity_rate == 1.0

    def test_punctuation_is_tidied_after_stripping(self):
        report = validate_citations("Traffic is filtered [^c9] .", ALLOWED)

        assert "  " not in report.text
        assert " ." not in report.text


class TestGrounding:
    def test_uncited_claim_is_flagged(self):
        text = "Firewalls filter traffic [^c1]. Attackers always use zero-day exploits."
        report = validate_citations(text, ALLOWED)

        assert len(report.uncited_claims) == 1
        assert "zero-day" in report.uncited_claims[0]
        assert report.grounded_rate == 0.5

    def test_fully_cited_answer_is_clean(self):
        text = "Firewalls filter traffic [^c1]. Segmentation limits movement [^c2]."
        report = validate_citations(text, ALLOWED)

        assert report.uncited_claims == []
        assert report.grounded_rate == 1.0
        assert report.is_clean

    def test_refusal_language_is_not_counted_as_an_uncited_claim(self):
        report = validate_citations(
            "I could not find information about this topic in the provided sources.",
            ALLOWED,
        )

        assert report.uncited_claims == []
        assert report.grounded_rate == 1.0

    def test_short_fragments_and_headings_are_not_claims(self):
        report = validate_citations("## Findings\nOK.\nKey results:", ALLOWED)

        assert report.uncited_claims == []

    def test_stripping_an_invalid_marker_exposes_the_uncited_claim(self):
        """A fabricated citation must not launder an ungrounded claim: once the
        marker is stripped, the sentence should register as uncited."""
        report = validate_citations(
            "Ransomware encrypts files within seconds of execution [^c9].", ALLOWED
        )

        assert report.invalid_markers == ["c9"]
        assert len(report.uncited_claims) == 1
        assert not report.is_clean


def test_report_defaults_are_safe():
    report = CitationReport(text="")
    assert report.validity_rate == 1.0
    assert report.grounded_rate == 1.0
    assert report.is_clean


class TestPromptLeakage:
    """Regression guard for a real failure found during browser testing.

    The prompt's formatting example originally used realistic prose including an
    invented "40% reduction in lateral movement" statistic. gemma3:12b copied it
    almost verbatim into a real answer, citing valid markers — so the validator
    scored it 100% valid and 100% grounded while the headline claim came from
    the prompt rather than the corpus. A plausible example is an instruction to
    plagiarise it.
    """

    def test_format_example_contains_no_reusable_claims(self):
        from seclit.generate.prompt import FORMAT_EXAMPLE

        # No fabricated statistics for a model to lift.
        assert "40%" not in FORMAT_EXAMPLE
        assert "%" not in FORMAT_EXAMPLE.replace("100%", "")

        # Placeholders, not prose that could pass as a finding.
        assert "<" in FORMAT_EXAMPLE and ">" in FORMAT_EXAMPLE

    def test_format_example_still_demonstrates_the_marker_syntax(self):
        """The example must stay useful — it exists to fix format adherence."""
        from seclit.generate.cite import extract_markers
        from seclit.generate.prompt import FORMAT_EXAMPLE

        markers = extract_markers(FORMAT_EXAMPLE)
        assert markers, "example must show at least one marker"
        assert len(set(markers)) >= 2, "example should show a multi-source citation"
