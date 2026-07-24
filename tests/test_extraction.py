"""Extraction tests.

Focused on de-hyphenation, which resolves a genuine ambiguity: PDF line
wrapping and real compound words both appear as ``word-\\nword`` in extracted
text, and the syntax cannot tell them apart. Getting it wrong silently breaks
lexical retrieval for exactly the multi-word technical terms BM25 exists to
catch.
"""

from __future__ import annotations

from seclit.ingest.extract import (
    _dehyphenate,
    collect_hyphenated,
    extraction_quality,
    is_compound,
    is_garbled,
)


class TestCollectHyphenated:
    def test_learns_inline_compounds(self):
        vocab = collect_hyphenated("A cluster-based scheme with zero-trust checks.")

        assert "cluster-based" in vocab
        assert "zero-trust" in vocab

    def test_is_case_insensitive(self):
        assert "cross-site" in collect_hyphenated("A Cross-Site attack.")


class TestIsCompound:
    def test_document_usage_is_the_primary_signal(self):
        vocab = collect_hyphenated("We propose a cluster-based design.")

        assert is_compound("cluster-based", vocab)

    def test_wrapped_words_are_not_compounds(self):
        vocab = collect_hyphenated("We propose a cluster-based design.")

        assert not is_compound("vulner-ability", vocab)
        assert not is_compound("inser-tion", vocab)
        assert not is_compound("net-work", vocab)

    def test_head_of_a_longer_compound_is_recognised(self):
        """A break inside 'denial-of-service' at 'denial-|of' must still keep
        the hyphen, even though 'denial-of' alone is never written."""
        vocab = collect_hyphenated("We study denial-of-service attacks.")

        assert is_compound("denial-of", vocab)

    def test_common_compounds_survive_without_document_evidence(self):
        """Some compounds appear in a paper *only* at a line break, leaving the
        document with nothing to learn from."""
        assert is_compound("well-known", set())
        assert is_compound("state-of", set())
        assert is_compound("zero-day", set())

    def test_unknown_pairs_default_to_joining(self):
        """The default must stay 'join' — wrapped words vastly outnumber
        compounds, measured at roughly 90/10 on this corpus."""
        assert not is_compound("qwer-tyui", set())


class TestDehyphenate:
    def test_joins_a_wrapped_word(self):
        assert _dehyphenate("mitigate vulner-\nability now", set()) == "mitigate vulnerability now"

    def test_preserves_a_compound_the_document_uses(self):
        vocab = collect_hyphenated("a cluster-based scheme")

        assert "cluster-based" in _dehyphenate("the cluster-\nbased approach", vocab)

    def test_preserves_a_common_compound_with_no_document_evidence(self):
        assert "denial-of" in _dehyphenate("a denial-\nof-service attack", set())
        assert "well-known" in _dehyphenate("a well-\nknown result", set())

    def test_handles_mixed_cases_in_one_pass(self):
        vocab = collect_hyphenated("zero-trust designs")
        out = _dehyphenate("the zero-\ntrust model reduces vulner-\nability", vocab)

        assert "zero-trust" in out
        assert "vulnerability" in out


class TestQualityGate:
    def test_healthy_prose_passes(self):
        text = " ".join(
            "The intrusion detection system evaluates network traffic." for _ in range(40)
        )

        assert not is_garbled(text)

    def test_broken_font_encoding_is_rejected(self):
        """Characters dropped by a broken encoding: 'Injector' -> 'Inje tor'."""
        text = " ".join("The inje tor mo dule is ompli ated to b uild" for _ in range(60))

        assert is_garbled(text)

    def test_short_words_alone_do_not_trigger_rejection(self):
        """Both signals must fire. Legitimately short-worded text drags mean
        word length down while the fragment ratio stays near zero."""
        text = " ".join("Word." for _ in range(400))
        ratio, mean_len = extraction_quality(text)

        assert ratio < 0.15
        assert not is_garbled(text)

    def test_too_little_text_is_not_judged(self):
        assert not is_garbled("Only a few words here.")
