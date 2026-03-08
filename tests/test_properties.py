"""
T27: Property-based tests for tokenizers using hypothesis.

Properties tested:
- decode(encode(seq)) == seq for all valid sequences
- len(encode(seq)) > 0 for non-empty sequences
- encoded values are within valid range
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))


# Strategy: generate valid DNA sequences (A, T, C, G, N)
dna_sequences = st.text(alphabet="ATCGN", min_size=1, max_size=200)


class TestCharTokenizerProperties:
    """Property-based tests for CharTokenizer."""

    @given(seq=dna_sequences)
    @settings(max_examples=100, deadline=None)
    def test_roundtrip(self, seq):
        from prepare import CharTokenizer
        tok = CharTokenizer()
        assert tok.decode(tok.encode(seq)) == seq

    @given(seq=dna_sequences)
    @settings(max_examples=50, deadline=None)
    def test_encode_nonempty(self, seq):
        from prepare import CharTokenizer
        tok = CharTokenizer()
        ids = tok.encode(seq)
        assert len(ids) > 0

    @given(seq=dna_sequences)
    @settings(max_examples=50, deadline=None)
    def test_encode_values_in_range(self, seq):
        from prepare import CharTokenizer
        tok = CharTokenizer()
        ids = tok.encode(seq)
        for i in ids:
            assert 0 <= i < tok.vocab_size


class TestKmerTokenizerProperties:
    """Property-based tests for KmerTokenizer."""

    @given(seq=st.text(alphabet="ATCG", min_size=6, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_roundtrip_k3(self, seq):
        assume(len(seq) >= 3)
        from prepare import KmerTokenizer
        tok = KmerTokenizer(k=3)
        assert tok.decode(tok.encode(seq)) == seq

    @given(seq=st.text(alphabet="ATCG", min_size=6, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_roundtrip_k6(self, seq):
        assume(len(seq) >= 6)
        from prepare import KmerTokenizer
        tok = KmerTokenizer(k=6)
        assert tok.decode(tok.encode(seq)) == seq

    @given(seq=st.text(alphabet="ATCG", min_size=3, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_encode_length_k3(self, seq):
        assume(len(seq) >= 3)
        from prepare import KmerTokenizer
        tok = KmerTokenizer(k=3)
        ids = tok.encode(seq)
        expected_len = len(seq) - 3 + 1
        assert len(ids) == expected_len

    @given(seq=st.text(alphabet="ATCG", min_size=3, max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_encode_values_in_range_k3(self, seq):
        assume(len(seq) >= 3)
        from prepare import KmerTokenizer
        tok = KmerTokenizer(k=3)
        ids = tok.encode(seq)
        for i in ids:
            assert 0 <= i < tok.vocab_size
