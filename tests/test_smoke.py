"""
Smoke tests for genomic-research.

Generates synthetic FASTA data, runs prepare + train with a short time budget.
"""

import os
import sys
import tempfile
import subprocess
import json
from pathlib import Path

import pytest


def generate_synthetic_fasta(path, n_sequences=50, min_len=100, max_len=500):
    """Generate a synthetic FASTA file with random DNA sequences."""
    import random
    random.seed(42)
    bases = "ATCG"
    with open(path, "w") as f:
        for i in range(n_sequences):
            length = random.randint(min_len, max_len)
            seq = "".join(random.choice(bases) for _ in range(length))
            f.write(f">seq_{i} synthetic sequence {i}\n")
            # Write in lines of 80 chars
            for j in range(0, len(seq), 80):
                f.write(seq[j:j + 80] + "\n")
    return path


@pytest.fixture
def synthetic_fasta(tmp_path):
    """Create a temporary synthetic FASTA file."""
    fasta_path = tmp_path / "synthetic.fasta"
    generate_synthetic_fasta(str(fasta_path))
    return str(fasta_path)


@pytest.fixture
def synthetic_csv(tmp_path):
    """Create a temporary synthetic CSV file with sequences and labels."""
    import random
    random.seed(42)
    csv_path = tmp_path / "synthetic.csv"
    bases = "ATCG"
    labels = ["virus_a", "virus_b", "virus_c"]
    with open(csv_path, "w") as f:
        f.write("id,sequence,species\n")
        for i in range(60):
            length = random.randint(50, 200)
            seq = "".join(random.choice(bases) for _ in range(length))
            label = labels[i % 3]
            f.write(f"seq_{i},{seq},{label}\n")
    return str(csv_path)


class TestTokenizers:
    """Test tokenizer implementations."""

    def test_char_tokenizer(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from prepare import CharTokenizer
        tok = CharTokenizer()
        assert tok.vocab_size == 10
        ids = tok.encode("ATCGN")
        assert len(ids) == 5
        assert all(i >= 5 for i in ids)
        decoded = tok.decode(ids)
        assert decoded == "ATCGN"

    def test_kmer_tokenizer(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from prepare import KmerTokenizer
        tok = KmerTokenizer(k=3)
        assert tok.vocab_size == 5 + 64  # 4^3 + 5 special
        ids = tok.encode("ATCGATCG")
        assert len(ids) == 6  # 8 - 3 + 1
        decoded = tok.decode(ids)
        assert decoded == "ATCGATCG"

    def test_char_tokenizer_roundtrip(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from prepare import CharTokenizer
        tok = CharTokenizer()
        seq = "AATTCCGGNN"
        assert tok.decode(tok.encode(seq)) == seq


class TestSequenceLoading:
    """Test sequence loading from files."""

    def test_load_fasta(self, synthetic_fasta):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from prepare import load_sequences
        seqs = load_sequences(synthetic_fasta)
        assert len(seqs) == 50
        for sid, seq in seqs:
            assert len(seq) > 0
            assert all(c in "ATCGN" for c in seq)

    def test_load_csv(self, synthetic_csv):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from prepare import load_sequences
        seqs = load_sequences(synthetic_csv, seq_col="sequence", id_col="id")
        assert len(seqs) == 60


class TestPrepareData:
    """Test data preparation pipeline."""

    def test_pretrain_prepare(self, synthetic_fasta):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from prepare import prepare_data, CACHE_DIR, CONFIG_PATH
        import shutil

        # Clean cache
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)

        config = prepare_data(
            seq_path=synthetic_fasta,
            task_type="pretrain",
            tokenizer_type="char",
            max_length=128,
        )

        assert config["task_type"] == "pretrain"
        assert config["vocab_size"] == 10
        assert config["n_train"] > 0
        assert config["n_val"] > 0
        assert os.path.exists(os.path.join(CACHE_DIR, "train_tokens.pt"))
        assert os.path.exists(os.path.join(CACHE_DIR, "val_tokens.pt"))
        assert os.path.exists(CONFIG_PATH)

        # Clean up
        shutil.rmtree(CACHE_DIR)


class TestEndToEnd:
    """End-to-end smoke test: prepare + train."""

    def test_pretrain_e2e(self, synthetic_fasta):
        """Run a full pretrain cycle with minimal time budget."""
        templates_dir = str(Path(__file__).parent.parent / "genomic_research" / "templates")

        # Run prepare.py
        result = subprocess.run(
            [sys.executable, os.path.join(templates_dir, "prepare.py"),
             "--fasta", synthetic_fasta, "--task", "pretrain",
             "--tokenizer", "char", "--max-length", "64"],
            capture_output=True, text=True, cwd=templates_dir,
        )
        assert result.returncode == 0, f"prepare.py failed:\n{result.stderr}"

        # Run train.py with short budget
        env = os.environ.copy()
        env["GENOMIC_TIME_BUDGET"] = "10"
        result = subprocess.run(
            [sys.executable, os.path.join(templates_dir, "train.py")],
            capture_output=True, text=True, cwd=templates_dir,
            env=env, timeout=120,
        )
        assert result.returncode == 0, f"train.py failed:\n{result.stderr}"
        assert "val_score:" in result.stdout

        # Check reports
        reports_dir = os.path.join(templates_dir, "reports")
        assert os.path.exists(os.path.join(reports_dir, "metrics.json"))

        with open(os.path.join(reports_dir, "metrics.json")) as f:
            metrics = json.load(f)
        assert "val_score" in metrics
        assert "val_perplexity" in metrics

        # Clean up
        import shutil
        shutil.rmtree(reports_dir, ignore_errors=True)
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "genomic-research")
        shutil.rmtree(cache_dir, ignore_errors=True)
