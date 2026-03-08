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


class TestDataAugmentation:
    """Test data augmentation utilities."""

    def test_reverse_complement(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        import torch

        # Simulate: A=5, T=6, C=7, G=8
        # RC of ATCG should be CGAT
        tokens = torch.tensor([[5, 6, 7, 8, 0]], dtype=torch.long)  # ATCG + PAD
        mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)

        from train import reverse_complement_tokens
        rc = reverse_complement_tokens(tokens, mask)
        # ATCG -> complement: TAGC -> reverse: CGAT -> tokens: 7,8,5,6
        assert rc[0, 0].item() == 7  # C
        assert rc[0, 1].item() == 8  # G
        assert rc[0, 2].item() == 5  # A
        assert rc[0, 3].item() == 6  # T
        assert rc[0, 4].item() == 0  # PAD unchanged

    def test_span_mask_tokens(self):
        """T20: Test span_mask_tokens produces correct mask ratios."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        import torch
        from train import span_mask_tokens

        torch.manual_seed(42)
        # A=5, T=6, C=7, G=8 — 20 real tokens + 4 PAD
        tokens = torch.tensor([[5, 6, 7, 8, 5, 6, 7, 8, 5, 6,
                                7, 8, 5, 6, 7, 8, 5, 6, 7, 8,
                                0, 0, 0, 0]], dtype=torch.long)
        mask = torch.tensor([[1]*20 + [0]*4], dtype=torch.long)

        masked, labels = span_mask_tokens(tokens.clone(), mask, mask_ratio=0.15,
                                          span_mean_length=3, mask_token_id=1, num_special=5)
        # Some tokens in the real region should be masked (replaced with mask_token_id=1)
        real_region = masked[0, :20]
        n_mask_tokens = (real_region == 1).sum().item()
        assert n_mask_tokens > 0, "At least some real tokens should be masked"
        assert n_mask_tokens <= 20, "Cannot mask more tokens than exist"
        # Output shape should match input
        assert masked.shape == tokens.shape
        assert labels.shape == tokens.shape

    def test_snp_noise_preserves_pad(self):
        """T20: Verify SNP noise doesn't corrupt PAD tokens."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        import torch
        from train import snp_noise

        torch.manual_seed(42)
        tokens = torch.tensor([[5, 6, 7, 8, 5, 6, 0, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long)

        noisy = snp_noise(tokens.clone(), mask, rate=0.5, num_special=5)
        # PAD tokens must remain 0
        assert noisy[0, 6].item() == 0
        assert noisy[0, 7].item() == 0
        # All non-PAD tokens should be >= NUM_SPECIAL
        for i in range(6):
            assert noisy[0, i].item() >= 5


class TestModelFactory:
    """Test model building."""

    def test_build_transformer(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from train import build_model
        model = build_model("transformer", vocab_size=10, d_model=64, n_heads=4,
                           d_ff=128, n_layers=2, max_len=64, dropout=0.1,
                           task_type="pretrain")
        assert model is not None
        params = sum(p.numel() for p in model.parameters())
        assert params > 0

    def test_build_cnn(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from train import build_model
        model = build_model("cnn", vocab_size=10, d_model=64, n_heads=4,
                           d_ff=128, n_layers=2, max_len=64, dropout=0.1,
                           task_type="pretrain")
        assert model is not None

    def test_build_lstm(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from train import build_model
        model = build_model("lstm", vocab_size=10, d_model=64, n_heads=4,
                           d_ff=128, n_layers=2, max_len=64, dropout=0.1,
                           task_type="pretrain")
        assert model is not None

    def test_model_forward(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        import torch
        from train import build_model
        model = build_model("transformer", vocab_size=10, d_model=64, n_heads=4,
                           d_ff=128, n_layers=2, max_len=64, dropout=0.1,
                           task_type="pretrain")
        model.eval()
        tokens = torch.randint(5, 10, (2, 32))
        mask = torch.ones(2, 32, dtype=torch.long)
        with torch.no_grad():
            out = model(tokens, attention_mask=mask)
        assert out.shape == (2, 32, 10)  # (batch, seq_len, vocab_size)


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

        # Check checkpoint
        ckpt_dir = os.path.join(templates_dir, "checkpoints")
        assert os.path.exists(os.path.join(ckpt_dir, "best_model.pt"))

        # Check results.tsv
        assert os.path.exists(os.path.join(templates_dir, "results.tsv"))

        # Clean up
        import shutil
        shutil.rmtree(reports_dir, ignore_errors=True)
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        results_tsv = os.path.join(templates_dir, "results.tsv")
        if os.path.exists(results_tsv):
            os.remove(results_tsv)
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "genomic-research")
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_classify_e2e(self, synthetic_csv):
        """Run a full classification cycle with minimal time budget."""
        templates_dir = str(Path(__file__).parent.parent / "genomic_research" / "templates")

        # Run prepare.py for classification
        result = subprocess.run(
            [sys.executable, os.path.join(templates_dir, "prepare.py"),
             "--csv", synthetic_csv, "--seq-col", "sequence",
             "--task", "classify", "--label-col", "species",
             "--tokenizer", "char", "--max-length", "64"],
            capture_output=True, text=True, cwd=templates_dir,
        )
        assert result.returncode == 0, f"prepare.py classify failed:\n{result.stderr}"
        assert "classify" in result.stdout

        # Run train.py
        env = os.environ.copy()
        env["GENOMIC_TIME_BUDGET"] = "10"
        result = subprocess.run(
            [sys.executable, os.path.join(templates_dir, "train.py")],
            capture_output=True, text=True, cwd=templates_dir,
            env=env, timeout=120,
        )
        assert result.returncode == 0, f"train.py classify failed:\n{result.stderr}"
        assert "val_accuracy" in result.stdout

        # Check confusion matrix report
        reports_dir = os.path.join(templates_dir, "reports")
        assert os.path.exists(os.path.join(reports_dir, "confusion_matrix.png"))

        with open(os.path.join(reports_dir, "metrics.json")) as f:
            metrics = json.load(f)
        assert "val_accuracy" in metrics
        assert "val_f1_macro" in metrics

        # Clean up
        import shutil
        shutil.rmtree(reports_dir, ignore_errors=True)
        shutil.rmtree(os.path.join(templates_dir, "checkpoints"), ignore_errors=True)
        results_tsv = os.path.join(templates_dir, "results.tsv")
        if os.path.exists(results_tsv):
            os.remove(results_tsv)
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "genomic-research")
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_clm_e2e(self, synthetic_fasta):
        """Run a CLM pre-training cycle."""
        templates_dir = str(Path(__file__).parent.parent / "genomic_research" / "templates")

        # Prepare
        result = subprocess.run(
            [sys.executable, os.path.join(templates_dir, "prepare.py"),
             "--fasta", synthetic_fasta, "--task", "pretrain",
             "--tokenizer", "char", "--max-length", "64"],
            capture_output=True, text=True, cwd=templates_dir,
        )
        assert result.returncode == 0, f"prepare.py failed:\n{result.stderr}"

        # Train with CLM objective via config override
        env = os.environ.copy()
        env["GENOMIC_TIME_BUDGET"] = "10"
        config_path = os.path.join(templates_dir, "_test_config.json")
        with open(config_path, "w") as f:
            json.dump({"objective": "clm"}, f)
        env["GENOMIC_CONFIG"] = config_path

        result = subprocess.run(
            [sys.executable, os.path.join(templates_dir, "train.py")],
            capture_output=True, text=True, cwd=templates_dir,
            env=env, timeout=120,
        )
        assert result.returncode == 0, f"train.py CLM failed:\n{result.stderr}"
        assert "clm" in result.stdout
        assert "val_perplexity" in result.stdout

        # Clean up
        import shutil
        os.remove(config_path)
        shutil.rmtree(os.path.join(templates_dir, "reports"), ignore_errors=True)
        shutil.rmtree(os.path.join(templates_dir, "checkpoints"), ignore_errors=True)
        results_tsv = os.path.join(templates_dir, "results.tsv")
        if os.path.exists(results_tsv):
            os.remove(results_tsv)
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "genomic-research")
        shutil.rmtree(cache_dir, ignore_errors=True)


class TestInference:
    """Test inference script."""

    def test_inference_pretrain(self, synthetic_fasta):
        """Test inference on a pre-trained model."""
        templates_dir = str(Path(__file__).parent.parent / "genomic_research" / "templates")

        # Prepare + train
        subprocess.run(
            [sys.executable, os.path.join(templates_dir, "prepare.py"),
             "--fasta", synthetic_fasta, "--task", "pretrain",
             "--tokenizer", "char", "--max-length", "64"],
            capture_output=True, text=True, cwd=templates_dir,
        )
        env = os.environ.copy()
        env["GENOMIC_TIME_BUDGET"] = "10"
        subprocess.run(
            [sys.executable, os.path.join(templates_dir, "train.py")],
            capture_output=True, text=True, cwd=templates_dir, env=env, timeout=120,
        )

        ckpt = os.path.join(templates_dir, "checkpoints", "best_model.pt")
        assert os.path.exists(ckpt), "No checkpoint found for inference test"

        # Run inference
        result = subprocess.run(
            [sys.executable, os.path.join(templates_dir, "inference.py"),
             "--checkpoint", ckpt, "--fasta", synthetic_fasta],
            capture_output=True, text=True, cwd=templates_dir, timeout=120,
        )
        assert result.returncode == 0, f"inference.py failed:\n{result.stderr}"

        # Clean up
        import shutil
        shutil.rmtree(os.path.join(templates_dir, "reports"), ignore_errors=True)
        shutil.rmtree(os.path.join(templates_dir, "checkpoints"), ignore_errors=True)
        results_tsv = os.path.join(templates_dir, "results.tsv")
        if os.path.exists(results_tsv):
            os.remove(results_tsv)
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "genomic-research")
        shutil.rmtree(cache_dir, ignore_errors=True)


class TestKmerVariants:
    """Test k-mer tokenizer with different k values."""

    def test_kmer_k3(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from prepare import KmerTokenizer
        tok = KmerTokenizer(k=3)
        assert tok.vocab_size == 5 + 64  # 4^3 + specials
        ids = tok.encode("ATCGATCG")
        decoded = tok.decode(ids)
        assert decoded == "ATCGATCG"

    def test_kmer_k4(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from prepare import KmerTokenizer
        tok = KmerTokenizer(k=4)
        assert tok.vocab_size == 5 + 256  # 4^4 + specials
        ids = tok.encode("ATCGATCGATCG")
        decoded = tok.decode(ids)
        assert decoded == "ATCGATCGATCG"

    def test_kmer_k6(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from prepare import KmerTokenizer
        tok = KmerTokenizer(k=6)
        assert tok.vocab_size == 5 + 4096  # 4^6 + specials
        ids = tok.encode("ATCGATCGATCGATCG")
        decoded = tok.decode(ids)
        assert decoded == "ATCGATCGATCGATCG"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_file_not_found(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from prepare import prepare_data
        with pytest.raises(FileNotFoundError):
            prepare_data("/nonexistent/file.fasta", "pretrain")

    def test_invalid_task_type(self, synthetic_fasta):
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from prepare import prepare_data
        with pytest.raises(ValueError, match="Unknown task type"):
            prepare_data(synthetic_fasta, "invalid_task")

    def test_short_sequence(self, synthetic_fasta):
        """Test with sequences shorter than max_length."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from prepare import prepare_data, CACHE_DIR
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
        config = prepare_data(synthetic_fasta, "pretrain", max_length=1024)
        assert config["n_train"] > 0
        shutil.rmtree(CACHE_DIR)

    def test_single_char_tokenizer_all_n(self):
        """Test tokenizer with all-N sequence."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "genomic_research" / "templates"))
        from prepare import CharTokenizer
        tok = CharTokenizer()
        ids = tok.encode("NNNNN")
        assert len(ids) == 5
        assert all(i == 9 for i in ids)
        assert tok.decode(ids) == "NNNNN"
