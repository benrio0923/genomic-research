"""
Data preparation and evaluation for genomic-research.

Supports:
  - FASTA (.fasta/.fa/.fna), FASTQ (.fastq/.fq), CSV (.csv) inputs
  - Three tokenizers: char, kmer, bpe
  - Tasks: pretrain (MLM/CLM), classify, regress

Data is cached in ~/.cache/genomic-research/.
"""

import os
import json
import math
import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = int(os.environ.get("GENOMIC_TIME_BUDGET", 300))
VAL_RATIO = 0.2
RANDOM_SEED = 42
CHUNK_OVERLAP = 0.5  # 50% overlap for long sequence chunking

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "genomic-research")
CONFIG_PATH = os.path.join(CACHE_DIR, "task_config.json")

# Special tokens
PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 1
CLS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
UNK_TOKEN_ID = 4
SPECIAL_TOKENS = {"[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[SEP]": 3, "[UNK]": 4}
NUM_SPECIAL = len(SPECIAL_TOKENS)

# ---------------------------------------------------------------------------
# Sequence loading
# ---------------------------------------------------------------------------

def _clean_sequence(seq):
    """Uppercase and replace non-ATCG with N."""
    seq = seq.upper().strip()
    seq = re.sub(r"[^ATCGN]", "N", seq)
    return seq


def load_sequences(path, seq_col=None, id_col=None):
    """
    Load sequences from FASTA, FASTQ, or CSV.
    Supports directory input (loads all sequence files) and glob patterns.

    Returns: list of (id, sequence) tuples.
    """
    import glob as glob_mod

    path = str(path)

    # Handle directory input — load all sequence files
    if os.path.isdir(path):
        all_seqs = []
        exts = (".fasta", ".fa", ".fna", ".fastq", ".fq", ".csv", ".gb", ".gbk", ".genbank")
        files = sorted(f for f in os.listdir(path)
                       if any(f.lower().endswith(e) for e in exts))
        if not files:
            raise ValueError(f"No sequence files found in directory: {path}")
        print(f"  Found {len(files)} sequence files in {path}")
        for fname in files:
            fpath = os.path.join(path, fname)
            seqs = load_sequences(fpath, seq_col=seq_col, id_col=id_col)
            all_seqs.extend(seqs)
        return all_seqs

    # Handle glob patterns (e.g., "data/*.fasta")
    if "*" in path or "?" in path:
        matched = sorted(glob_mod.glob(path))
        if not matched:
            raise FileNotFoundError(f"No files matched pattern: {path}")
        print(f"  Glob matched {len(matched)} files")
        all_seqs = []
        for fpath in matched:
            seqs = load_sequences(fpath, seq_col=seq_col, id_col=id_col)
            all_seqs.extend(seqs)
        return all_seqs

    ext = Path(path).suffix.lower()

    if ext in (".fasta", ".fa", ".fna"):
        return _load_fasta(path)
    elif ext in (".fastq", ".fq"):
        return _load_fastq(path)
    elif ext in (".gb", ".gbk", ".genbank"):
        return _load_genbank(path)
    elif ext == ".csv":
        return _load_csv_sequences(path, seq_col=seq_col, id_col=id_col)
    else:
        # Try FASTA by default
        try:
            return _load_fasta(path)
        except Exception:
            raise ValueError(f"Cannot determine format for '{path}'. Use .fasta, .fastq, .gb, or .csv extension.")


def _load_fasta(path):
    """Load sequences from FASTA file using BioPython."""
    from Bio import SeqIO
    sequences = []
    for record in SeqIO.parse(path, "fasta"):
        seq = _clean_sequence(str(record.seq))
        if len(seq) > 0:
            sequences.append((record.id, seq))
    return sequences


def _load_genbank(path):
    """Load sequences from GenBank (.gb/.gbk) file using BioPython."""
    from Bio import SeqIO
    sequences = []
    for record in SeqIO.parse(path, "genbank"):
        seq = _clean_sequence(str(record.seq))
        if len(seq) > 0:
            # Use accession or locus name as ID
            sid = record.id if record.id != "<unknown id>" else record.name
            sequences.append((sid, seq))
    if sequences:
        print(f"  GenBank: loaded {len(sequences)} sequences from {path}")
    return sequences


def _load_fastq(path, min_quality=0, min_length=0):
    """Load sequences from FASTQ file using BioPython with optional QC filtering.

    Args:
        min_quality: Minimum mean Phred quality score (0 = no filtering).
        min_length: Minimum sequence length after trimming (0 = no filtering).
    """
    from Bio import SeqIO
    sequences = []
    n_total = 0
    n_filtered_quality = 0
    n_filtered_length = 0
    quality_scores = []

    for record in SeqIO.parse(path, "fastq"):
        n_total += 1
        quals = record.letter_annotations.get("phred_quality", [])
        if quals:
            mean_q = sum(quals) / len(quals)
            quality_scores.append(mean_q)
            if min_quality > 0 and mean_q < min_quality:
                n_filtered_quality += 1
                continue

        seq = _clean_sequence(str(record.seq))
        if min_length > 0 and len(seq) < min_length:
            n_filtered_length += 1
            continue
        if len(seq) > 0:
            sequences.append((record.id, seq))

    # Print QC summary
    if n_total > 0:
        print(f"FASTQ QC: {n_total} reads total")
        if quality_scores:
            q_arr = np.array(quality_scores)
            print(f"  Mean quality: {q_arr.mean():.1f} (min={q_arr.min():.1f}, max={q_arr.max():.1f})")
        if n_filtered_quality > 0:
            print(f"  Filtered (quality < {min_quality}): {n_filtered_quality}")
        if n_filtered_length > 0:
            print(f"  Filtered (length < {min_length}): {n_filtered_length}")
        print(f"  Passed: {len(sequences)}")

    return sequences


def _load_csv_sequences(path, seq_col=None, id_col=None):
    """Load sequences from CSV. seq_col specifies the sequence column."""
    import csv
    sequences = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if not fields:
            raise ValueError(f"CSV file has no header row: {path}")
        if seq_col and seq_col not in fields:
            raise ValueError(f"Column '{seq_col}' not found in CSV. Available: {fields}")
        if id_col and id_col not in fields:
            raise ValueError(f"ID column '{id_col}' not found in CSV. Available: {fields}")
        if seq_col is None:
            # Auto-detect: look for common names
            for candidate in ["sequence", "seq", "dna", "nucleotide", "Sequence"]:
                if candidate in fields:
                    seq_col = candidate
                    break
            if seq_col is None:
                raise ValueError(f"Cannot auto-detect sequence column. Available: {fields}. Use --seq-col.")

        for i, row in enumerate(reader):
            seq = _clean_sequence(row[seq_col])
            sid = row.get(id_col, f"seq_{i}") if id_col else f"seq_{i}"
            if len(seq) > 0:
                sequences.append((sid, seq))
    return sequences


def load_paired_sequences(r1_path, r2_path, merge="concatenate"):
    """Load paired-end FASTQ reads and merge R1/R2 pairs.

    Args:
        r1_path: Path to R1 (forward) FASTQ file.
        r2_path: Path to R2 (reverse) FASTQ file.
        merge: How to combine pairs — "concatenate" joins with NNN separator,
               "r1_only" uses only forward reads, "interleave" alternates.

    Returns: list of (id, sequence) tuples.
    """
    from Bio import SeqIO

    r1_records = {r.id: r for r in SeqIO.parse(r1_path, "fastq")}
    r2_records = {r.id: r for r in SeqIO.parse(r2_path, "fastq")}

    # Match pairs by ID (strip /1 /2 suffixes if present)
    def _strip_pair_suffix(rid):
        for suffix in ("/1", "/2", ".1", ".2", " 1", " 2"):
            if rid.endswith(suffix):
                return rid[:-len(suffix)]
        return rid

    r1_by_base = {}
    for rid, rec in r1_records.items():
        base = _strip_pair_suffix(rid)
        r1_by_base[base] = rec

    sequences = []
    n_paired = 0
    n_unpaired = 0

    for rid, rec2 in r2_records.items():
        base = _strip_pair_suffix(rid)
        if base in r1_by_base:
            rec1 = r1_by_base[base]
            s1 = _clean_sequence(str(rec1.seq))
            s2 = _clean_sequence(str(rec2.seq))
            n_paired += 1

            if merge == "concatenate":
                combined = s1 + "NNN" + s2
                sequences.append((base, combined))
            elif merge == "r1_only":
                sequences.append((base, s1))
            elif merge == "interleave":
                sequences.append((base + "_R1", s1))
                sequences.append((base + "_R2", s2))
            else:
                sequences.append((base, s1 + "NNN" + s2))
        else:
            n_unpaired += 1

    print(f"  Paired-end: {n_paired} pairs matched, {n_unpaired} unpaired")
    return sequences


def compute_sequence_weights(sequences, n_clusters=None):
    """Weight sequences inversely by cluster frequency for balanced training.

    Uses k-mer frequency vectors + simple clustering to group similar sequences,
    then weights each sequence inversely by its cluster size.

    Args:
        sequences: list of (id, sequence) tuples.
        n_clusters: Number of clusters. Default: sqrt(n_sequences), capped at 100.

    Returns: list of float weights (same length as sequences).
    """
    n = len(sequences)
    if n <= 1:
        return [1.0] * n

    if n_clusters is None:
        n_clusters = min(100, max(2, int(n ** 0.5)))
    n_clusters = min(n_clusters, n)

    # Build 4-mer frequency vectors (256-dim)
    def _kmer_freq(seq, k=4):
        counts = {}
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if all(c in "ATCG" for c in kmer):
                counts[kmer] = counts.get(kmer, 0) + 1
        total = sum(counts.values()) or 1
        # Fixed ordering: all 4^k k-mers
        bases = "ATCG"
        all_kmers = []
        def _gen(prefix, depth):
            if depth == 0:
                all_kmers.append(prefix)
                return
            for b in bases:
                _gen(prefix + b, depth - 1)
        _gen("", k)
        return [counts.get(km, 0) / total for km in all_kmers]

    # Build feature matrix
    features = np.array([_kmer_freq(seq) for _, seq in sequences])

    # Simple k-means clustering
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=1, max_iter=50)
    labels = kmeans.fit_predict(features)

    # Count cluster sizes
    cluster_sizes = np.bincount(labels, minlength=n_clusters)

    # Weight inversely by cluster size (normalized)
    weights = [1.0 / max(cluster_sizes[labels[i]], 1) for i in range(n)]
    total_w = sum(weights)
    weights = [w * n / total_w for w in weights]  # normalize so mean weight = 1.0

    print(f"  Sequence weighting: {n_clusters} clusters, weight range [{min(weights):.3f}, {max(weights):.3f}]")
    return weights


def load_gff_annotations(gff_path, seq_dict=None):
    """Load GFF/GTF annotations and create per-position labels.

    Args:
        gff_path: Path to GFF3 or GTF file.
        seq_dict: Optional dict mapping seq_id → sequence string (for length info).

    Returns: dict mapping seq_id → list of (start, end, feature_type) tuples.
    """
    annotations = {}
    with open(gff_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            seq_id = parts[0]
            feature_type = parts[2]  # gene, CDS, exon, etc.
            try:
                start = int(parts[3]) - 1  # GFF is 1-based
                end = int(parts[4])        # GFF end is inclusive
            except ValueError:
                continue
            annotations.setdefault(seq_id, []).append((start, end, feature_type))

    n_features = sum(len(v) for v in annotations.values())
    print(f"  GFF/GTF: loaded {n_features} features across {len(annotations)} sequences")
    return annotations


def gff_to_position_labels(annotations, seq_length, feature_types=None):
    """Convert GFF annotations to per-position integer labels.

    Args:
        annotations: list of (start, end, feature_type) from load_gff_annotations.
        seq_length: Length of the sequence.
        feature_types: Optional list of feature types to include. Default: all types.

    Returns: numpy array of shape (seq_length,) with integer labels.
             0 = intergenic, 1+ = feature type index.
    """
    # Build feature type → label mapping
    if feature_types is None:
        all_types = sorted(set(ft for _, _, ft in annotations))
    else:
        all_types = list(feature_types)
    type_to_label = {ft: i + 1 for i, ft in enumerate(all_types)}

    labels = np.zeros(seq_length, dtype=np.int64)
    for start, end, ft in annotations:
        if ft in type_to_label:
            s = max(0, start)
            e = min(end, seq_length)
            labels[s:e] = type_to_label[ft]

    return labels, type_to_label


def _clean_protein_sequence(seq):
    """Clean protein sequence: uppercase, replace non-standard with X."""
    seq = seq.upper().strip()
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(c if c in valid else "X" for c in seq)


def detect_sequence_type(sequences, sample_size=10):
    """Auto-detect if sequences are DNA or protein.

    Returns: "dna" or "protein"
    """
    sample = [seq for _, seq in sequences[:sample_size]]
    dna_chars = set("ATCGN")
    protein_chars = set("ACDEFGHIKLMNPQRSTVWYX")

    dna_score = 0
    for seq in sample:
        upper = seq.upper()
        if len(upper) > 0:
            dna_frac = sum(1 for c in upper if c in dna_chars) / len(upper)
            dna_score += dna_frac

    avg_dna = dna_score / max(len(sample), 1)
    return "dna" if avg_dna > 0.85 else "protein"


class ProteinTokenizer:
    """Character-level protein tokenizer: 20 amino acids + X (unknown) + special tokens."""

    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        for i, aa in enumerate(self.AMINO_ACIDS):
            self.char_to_id[aa] = NUM_SPECIAL + i
            self.id_to_char[NUM_SPECIAL + i] = aa
        # X = unknown amino acid
        self.char_to_id["X"] = NUM_SPECIAL + len(self.AMINO_ACIDS)
        self.id_to_char[NUM_SPECIAL + len(self.AMINO_ACIDS)] = "X"
        self.vocab_size = NUM_SPECIAL + len(self.AMINO_ACIDS) + 1  # 26

    def encode(self, sequence):
        return [self.char_to_id.get(c, UNK_TOKEN_ID) for c in sequence.upper()]

    def decode(self, ids):
        return "".join(self.id_to_char.get(i, "?") for i in ids if i >= NUM_SPECIAL)


def load_vcf_variants(vcf_path, reference_seq=None):
    """Load VCF variant file and optionally apply variants to a reference sequence.

    Args:
        vcf_path: Path to VCF file.
        reference_seq: Optional reference sequence string to apply variants to.

    Returns:
        If reference_seq is provided: list of (sample_id, mutated_sequence) tuples.
        If not: list of variant dicts with chrom, pos, ref, alt fields.
    """
    variants = []
    samples = []
    with open(vcf_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                parts = line.split("\t")
                if len(parts) > 9:
                    samples = parts[9:]
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            try:
                chrom = parts[0]
                pos = int(parts[1]) - 1  # VCF is 1-based
                ref = parts[3]
                alt = parts[4].split(",")[0]  # take first alt allele
                variant = {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt}
                # Parse genotypes if samples present
                if samples and len(parts) > 9:
                    gt_field = parts[8].split(":").index("GT") if "GT" in parts[8] else 0
                    for i, gt_str in enumerate(parts[9:]):
                        gt = gt_str.split(":")[gt_field] if ":" in gt_str else gt_str
                        variant[f"sample_{i}"] = gt
                variants.append(variant)
            except (ValueError, IndexError):
                continue

    print(f"  VCF: loaded {len(variants)} variants")

    if reference_seq is None:
        return variants

    # Apply variants to reference to generate sample sequences
    ref_seq = list(reference_seq.upper())
    sequences = []

    if not samples:
        # Single mutant sequence with all ALT alleles applied
        mutant = ref_seq.copy()
        for v in sorted(variants, key=lambda x: x["pos"], reverse=True):
            p = v["pos"]
            if 0 <= p < len(mutant):
                mutant[p:p + len(v["ref"])] = list(v["alt"])
        sequences.append(("mutant", _clean_sequence("".join(mutant))))
    else:
        # Per-sample sequences based on genotypes
        for si, sample in enumerate(samples):
            mutant = ref_seq.copy()
            for v in sorted(variants, key=lambda x: x["pos"], reverse=True):
                gt = v.get(f"sample_{si}", "0/0")
                has_alt = "1" in gt.replace("|", "/").split("/")
                if has_alt:
                    p = v["pos"]
                    if 0 <= p < len(mutant):
                        mutant[p:p + len(v["ref"])] = list(v["alt"])
            sequences.append((sample, _clean_sequence("".join(mutant))))

    return sequences


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Character-level tokenizer: A=5, T=6, C=7, G=8, N=9."""

    def __init__(self):
        self.char_to_id = {"A": 5, "T": 6, "C": 7, "G": 8, "N": 9}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = NUM_SPECIAL + 5  # 10

    def encode(self, sequence):
        return [self.char_to_id.get(c, UNK_TOKEN_ID) for c in sequence]

    def decode(self, ids):
        return "".join(self.id_to_char.get(i, "?") for i in ids if i >= NUM_SPECIAL)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "char", "vocab_size": self.vocab_size}, f)

    @classmethod
    def load(cls, path):
        return cls()


class KmerTokenizer:
    """K-mer tokenizer: generates all possible k-mers."""

    def __init__(self, k=6):
        self.k = k
        self.kmer_to_id = {}
        bases = "ATCG"
        # Generate all k-mers
        kmers = [""]
        for _ in range(k):
            kmers = [km + b for km in kmers for b in bases]
        for i, kmer in enumerate(sorted(kmers)):
            self.kmer_to_id[kmer] = NUM_SPECIAL + i
        self.id_to_kmer = {v: k for k, v in self.kmer_to_id.items()}
        self.vocab_size = NUM_SPECIAL + len(self.kmer_to_id)

    def encode(self, sequence):
        tokens = []
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]
            tokens.append(self.kmer_to_id.get(kmer, UNK_TOKEN_ID))
        return tokens

    def decode(self, ids):
        parts = [self.id_to_kmer.get(i, "?") for i in ids if i >= NUM_SPECIAL]
        if not parts:
            return ""
        # Reconstruct: first kmer full, then last char of each subsequent
        result = parts[0]
        for p in parts[1:]:
            result += p[-1]
        return result

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "kmer", "k": self.k, "vocab_size": self.vocab_size}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            cfg = json.load(f)
        return cls(k=cfg["k"])


class BPETokenizer:
    """BPE tokenizer using HuggingFace tokenizers library."""

    def __init__(self, vocab_size=4096):
        self._vocab_size = vocab_size
        self._tokenizer = None

    @property
    def vocab_size(self):
        if self._tokenizer is not None:
            return self._tokenizer.get_vocab_size()
        return self._vocab_size

    def train(self, sequences, vocab_size=None):
        """Train BPE tokenizer on a list of sequences."""
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import Split
            from tokenizers import Regex
        except ImportError:
            raise ImportError("BPE tokenizer requires: pip install genomic-research[bpe]")

        vs = vocab_size or self._vocab_size
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Split(Regex(r"."), behavior="isolated")
        trainer = BpeTrainer(
            vocab_size=vs,
            special_tokens=["[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]"],
            min_frequency=2,
        )

        # Train from iterator
        tokenizer.train_from_iterator(sequences, trainer=trainer)
        self._tokenizer = tokenizer
        self._vocab_size = tokenizer.get_vocab_size()

    def encode(self, sequence):
        if self._tokenizer is None:
            raise RuntimeError("BPE tokenizer not trained. Call train() first.")
        return self._tokenizer.encode(sequence).ids

    def decode(self, ids):
        if self._tokenizer is None:
            return ""
        return self._tokenizer.decode(ids)

    def save(self, path):
        if self._tokenizer is None:
            raise RuntimeError("BPE tokenizer not trained.")
        tok_path = str(path) + ".bpe.json"
        self._tokenizer.save(tok_path)
        with open(path, "w") as f:
            json.dump({"type": "bpe", "vocab_size": self._vocab_size, "tokenizer_path": tok_path}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            cfg = json.load(f)
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError("BPE tokenizer requires: pip install genomic-research[bpe]")
        obj = cls(vocab_size=cfg["vocab_size"])
        obj._tokenizer = Tokenizer.from_file(cfg["tokenizer_path"])
        return obj


class CodonTokenizer:
    """Codon-aware tokenizer: tokenizes in reading frames (3 bases per token).

    64 codons + special tokens. Handles coding regions more naturally.
    """

    def __init__(self):
        self.codon_to_id = {}
        self.id_to_codon = {}
        bases = "ATCG"
        codons = [a + b + c for a in bases for b in bases for c in bases]
        for i, codon in enumerate(sorted(codons)):
            self.codon_to_id[codon] = NUM_SPECIAL + i
            self.id_to_codon[NUM_SPECIAL + i] = codon
        # Codons with N
        self.vocab_size = NUM_SPECIAL + len(self.codon_to_id) + 1  # +1 for unknown codon
        self._unk_codon_id = NUM_SPECIAL + len(self.codon_to_id)

    def encode(self, sequence):
        tokens = []
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            tokens.append(self.codon_to_id.get(codon, self._unk_codon_id))
        return tokens

    def decode(self, ids):
        return "".join(self.id_to_codon.get(i, "NNN") for i in ids if i >= NUM_SPECIAL)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "codon", "vocab_size": self.vocab_size}, f)

    @classmethod
    def load(cls, path):
        return cls()


def detect_orfs(sequence, min_length=100):
    """Find open reading frames in a DNA sequence.

    Args:
        sequence: DNA sequence string (ATCG).
        min_length: Minimum ORF length in nucleotides.

    Returns: list of (start, end, frame, strand) tuples.
    """
    start_codon = "ATG"
    stop_codons = {"TAA", "TAG", "TGA"}
    orfs = []

    # Search all 3 reading frames on forward strand
    for frame in range(3):
        i = frame
        orf_start = None
        while i <= len(sequence) - 3:
            codon = sequence[i:i+3]
            if codon == start_codon and orf_start is None:
                orf_start = i
            elif codon in stop_codons and orf_start is not None:
                orf_len = i + 3 - orf_start
                if orf_len >= min_length:
                    orfs.append((orf_start, i + 3, frame, "+"))
                orf_start = None
            i += 3

    # Reverse complement for reverse strand
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    rc_seq = "".join(complement.get(c, "N") for c in reversed(sequence))
    for frame in range(3):
        i = frame
        orf_start = None
        while i <= len(rc_seq) - 3:
            codon = rc_seq[i:i+3]
            if codon == start_codon and orf_start is None:
                orf_start = i
            elif codon in stop_codons and orf_start is not None:
                orf_len = i + 3 - orf_start
                if orf_len >= min_length:
                    # Map back to forward strand coordinates
                    fwd_end = len(sequence) - orf_start
                    fwd_start = len(sequence) - (i + 3)
                    orfs.append((fwd_start, fwd_end, frame, "-"))
                orf_start = None
            i += 3

    return sorted(orfs, key=lambda x: x[0])


def orf_to_position_labels(orfs, seq_length):
    """Convert ORF positions to per-position binary labels (coding=1, non-coding=0)."""
    labels = np.zeros(seq_length, dtype=np.int64)
    for start, end, _, _ in orfs:
        s = max(0, start)
        e = min(end, seq_length)
        labels[s:e] = 1
    return labels


def compute_kmer_spectrum(sequences, k_values=(2, 3, 4)):
    """Compute k-mer frequency spectrum features for each sequence.

    Args:
        sequences: list of (id, sequence) tuples.
        k_values: tuple of k values to compute.

    Returns: numpy array of shape (n_sequences, total_features).
    """
    all_features = []
    for _, seq in sequences:
        features = []
        for k in k_values:
            bases = "ATCG"
            all_kmers = []
            def _gen(prefix, depth):
                if depth == 0:
                    all_kmers.append(prefix)
                    return
                for b in bases:
                    _gen(prefix + b, depth - 1)
            _gen("", k)

            counts = {}
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if all(c in "ATCG" for c in kmer):
                    counts[kmer] = counts.get(kmer, 0) + 1
            total = sum(counts.values()) or 1
            features.extend([counts.get(km, 0) / total for km in all_kmers])
        all_features.append(features)

    return np.array(all_features, dtype=np.float32)


def compute_gc_content_features(sequence, window_size=50, stride=10):
    """Compute sliding window GC content features for a sequence.

    Returns: numpy array of shape (n_windows,) with GC fraction per window.
    """
    gc_features = []
    for start in range(0, max(1, len(sequence) - window_size + 1), stride):
        window = sequence[start:start + window_size]
        gc = sum(1 for c in window if c in "GC") / max(len(window), 1)
        gc_features.append(gc)
    return np.array(gc_features, dtype=np.float32)


def phylogenetic_split(sequences, val_ratio=0.2, kmer_k=4, seed=42):
    """Split sequences into train/val based on k-mer distance to reduce data leakage.

    Groups similar sequences by k-mer profile clustering, then assigns
    entire clusters to train or val to ensure phylogenetically distant splits.

    Args:
        sequences: list of (id, sequence) tuples.
        val_ratio: fraction for validation set.
        kmer_k: k-mer size for distance computation.
        seed: random seed.

    Returns: (train_indices, val_indices) as numpy arrays.
    """
    rng = np.random.RandomState(seed)
    n = len(sequences)
    if n < 4:
        # Too few sequences for clustering, fall back to random
        idx = np.arange(n)
        rng.shuffle(idx)
        split = max(1, int(n * val_ratio))
        return idx[split:], idx[:split]

    # Build k-mer frequency vectors
    bases = "ATCG"
    all_kmers = []
    def _gen(prefix, depth):
        if depth == 0:
            all_kmers.append(prefix)
            return
        for b in bases:
            _gen(prefix + b, depth - 1)
    _gen("", kmer_k)
    kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}
    dim = len(all_kmers)

    profiles = np.zeros((n, dim), dtype=np.float32)
    for si, (_, seq) in enumerate(sequences):
        for i in range(len(seq) - kmer_k + 1):
            kmer = seq[i:i+kmer_k]
            ki = kmer_to_idx.get(kmer)
            if ki is not None:
                profiles[si, ki] += 1
        total = profiles[si].sum()
        if total > 0:
            profiles[si] /= total

    # Simple greedy clustering by cosine distance
    from sklearn.cluster import KMeans
    n_clusters = max(5, n // 10)
    n_clusters = min(n_clusters, n)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=3, max_iter=100)
    labels = km.fit_predict(profiles)

    # Assign entire clusters to val until we reach val_ratio
    cluster_ids = np.unique(labels)
    rng.shuffle(cluster_ids)
    val_count = 0
    target_val = int(n * val_ratio)
    val_mask = np.zeros(n, dtype=bool)
    for c in cluster_ids:
        members = np.where(labels == c)[0]
        if val_count + len(members) <= target_val + len(members) // 2:
            val_mask[members] = True
            val_count += len(members)
            if val_count >= target_val:
                break

    if val_count == 0:
        # Fallback: assign smallest cluster to val
        sizes = [(labels == c).sum() for c in cluster_ids]
        smallest = cluster_ids[np.argmin(sizes)]
        val_mask[labels == smallest] = True

    return np.where(~val_mask)[0], np.where(val_mask)[0]


def detect_recombination_breakpoints(sequences, window_size=200, step=50, kmer_k=4):
    """Detect potential recombination breakpoints in sequences.

    Uses a sliding-window approach comparing k-mer profiles of the left vs right
    halves at each position. Sharp changes in similarity indicate breakpoints.

    Args:
        sequences: list of (id, sequence) tuples.
        window_size: size of comparison windows.
        step: step size for scanning.
        kmer_k: k-mer size for profile comparison.

    Returns: dict mapping sequence_id → list of (position, score) breakpoint candidates.
    """
    bases = "ATCG"
    all_kmers = []
    def _gen(prefix, depth):
        if depth == 0:
            all_kmers.append(prefix)
            return
        for b in bases:
            _gen(prefix + b, depth - 1)
    _gen("", kmer_k)
    kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}
    dim = len(all_kmers)

    def _kmer_profile(seq_fragment):
        vec = np.zeros(dim, dtype=np.float32)
        for i in range(len(seq_fragment) - kmer_k + 1):
            km = seq_fragment[i:i+kmer_k]
            ki = kmer_to_idx.get(km)
            if ki is not None:
                vec[ki] += 1
        total = vec.sum()
        if total > 0:
            vec /= total
        return vec

    results = {}
    for seq_id, seq in sequences:
        if len(seq) < window_size * 2:
            results[seq_id] = []
            continue

        scores = []
        for pos in range(window_size, len(seq) - window_size, step):
            left = _kmer_profile(seq[pos - window_size:pos])
            right = _kmer_profile(seq[pos:pos + window_size])
            # Jensen-Shannon divergence (simplified via L2 distance)
            diff = np.sqrt(np.sum((left - right) ** 2))
            scores.append((pos, float(diff)))

        if not scores:
            results[seq_id] = []
            continue

        # Find peaks (positions where divergence is significantly above mean)
        score_vals = np.array([s[1] for s in scores])
        mean_s = score_vals.mean()
        std_s = score_vals.std() + 1e-9
        breakpoints = [(pos, sc) for pos, sc in scores if sc > mean_s + 2 * std_s]
        results[seq_id] = breakpoints

    return results


# ---------------------------------------------------------------------------
# T133/T134/T135: Downstream task label loaders
# ---------------------------------------------------------------------------

def load_task_labels(label_path, task_name="classify"):
    """Load labels for downstream tasks (AMR, host prediction, geographic origin, etc.).

    Supports CSV with columns: id, label (or custom column names).
    Returns dict mapping sequence_id → label string.

    Args:
        label_path: path to CSV file with labels.
        task_name: descriptive name for error messages.

    Returns: dict of {sequence_id: label_string}
    """
    import csv
    labels = {}
    with open(label_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # Auto-detect columns
        id_col = None
        label_col = None
        for col in fieldnames:
            cl = col.lower().strip()
            if cl in ("id", "sequence_id", "seq_id", "accession", "name"):
                id_col = col
            elif cl in ("label", "class", "category", "target", task_name.lower(),
                         "resistance", "host", "origin", "location", "species"):
                label_col = col
        if id_col is None and len(fieldnames) >= 2:
            id_col = fieldnames[0]
        if label_col is None and len(fieldnames) >= 2:
            label_col = fieldnames[-1]
        if id_col is None or label_col is None:
            raise ValueError(f"Cannot detect id/label columns in {label_path} for task '{task_name}'. "
                             f"Found columns: {fieldnames}")
        for row in reader:
            labels[row[id_col].strip()] = row[label_col].strip()
    return labels


def create_tokenizer(tokenizer_type="char", kmer_size=6, bpe_vocab_size=4096):
    """Factory function to create a tokenizer."""
    if tokenizer_type == "char":
        return CharTokenizer()
    elif tokenizer_type == "kmer":
        return KmerTokenizer(k=kmer_size)
    elif tokenizer_type == "bpe":
        return BPETokenizer(vocab_size=bpe_vocab_size)
    elif tokenizer_type == "codon":
        return CodonTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Use char, kmer, bpe, or codon.")


def load_tokenizer(path):
    """Load a saved tokenizer from metadata file."""
    with open(path) as f:
        cfg = json.load(f)
    t = cfg["type"]
    if t == "char":
        return CharTokenizer.load(path)
    elif t == "kmer":
        return KmerTokenizer.load(path)
    elif t == "bpe":
        return BPETokenizer.load(path)
    elif t == "codon":
        return CodonTokenizer.load(path)
    elif t == "protein":
        return ProteinTokenizer.load(path)
    else:
        raise ValueError(f"Unknown tokenizer type: {t}")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _chunk_tokens(token_ids, max_length, overlap_ratio=CHUNK_OVERLAP, strategy="fixed"):
    """Split a long token sequence into chunks.

    Strategies:
        fixed: fixed overlap ratio (default 50%)
        none: no overlap, consecutive non-overlapping chunks
        random: random start offset for each chunk
        slide: sliding window with 1-token stride (generates many chunks)
    """
    if len(token_ids) <= max_length:
        return [token_ids]

    if strategy == "none":
        # No overlap
        chunks = []
        for start in range(0, len(token_ids), max_length):
            chunk = token_ids[start:start + max_length]
            if len(chunk) < max_length // 4:
                break
            chunks.append(chunk)
        return chunks

    if strategy == "random":
        # Random offset chunks
        rng = np.random.RandomState(42)
        n_chunks = max(1, len(token_ids) // max_length)
        chunks = []
        for _ in range(n_chunks):
            start = rng.randint(0, max(1, len(token_ids) - max_length))
            chunks.append(token_ids[start:start + max_length])
        return chunks

    if strategy == "slide":
        # Sliding window with small stride (max_length // 4)
        stride = max(1, max_length // 4)
        chunks = []
        for start in range(0, len(token_ids) - max_length // 4, stride):
            chunk = token_ids[start:start + max_length]
            if len(chunk) < max_length // 4:
                break
            chunks.append(chunk)
        return chunks

    # Default: fixed overlap
    stride = max(1, int(max_length * (1 - overlap_ratio)))
    chunks = []
    for start in range(0, len(token_ids), stride):
        chunk = token_ids[start:start + max_length]
        if len(chunk) < max_length // 4:
            break
        chunks.append(chunk)
    return chunks


def _reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    complement = str.maketrans("ATCGatcgNn", "TAGCtagcNn")
    return seq.translate(complement)[::-1]


def prepare_data(
    seq_path,
    task_type="pretrain",
    tokenizer_type="char",
    kmer_size=6,
    max_length=512,
    labels_path=None,
    label_col=None,
    seq_col=None,
    id_col=None,
    bpe_vocab_size=4096,
    sample_n=0,
    sample_frac=0.0,
    rc_double=False,
    chunk_strategy="fixed",
    n_folds=1,
):
    """
    Prepare genomic data for training.

    Args:
        seq_path: path to FASTA/FASTQ/CSV file
        task_type: "pretrain", "classify", or "regress"
        tokenizer_type: "char", "kmer", or "bpe"
        kmer_size: k-mer size (only for kmer tokenizer)
        max_length: maximum sequence length in tokens
        labels_path: path to labels CSV (for classify/regress)
        label_col: label column name in labels CSV
        seq_col: sequence column name (for CSV input)
        id_col: ID column name (for CSV input)
        bpe_vocab_size: vocabulary size for BPE tokenizer
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Input validation
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Input file not found: {seq_path}")
    if labels_path and not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if task_type not in ("pretrain", "classify", "regress"):
        raise ValueError(f"Unknown task type: {task_type}. Must be 'pretrain', 'classify', or 'regress'.")
    if seq_col and not seq_path.lower().endswith(".csv"):
        print("Warning: --seq-col specified but file does not end with .csv")
    if task_type in ("classify", "regress") and not labels_path and not seq_col:
        print("Warning: classify/regress task but no labels source specified. "
              "Use --labels <file> or --seq-col + --label-col for CSV input.")

    # Load sequences
    print(f"Loading sequences from {seq_path}...")
    sequences = load_sequences(seq_path, seq_col=seq_col, id_col=id_col)
    print(f"  Loaded {len(sequences)} sequences")

    if len(sequences) == 0:
        raise ValueError("No sequences found in input file.")

    # Subsample if requested (for quick experiments on large datasets)
    if sample_n > 0 and sample_n < len(sequences):
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(sequences), size=sample_n, replace=False)
        sequences = [sequences[i] for i in sorted(idx)]
        print(f"  Subsampled to {len(sequences)} sequences (sample_n={sample_n})")
    elif 0 < sample_frac < 1.0:
        rng = np.random.RandomState(RANDOM_SEED)
        n = max(1, int(len(sequences) * sample_frac))
        idx = rng.choice(len(sequences), size=n, replace=False)
        sequences = [sequences[i] for i in sorted(idx)]
        print(f"  Subsampled to {len(sequences)} sequences (sample_frac={sample_frac})")

    # Deduplicate sequences (exact matches)
    seen_hashes = set()
    unique_sequences = []
    for sid, seq in sequences:
        h = hashlib.md5(seq.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_sequences.append((sid, seq))
    n_dupes = len(sequences) - len(unique_sequences)
    if n_dupes > 0:
        print(f"  Removed {n_dupes} exact duplicate sequences ({len(unique_sequences)} unique)")
        sequences = unique_sequences

    # Reverse complement data doubling
    if rc_double:
        rc_seqs = [(f"{sid}_rc", _reverse_complement(seq)) for sid, seq in sequences]
        n_before = len(sequences)
        sequences = sequences + rc_seqs
        print(f"  RC doubling: {n_before} → {len(sequences)} sequences")

    # Data statistics report
    seq_lengths = np.array([len(s) for _, s in sequences])
    all_concat = "".join(s for _, s in sequences)
    total_bp = len(all_concat)
    nuc_counts = {b: all_concat.count(b) for b in "ATCGN"}
    gc_content = (nuc_counts["G"] + nuc_counts["C"]) / max(total_bp, 1)
    n_content = nuc_counts["N"] / max(total_bp, 1)

    print(f"  Length range: {seq_lengths.min()} - {seq_lengths.max()} bp")
    print(f"  Mean length: {seq_lengths.mean():.0f} bp (median: {np.median(seq_lengths):.0f})")
    print(f"  Total: {total_bp:,} bp | GC: {gc_content:.1%} | N: {n_content:.1%}")

    # Save data report
    data_report = {
        "n_sequences": len(sequences),
        "total_bp": total_bp,
        "length_min": int(seq_lengths.min()),
        "length_max": int(seq_lengths.max()),
        "length_mean": float(seq_lengths.mean()),
        "length_median": float(np.median(seq_lengths)),
        "gc_content": round(gc_content, 4),
        "n_content": round(n_content, 4),
        "nucleotide_counts": nuc_counts,
    }
    with open(os.path.join(CACHE_DIR, "data_report.json"), "w") as f:
        json.dump(data_report, f, indent=2)

    # Create tokenizer
    tokenizer = create_tokenizer(tokenizer_type, kmer_size, bpe_vocab_size)

    # Train BPE if needed
    if tokenizer_type == "bpe":
        print(f"Training BPE tokenizer (vocab_size={bpe_vocab_size})...")
        tokenizer.train([s for _, s in sequences], vocab_size=bpe_vocab_size)

    # Tokenize sequences
    print(f"Tokenizing with {tokenizer_type} (vocab_size={tokenizer.vocab_size})...")
    tokenized = []
    for sid, seq in sequences:
        tokens = tokenizer.encode(seq)
        tokenized.append((sid, tokens))

    # Load labels if needed
    labels_map = {}
    target_names = None
    n_classes = None

    if task_type in ("classify", "regress") and labels_path:
        import csv
        with open(labels_path, newline="") as f:
            reader = csv.DictReader(f)
            if label_col is None:
                # Use first column that isn't an ID column
                for col in reader.fieldnames:
                    if col.lower() not in ("id", "name", "seq_id"):
                        label_col = col
                        break
            for row in reader:
                key = row.get(id_col or "id", row.get("id", ""))
                labels_map[key] = row[label_col]

    elif task_type in ("classify", "regress") and seq_col:
        # Labels might be in the same CSV
        import csv
        with open(seq_path, newline="") as f:
            reader = csv.DictReader(f)
            if label_col is None:
                for col in reader.fieldnames:
                    if col not in (seq_col, id_col, "id"):
                        label_col = col
                        break
            if label_col:
                f.seek(0)
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    key = row.get(id_col, f"seq_{i}") if id_col else f"seq_{i}"
                    labels_map[key] = row[label_col]

    # Process labels for classification
    if task_type == "classify" and labels_map:
        unique_labels = sorted(set(labels_map.values()))
        label_to_id = {l: i for i, l in enumerate(unique_labels)}
        labels_map = {k: label_to_id[v] for k, v in labels_map.items()}
        target_names = unique_labels
        n_classes = len(unique_labels)
        # Warn about small classes
        from collections import Counter
        label_counts = Counter(labels_map.values())
        small_classes = [target_names[lid] for lid, cnt in label_counts.items() if cnt < 3]
        if small_classes:
            print(f"  Warning: {len(small_classes)} class(es) have < 3 samples: {small_classes[:5]}")
        if n_classes < 2:
            raise ValueError("Classification requires at least 2 classes.")
    elif task_type == "regress" and labels_map:
        labels_map = {k: float(v) for k, v in labels_map.items()}
        n_classes = 1

    # Create samples (with chunking for pretrain)
    all_token_ids = []
    all_labels = []

    if task_type == "pretrain":
        for sid, tokens in tokenized:
            chunks = _chunk_tokens(tokens, max_length, strategy=chunk_strategy)
            for chunk in chunks:
                all_token_ids.append(chunk)
        print(f"  Created {len(all_token_ids)} chunks from {len(tokenized)} sequences")
    else:
        # For classify/regress: chunk long sequences, each chunk inherits the label.
        # This allows the model to see patterns across the full genome.
        n_chunked = 0
        for sid, tokens in tokenized:
            label = None
            if labels_map:
                label = labels_map.get(sid, 0 if task_type == "classify" else 0.0)

            if len(tokens) > max_length:
                chunks = _chunk_tokens(tokens, max_length, strategy=chunk_strategy)
                n_chunked += 1
            else:
                chunks = [tokens]

            for chunk in chunks:
                all_token_ids.append(chunk)
                if label is not None:
                    all_labels.append(label)
        if n_chunked > 0:
            print(f"  Chunked {n_chunked}/{len(tokenized)} long sequences → {len(all_token_ids)} samples")

    # Pad sequences to uniform length
    actual_max_len = min(max_length, max(len(t) for t in all_token_ids))
    padded = np.full((len(all_token_ids), actual_max_len), PAD_TOKEN_ID, dtype=np.int64)
    attention_mask = np.zeros((len(all_token_ids), actual_max_len), dtype=np.int64)

    for i, tokens in enumerate(all_token_ids):
        length = min(len(tokens), actual_max_len)
        padded[i, :length] = tokens[:length]
        attention_mask[i, :length] = 1

    # Per-sample metadata for bias analysis (T100, T101)
    sample_lengths = np.array([min(len(t), actual_max_len) for t in all_token_ids], dtype=np.int32)
    # Compute per-sample GC content from token IDs (for char tokenizer: G=7+5=12..depends on token mapping)
    # Simpler: compute from padded tokens using the tokenizer's decode
    sample_gc = np.zeros(len(all_token_ids), dtype=np.float32)
    if tokenizer_type == "char":
        # CharTokenizer: C=7, G=8
        gc_ids = np.array([7, 8])
        for i in range(len(all_token_ids)):
            toks = padded[i, :sample_lengths[i]]
            gc_count = np.isin(toks, gc_ids).sum()
            sample_gc[i] = gc_count / max(sample_lengths[i], 1)
    else:
        # For kmer/bpe: decode tokens then count GC
        for i in range(len(all_token_ids)):
            try:
                decoded = tokenizer.decode(padded[i, :sample_lengths[i]].tolist())
                gc_count = decoded.count("G") + decoded.count("C")
                sample_gc[i] = gc_count / max(len(decoded), 1)
            except Exception:
                pass

    torch.save(torch.from_numpy(sample_lengths), os.path.join(CACHE_DIR, "sample_lengths.pt"))
    torch.save(torch.from_numpy(sample_gc), os.path.join(CACHE_DIR, "sample_gc.pt"))

    # Train/val split
    from sklearn.model_selection import train_test_split

    n = len(padded)
    indices = np.arange(n)

    if task_type == "classify" and all_labels:
        labels_arr = np.array(all_labels, dtype=np.int64)
        train_idx, val_idx = train_test_split(
            indices, test_size=VAL_RATIO, random_state=RANDOM_SEED,
            stratify=labels_arr,
        )
    else:
        train_idx, val_idx = train_test_split(
            indices, test_size=VAL_RATIO, random_state=RANDOM_SEED,
        )

    train_tokens = padded[train_idx]
    val_tokens = padded[val_idx]
    train_mask = attention_mask[train_idx]
    val_mask = attention_mask[val_idx]

    # Save tensors
    torch.save(torch.from_numpy(train_tokens), os.path.join(CACHE_DIR, "train_tokens.pt"))
    torch.save(torch.from_numpy(val_tokens), os.path.join(CACHE_DIR, "val_tokens.pt"))
    torch.save(torch.from_numpy(train_mask), os.path.join(CACHE_DIR, "train_mask.pt"))
    torch.save(torch.from_numpy(val_mask), os.path.join(CACHE_DIR, "val_mask.pt"))

    if task_type in ("classify", "regress") and all_labels:
        if task_type == "classify":
            labels_arr = np.array(all_labels, dtype=np.int64)
        else:
            labels_arr = np.array(all_labels, dtype=np.float32)
        train_labels = labels_arr[train_idx]
        val_labels = labels_arr[val_idx]
        torch.save(torch.from_numpy(train_labels), os.path.join(CACHE_DIR, "train_labels.pt"))
        torch.save(torch.from_numpy(val_labels), os.path.join(CACHE_DIR, "val_labels.pt"))

    # K-fold cross-validation indices (optional)
    if n_folds > 1:
        from sklearn.model_selection import KFold, StratifiedKFold
        if task_type == "classify" and all_labels:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
            splits = list(kf.split(padded, np.array(all_labels, dtype=np.int64)))
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
            splits = list(kf.split(padded))
        fold_indices = {"n_folds": n_folds, "folds": []}
        for fold_i, (tr_idx, vl_idx) in enumerate(splits):
            fold_indices["folds"].append({
                "train": tr_idx.tolist(),
                "val": vl_idx.tolist(),
            })
        fold_path = os.path.join(CACHE_DIR, "fold_indices.json")
        with open(fold_path, "w") as f:
            json.dump(fold_indices, f)
        print(f"  Saved {n_folds}-fold CV indices to {fold_path}")

    # Save tokenizer
    tokenizer.save(os.path.join(CACHE_DIR, "tokenizer.json"))

    # Save config
    config = {
        "source": os.path.abspath(seq_path),
        "task_type": task_type,
        "tokenizer_type": tokenizer_type,
        "vocab_size": tokenizer.vocab_size,
        "max_length": actual_max_len,
        "n_train": int(len(train_tokens)),
        "n_val": int(len(val_tokens)),
        "n_sequences": len(sequences),
        "kmer_size": kmer_size if tokenizer_type == "kmer" else None,
        "n_folds": n_folds,
        "chunk_strategy": chunk_strategy,
    }

    if task_type == "classify":
        config["n_classes"] = n_classes
        config["target_names"] = target_names
        # Compute class weights for imbalanced data
        train_label_counts = np.bincount(train_labels.astype(int), minlength=n_classes)
        total = train_label_counts.sum()
        class_weights = (total / (n_classes * train_label_counts.clip(min=1))).tolist()
        config["class_weights"] = class_weights
    elif task_type == "regress":
        config["n_classes"] = 1
        if all_labels:
            train_labels_f = np.array(all_labels, dtype=np.float32)[train_idx]
            config["target_mean"] = float(train_labels_f.mean())
            config["target_std"] = float(train_labels_f.std() + 1e-8)

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    # Print summary
    print(f"\nTask type:      {task_type}")
    print(f"Tokenizer:      {tokenizer_type} (vocab_size={tokenizer.vocab_size})")
    print(f"Max length:     {actual_max_len} tokens")
    print(f"Train samples:  {len(train_tokens)}")
    print(f"Val samples:    {len(val_tokens)}")
    if task_type == "classify":
        print(f"Classes:        {n_classes} ({', '.join(target_names[:5])}{'...' if len(target_names) > 5 else ''})")
    print(f"Saved to:       {CACHE_DIR}")

    return config


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def load_config():
    """Load task config from cache."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"No task config found at {CONFIG_PATH}.\n"
            f"Run prepare.py first:\n"
            f"  python prepare.py --fasta <file> --task pretrain"
        )
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    required = ["task_type", "vocab_size", "max_length", "n_train", "n_val"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Invalid task config: missing keys {missing}. Re-run prepare.py.")
    return config


def load_data(device="cpu"):
    """Load cached train/val tensors."""
    train_tokens = torch.load(os.path.join(CACHE_DIR, "train_tokens.pt"), map_location=device, weights_only=True)
    val_tokens = torch.load(os.path.join(CACHE_DIR, "val_tokens.pt"), map_location=device, weights_only=True)
    train_mask = torch.load(os.path.join(CACHE_DIR, "train_mask.pt"), map_location=device, weights_only=True)
    val_mask = torch.load(os.path.join(CACHE_DIR, "val_mask.pt"), map_location=device, weights_only=True)

    data = {
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_mask": train_mask,
        "val_mask": val_mask,
    }

    # Load labels if they exist
    labels_path = os.path.join(CACHE_DIR, "train_labels.pt")
    if os.path.exists(labels_path):
        data["train_labels"] = torch.load(labels_path, map_location=device, weights_only=True)
        data["val_labels"] = torch.load(os.path.join(CACHE_DIR, "val_labels.pt"), map_location=device, weights_only=True)

    return data


def load_fold(fold_idx, device="cpu"):
    """Load train/val data for a specific k-fold CV split.

    Returns the same dict structure as load_data(), but with fold-specific
    train/val indices applied to the full dataset.

    Args:
        fold_idx: 0-based fold index.
        device: torch device for tensors.
    """
    fold_path = os.path.join(CACHE_DIR, "fold_indices.json")
    if not os.path.exists(fold_path):
        raise FileNotFoundError(
            "No fold indices found. Re-run prepare.py with --n-folds > 1."
        )
    with open(fold_path) as f:
        fold_info = json.load(f)

    n_folds = fold_info["n_folds"]
    if fold_idx < 0 or fold_idx >= n_folds:
        raise ValueError(f"fold_idx={fold_idx} out of range [0, {n_folds})")

    train_idx = fold_info["folds"][fold_idx]["train"]
    val_idx = fold_info["folds"][fold_idx]["val"]

    # Load full dataset (train + val combined)
    all_tokens = torch.cat([
        torch.load(os.path.join(CACHE_DIR, "train_tokens.pt"), map_location=device, weights_only=True),
        torch.load(os.path.join(CACHE_DIR, "val_tokens.pt"), map_location=device, weights_only=True),
    ])
    all_mask = torch.cat([
        torch.load(os.path.join(CACHE_DIR, "train_mask.pt"), map_location=device, weights_only=True),
        torch.load(os.path.join(CACHE_DIR, "val_mask.pt"), map_location=device, weights_only=True),
    ])

    data = {
        "train_tokens": all_tokens[train_idx],
        "val_tokens": all_tokens[val_idx],
        "train_mask": all_mask[train_idx],
        "val_mask": all_mask[val_idx],
    }

    # Load labels if they exist
    train_labels_path = os.path.join(CACHE_DIR, "train_labels.pt")
    val_labels_path = os.path.join(CACHE_DIR, "val_labels.pt")
    if os.path.exists(train_labels_path) and os.path.exists(val_labels_path):
        all_labels = torch.cat([
            torch.load(train_labels_path, map_location=device, weights_only=True),
            torch.load(val_labels_path, map_location=device, weights_only=True),
        ])
        data["train_labels"] = all_labels[train_idx]
        data["val_labels"] = all_labels[val_idx]

    return data


def make_dataloader(tokens, mask, batch_size, shuffle=True, drop_last=False, labels=None,
                    num_workers=0, pin_memory=False):
    """Create a PyTorch DataLoader from token tensors.

    Args:
        num_workers: Number of data loading workers (0 = main process only).
        pin_memory: Pin memory for faster CUDA transfer (set True for GPU training).
    """
    if labels is not None:
        dataset = TensorDataset(tokens, mask, labels)
    else:
        dataset = TensorDataset(tokens, mask)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


class StreamingSequenceDataset(torch.utils.data.IterableDataset):
    """Streaming dataset that reads FASTA/FASTQ on-the-fly for large datasets.

    Tokenizes and chunks sequences lazily, yielding (token_ids, attention_mask) tuples.
    Supports shuffling via a small buffer (reservoir sampling).
    """

    def __init__(self, path, tokenizer, max_length, task_type="pretrain",
                 chunk_overlap=0.5, buffer_size=10000):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.chunk_overlap = chunk_overlap
        self.buffer_size = buffer_size

    def _tokenize_and_chunk(self, sequence):
        """Tokenize a sequence and split into fixed-length chunks."""
        token_ids = self.tokenizer.encode(sequence)
        if len(token_ids) <= self.max_length - 2:
            # Fits in one chunk: [CLS] + tokens + [SEP] + padding
            ids = [CLS_TOKEN_ID] + token_ids + [SEP_TOKEN_ID]
            pad_len = self.max_length - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [PAD_TOKEN_ID] * pad_len
            yield ids, mask
        else:
            # Split into overlapping chunks
            step = max(1, int(self.max_length * (1 - self.chunk_overlap)))
            usable = self.max_length - 2
            for start in range(0, len(token_ids) - usable + 1, step):
                chunk = token_ids[start:start + usable]
                ids = [CLS_TOKEN_ID] + chunk + [SEP_TOKEN_ID]
                pad_len = self.max_length - len(ids)
                mask = [1] * len(ids) + [0] * pad_len
                ids = ids + [PAD_TOKEN_ID] * pad_len
                yield ids, mask

    def __iter__(self):
        from Bio import SeqIO
        ext = Path(self.path).suffix.lower()
        fmt = "fastq" if ext in (".fastq", ".fq") else "fasta"

        buffer = []
        for record in SeqIO.parse(self.path, fmt):
            seq = _clean_sequence(str(record.seq))
            if len(seq) == 0:
                continue
            for ids, mask in self._tokenize_and_chunk(seq):
                if len(buffer) < self.buffer_size:
                    buffer.append((
                        torch.tensor(ids, dtype=torch.long),
                        torch.tensor(mask, dtype=torch.long),
                    ))
                else:
                    # Reservoir sampling: replace random element
                    import random as _rand
                    idx = _rand.randint(0, len(buffer) - 1)
                    yield buffer[idx]
                    buffer[idx] = (
                        torch.tensor(ids, dtype=torch.long),
                        torch.tensor(mask, dtype=torch.long),
                    )

        # Flush remaining buffer (shuffled)
        import random as _rand
        _rand.shuffle(buffer)
        for item in buffer:
            yield item


def make_streaming_dataloader(path, tokenizer, max_length, batch_size,
                              task_type="pretrain", num_workers=0, pin_memory=False):
    """Create a streaming DataLoader for large datasets.

    Reads FASTA/FASTQ files lazily — never loads entire dataset into memory.
    """
    dataset = StreamingSequenceDataset(path, tokenizer, max_length, task_type)
    return DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — fixed metrics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, task_type, config, objective="mlm", batch_size=64, device="cpu", mask_ratio=0.15):
    """
    Evaluate model on validation set.

    Pre-training (MLM): val_score = -perplexity
    Pre-training (CLM): val_score = -perplexity
    Classification:     val_score = accuracy
    Regression:         val_score = -MSE

    Returns dict with all metrics.
    """
    model.eval()
    val_tokens = data["val_tokens"]
    val_mask = data["val_mask"]
    vocab_size = config["vocab_size"]

    # Handle empty or very small validation set
    if len(val_tokens) == 0:
        print("  Warning: empty validation set, skipping evaluation")
        return {"val_score": 0.0, "val_loss": float("inf")}
    if len(val_tokens) < batch_size:
        batch_size = max(1, len(val_tokens))

    if task_type == "pretrain":
        return _evaluate_pretrain(model, val_tokens, val_mask, vocab_size, objective, batch_size, device, mask_ratio)
    elif task_type == "classify":
        val_labels = data["val_labels"]
        return _evaluate_classify(model, val_tokens, val_mask, val_labels, batch_size, device, config)
    elif task_type == "regress":
        val_labels = data["val_labels"]
        return _evaluate_regress(model, val_tokens, val_mask, val_labels, batch_size, device)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def _evaluate_pretrain(model, val_tokens, val_mask, vocab_size, objective, batch_size, device, mask_ratio):
    """Evaluate pre-training (MLM or CLM)."""
    loader = DataLoader(TensorDataset(val_tokens, val_mask), batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=PAD_TOKEN_ID)

    for tokens_batch, mask_batch in loader:
        tokens_batch = tokens_batch.to(device)
        mask_batch = mask_batch.to(device)

        if objective == "mlm":
            # Create MLM targets
            input_ids = tokens_batch.clone()
            labels = tokens_batch.clone()

            # Mask random positions (only where attention_mask == 1 and not special tokens)
            mask_candidates = (mask_batch == 1) & (tokens_batch >= NUM_SPECIAL)
            rand = torch.rand_like(tokens_batch.float())
            mask_positions = mask_candidates & (rand < mask_ratio)

            # Replace masked positions with [MASK]
            input_ids[mask_positions] = MASK_TOKEN_ID
            # Only compute loss on masked positions
            labels[~mask_positions] = PAD_TOKEN_ID

            logits = model(input_ids, attention_mask=mask_batch)  # (B, L, V)
            loss_per_token = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))

            valid = labels.reshape(-1) != PAD_TOKEN_ID
            loss_sum = loss_per_token[valid].sum().item()
            n_valid = valid.sum().item()

            # Token accuracy on masked positions
            if n_valid > 0:
                preds = logits.reshape(-1, vocab_size).argmax(dim=-1)
                correct_tokens += (preds[valid] == labels.reshape(-1)[valid]).sum().item()

        else:  # CLM
            input_ids = tokens_batch[:, :-1]
            target_ids = tokens_batch[:, 1:]
            input_mask = mask_batch[:, :-1]
            target_mask = mask_batch[:, 1:]

            logits = model(input_ids, attention_mask=input_mask)  # (B, L-1, V)
            loss_per_token = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

            valid = target_mask.reshape(-1) == 1
            loss_sum = loss_per_token[valid].sum().item()
            n_valid = valid.sum().item()

        if n_valid > 0:
            total_loss += loss_sum
            total_tokens += n_valid

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))  # cap to avoid overflow

    result = {
        "val_score": -perplexity,
        "val_perplexity": perplexity,
        "val_loss": avg_loss,
        "metric_direction": "higher_is_better",
    }

    if objective == "mlm" and total_tokens > 0:
        result["val_token_accuracy"] = correct_tokens / total_tokens

    return result


def _evaluate_classify(model, val_tokens, val_mask, val_labels, batch_size, device, config):
    """Evaluate classification task."""
    loader = DataLoader(TensorDataset(val_tokens, val_mask, val_labels), batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    all_targets = []

    for tokens_batch, mask_batch, labels_batch in loader:
        tokens_batch = tokens_batch.to(device)
        mask_batch = mask_batch.to(device)
        labels_batch = labels_batch.to(device)

        logits = model(tokens_batch, attention_mask=mask_batch)  # (B, n_classes)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        all_targets.append(labels_batch.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    accuracy = float((all_preds == all_targets).mean())
    n_classes = config.get("n_classes", len(np.unique(all_targets)))

    # Per-class metrics
    per_class = {}
    for c in range(n_classes):
        tp = ((all_preds == c) & (all_targets == c)).sum()
        fp = ((all_preds == c) & (all_targets != c)).sum()
        fn = ((all_preds != c) & (all_targets == c)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int((all_targets == c).sum())

        per_class[c] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }

    macro_precision = np.mean([v["precision"] for v in per_class.values()])
    macro_recall = np.mean([v["recall"] for v in per_class.values()])
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])

    total_support = sum(v["support"] for v in per_class.values())
    weighted_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / total_support if total_support > 0 else 0.0

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for pred, true in zip(all_preds, all_targets):
        cm[int(true), int(pred)] += 1

    # ROC-AUC (one-vs-rest for multi-class)
    roc_auc = None
    try:
        from sklearn.metrics import roc_auc_score
        if n_classes == 2:
            roc_auc = float(roc_auc_score(all_targets, all_probs[:, 1]))
        elif n_classes > 2 and len(np.unique(all_targets)) > 1:
            roc_auc = float(roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro"))
    except (ValueError, ImportError):
        pass

    result = {
        "val_score": accuracy,
        "val_accuracy": accuracy,
        "val_f1_macro": float(macro_f1),
        "val_f1_weighted": float(weighted_f1),
        "val_precision_macro": float(macro_precision),
        "val_recall_macro": float(macro_recall),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "targets": all_targets,
        "probabilities": all_probs,
        "metric_direction": "higher_is_better",
    }
    if roc_auc is not None:
        result["val_roc_auc"] = roc_auc
    return result


def _evaluate_regress(model, val_tokens, val_mask, val_labels, batch_size, device):
    """Evaluate regression task."""
    loader = DataLoader(TensorDataset(val_tokens, val_mask, val_labels), batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []

    for tokens_batch, mask_batch, labels_batch in loader:
        tokens_batch = tokens_batch.to(device)
        mask_batch = mask_batch.to(device)
        labels_batch = labels_batch.to(device)

        preds = model(tokens_batch, attention_mask=mask_batch).squeeze(-1)
        all_preds.append(preds.cpu())
        all_targets.append(labels_batch.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    residuals = all_preds - all_targets
    mse = float((residuals ** 2).mean())
    rmse = float(math.sqrt(mse))
    mae = float(np.abs(residuals).mean())

    ss_res = float((residuals ** 2).sum())
    y_mean = float(all_targets.mean())
    ss_tot = float(((all_targets - y_mean) ** 2).sum())
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Spearman rank correlation
    spearman_r = None
    try:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(all_preds, all_targets)
        if not np.isnan(corr):
            spearman_r = float(corr)
    except ImportError:
        pass

    # Pearson correlation
    if len(all_targets) > 1 and np.std(all_targets) > 0 and np.std(all_preds) > 0:
        pearson_r = float(np.corrcoef(all_preds, all_targets)[0, 1])
    else:
        pearson_r = 0.0

    result = {
        "val_score": -mse,
        "val_mse": mse,
        "val_rmse": rmse,
        "val_mae": mae,
        "val_r2": float(r2),
        "val_pearson_r": pearson_r,
        "predictions": all_preds,
        "targets": all_targets,
        "metric_direction": "higher_is_better",
    }
    if spearman_r is not None:
        result["val_spearman_r"] = spearman_r
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results, task_type, config, training_history=None,
                    report_dir="reports", run_info=None, embeddings=None, embed_labels=None,
                    attention_weights=None, model_weights=None):
    """Generate experiment report with plots and metrics.

    Args:
        embeddings: Optional (N, D) numpy array of sequence embeddings for t-SNE.
        embed_labels: Optional (N,) labels for coloring t-SNE points.
        attention_weights: Optional (L, L) attention weight matrix for heatmap.
        model_weights: Optional dict of {name: numpy_array} for weight distribution plots.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(report_dir, exist_ok=True)

    target_names = config.get("target_names", [])
    preds = results.get("predictions")
    targets = results.get("targets")

    # --- Training loss curve ---
    if training_history and "steps" in training_history and "losses" in training_history:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        steps = training_history["steps"]
        losses = training_history["losses"]

        ax1.plot(steps, losses, color="#2196F3", alpha=0.7, linewidth=0.8, label="Train Loss")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss", color="#2196F3")
        ax1.tick_params(axis="y", labelcolor="#2196F3")
        ax1.set_title("Training Loss & Learning Rate")

        if "lrs" in training_history:
            ax2 = ax1.twinx()
            ax2.plot(steps, training_history["lrs"], color="#FF9800", alpha=0.7, linewidth=0.8, label="Learning Rate")
            ax2.set_ylabel("Learning Rate", color="#FF9800")
            ax2.tick_params(axis="y", labelcolor="#FF9800")

        fig.tight_layout()
        fig.savefig(os.path.join(report_dir, "training_curve.png"), dpi=150)
        plt.close(fig)

    # --- Validation score curve ---
    if training_history and "eval_steps" in training_history and "eval_scores" in training_history:
        eval_steps = training_history["eval_steps"]
        eval_scores = training_history["eval_scores"]
        if len(eval_steps) > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(eval_steps, eval_scores, "o-", color="#4CAF50", markersize=5)
            ax.set_xlabel("Step")
            ax.set_ylabel("val_score")
            ax.set_title("Validation Score During Training")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(report_dir, "val_score_curve.png"), dpi=150)
            plt.close(fig)

    # --- Pre-training specific plots ---
    if task_type == "pretrain":
        _generate_pretrain_report(results, training_history, report_dir)
    elif task_type == "classify":
        probs = results.get("probabilities")
        _generate_classification_report(results, config, target_names, preds, targets, report_dir, probs)
    elif task_type == "regress":
        _generate_regression_report(results, config, preds, targets, report_dir)

    # --- Attention weight heatmap (T93) ---
    if attention_weights is not None:
        try:
            L = min(attention_weights.shape[0], 128)  # Cap at 128 for readability
            attn_crop = attention_weights[:L, :L]
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(attn_crop, cmap="viridis", aspect="auto")
            ax.set_title("Attention Weights (Layer 0, Head Average)")
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(os.path.join(report_dir, "attention_map.png"), dpi=150)
            plt.close(fig)
        except Exception:
            pass

    # --- Embedding t-SNE visualization ---
    if embeddings is not None and len(embeddings) > 10:
        try:
            from sklearn.manifold import TSNE
            perp = min(30, len(embeddings) - 1)
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
            coords = tsne.fit_transform(embeddings)

            fig, ax = plt.subplots(figsize=(8, 8))
            if embed_labels is not None:
                unique_labels = np.unique(embed_labels)
                colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_labels), 10)))
                for i, label in enumerate(unique_labels[:10]):
                    mask = embed_labels == label
                    lbl = target_names[int(label)] if int(label) < len(target_names) else str(label)
                    ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i % 10]],
                              label=lbl, alpha=0.6, s=15, edgecolors="none")
                ax.legend(fontsize=8, markerscale=2)
            else:
                ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=15,
                          color="#2196F3", edgecolors="none")
            ax.set_title("Embedding t-SNE")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(report_dir, "embedding_tsne.png"), dpi=150)
            plt.close(fig)
        except (ImportError, Exception):
            pass

    # --- Embedding PCA visualization ---
    if embeddings is not None and len(embeddings) > 10:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(embeddings)
            ev_ratio = pca.explained_variance_ratio_

            fig, ax = plt.subplots(figsize=(8, 8))
            if embed_labels is not None:
                unique_labels = np.unique(embed_labels)
                colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_labels), 10)))
                for i, label in enumerate(unique_labels[:10]):
                    mask = embed_labels == label
                    lbl = target_names[int(label)] if int(label) < len(target_names) else str(label)
                    ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i % 10]],
                              label=lbl, alpha=0.6, s=15, edgecolors="none")
                ax.legend(fontsize=8, markerscale=2)
            else:
                ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=15,
                          color="#2196F3", edgecolors="none")
            ax.set_title("Embedding PCA")
            ax.set_xlabel(f"PC1 ({ev_ratio[0]:.1%} var)")
            ax.set_ylabel(f"PC2 ({ev_ratio[1]:.1%} var)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(report_dir, "embedding_pca.png"), dpi=150)
            plt.close(fig)
        except (ImportError, Exception):
            pass

    # --- Embedding UMAP visualization (T95, optional) ---
    if embeddings is not None and len(embeddings) > 10:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings) - 1))
            coords = reducer.fit_transform(embeddings)

            fig, ax = plt.subplots(figsize=(8, 8))
            if embed_labels is not None:
                unique_labels = np.unique(embed_labels)
                colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_labels), 10)))
                for i, label in enumerate(unique_labels[:10]):
                    lmask = embed_labels == label
                    lbl = target_names[int(label)] if int(label) < len(target_names) else str(label)
                    ax.scatter(coords[lmask, 0], coords[lmask, 1], c=[colors[i % 10]],
                              label=lbl, alpha=0.6, s=15, edgecolors="none")
                ax.legend(fontsize=8, markerscale=2)
            else:
                ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=15,
                          color="#2196F3", edgecolors="none")
            ax.set_title("Embedding UMAP")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(report_dir, "embedding_umap.png"), dpi=150)
            plt.close(fig)
        except ImportError:
            pass  # umap-learn not installed, skip silently
        except Exception:
            pass

    # --- Gradient norm plot ---
    if training_history and "grad_norms" in training_history and training_history["grad_norms"]:
        try:
            gn_data = training_history["grad_norms"]
            # Group norms by layer name prefix
            layer_groups = {}
            for entry in gn_data:
                for name, norm in entry["norms"].items():
                    # Simplify name to layer group
                    parts = name.split(".")
                    group = ".".join(parts[:3]) if len(parts) > 3 else name
                    layer_groups.setdefault(group, {"steps": [], "norms": []})
                    layer_groups[group]["steps"].append(entry["step"])
                    layer_groups[group]["norms"].append(norm)

            if layer_groups:
                fig, ax = plt.subplots(figsize=(12, 6))
                # Plot top 10 layer groups by mean norm
                sorted_groups = sorted(layer_groups.items(),
                                       key=lambda x: np.mean(x[1]["norms"]), reverse=True)
                colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(sorted_groups))))
                for i, (group, data) in enumerate(sorted_groups[:10]):
                    ax.plot(data["steps"], data["norms"], alpha=0.7, linewidth=1,
                           color=colors[i], label=group)
                ax.set_xlabel("Step")
                ax.set_ylabel("Gradient L2 Norm")
                ax.set_title("Per-Layer Gradient Norms")
                ax.legend(fontsize=6, loc="upper right")
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(report_dir, "gradient_norms.png"), dpi=150)
                plt.close(fig)
        except Exception:
            pass

    # --- GC content bias analysis (T100) ---
    gc_path = os.path.join(CACHE_DIR, "sample_gc.pt")
    if os.path.exists(gc_path) and results.get("predictions") is not None and results.get("targets") is not None:
        try:
            all_gc = torch.load(gc_path, map_location="cpu", weights_only=True).numpy()
            val_n = config.get("n_val", 0)
            # val samples are the last n_val in the original data
            if len(all_gc) >= val_n and val_n > 0:
                val_gc = all_gc[-val_n:]
                preds_arr = np.array(results["predictions"])
                targets_arr = np.array(results["targets"])
                n_bins = min(5, len(val_gc) // 5)
                if n_bins >= 2:
                    bins = np.quantile(val_gc, np.linspace(0, 1, n_bins + 1))
                    bins[-1] += 0.001
                    bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(n_bins)]
                    bin_accs = []
                    for i in range(n_bins):
                        mask = (val_gc >= bins[i]) & (val_gc < bins[i + 1])
                        if mask.sum() > 0:
                            if task_type == "classify":
                                bin_accs.append(float((preds_arr[mask] == targets_arr[mask]).mean()))
                            else:
                                bin_accs.append(float(-np.mean((preds_arr[mask] - targets_arr[mask]) ** 2)))
                        else:
                            bin_accs.append(0)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.bar(range(n_bins), bin_accs, color="#4CAF50", alpha=0.8)
                    ax.set_xticks(range(n_bins))
                    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
                    ax.set_xlabel("GC Content Range")
                    ax.set_ylabel("Accuracy" if task_type == "classify" else "-MSE")
                    ax.set_title("Performance by GC Content")
                    ax.grid(axis="y", alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(os.path.join(report_dir, "gc_bias.png"), dpi=150)
                    plt.close(fig)
        except Exception:
            pass

    # --- Sequence length bias analysis (T101) ---
    len_path = os.path.join(CACHE_DIR, "sample_lengths.pt")
    if os.path.exists(len_path) and results.get("predictions") is not None and results.get("targets") is not None:
        try:
            all_lens = torch.load(len_path, map_location="cpu", weights_only=True).numpy()
            val_n = config.get("n_val", 0)
            if len(all_lens) >= val_n and val_n > 0:
                val_lens = all_lens[-val_n:]
                preds_arr = np.array(results["predictions"])
                targets_arr = np.array(results["targets"])
                n_bins = min(5, len(val_lens) // 5)
                if n_bins >= 2:
                    bins = np.quantile(val_lens.astype(float), np.linspace(0, 1, n_bins + 1))
                    bins[-1] += 1
                    bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(n_bins)]
                    bin_accs = []
                    for i in range(n_bins):
                        mask = (val_lens >= bins[i]) & (val_lens < bins[i + 1])
                        if mask.sum() > 0:
                            if task_type == "classify":
                                bin_accs.append(float((preds_arr[mask] == targets_arr[mask]).mean()))
                            else:
                                bin_accs.append(float(-np.mean((preds_arr[mask] - targets_arr[mask]) ** 2)))
                        else:
                            bin_accs.append(0)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.bar(range(n_bins), bin_accs, color="#2196F3", alpha=0.8)
                    ax.set_xticks(range(n_bins))
                    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
                    ax.set_xlabel("Sequence Length (tokens)")
                    ax.set_ylabel("Accuracy" if task_type == "classify" else "-MSE")
                    ax.set_title("Performance by Sequence Length")
                    ax.grid(axis="y", alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(os.path.join(report_dir, "length_bias.png"), dpi=150)
                    plt.close(fig)
        except Exception:
            pass

    # --- Weight distribution analysis (T106) ---
    if model_weights:
        try:
            # Group weights by top-level module
            groups = {}
            for name, w in model_weights.items():
                group = name.split(".")[0]
                groups.setdefault(group, []).append(w.flatten())

            n_groups = min(len(groups), 9)
            if n_groups >= 1:
                sorted_groups = sorted(groups.items(), key=lambda x: sum(len(a) for a in x[1]), reverse=True)
                ncols = min(3, n_groups)
                nrows = (n_groups + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
                if n_groups == 1:
                    axes = np.array([axes])
                axes = axes.flatten()

                for i, (group, arrays) in enumerate(sorted_groups[:n_groups]):
                    all_w = np.concatenate(arrays)
                    ax = axes[i]
                    ax.hist(all_w, bins=50, color="#2196F3", alpha=0.7, edgecolor="none")
                    ax.set_title(f"{group} ({len(all_w):,})", fontsize=9)
                    ax.set_xlabel("Value", fontsize=8)
                    ax.tick_params(labelsize=7)

                for i in range(n_groups, len(axes)):
                    axes[i].set_visible(False)

                fig.suptitle("Weight Distributions by Layer Group", fontsize=11)
                fig.tight_layout()
                fig.savefig(os.path.join(report_dir, "weight_distribution.png"), dpi=150)
                plt.close(fig)
        except Exception:
            pass

    # --- Statistical significance (T108) ---
    if results.get("predictions") is not None and results.get("targets") is not None:
        try:
            preds_arr = np.array(results["predictions"])
            targets_arr = np.array(results["targets"])
            n_bootstrap = 1000
            rng = np.random.RandomState(42)

            if task_type == "classify":
                # Bootstrap accuracy CI
                accs = []
                for _ in range(n_bootstrap):
                    idx = rng.choice(len(preds_arr), size=len(preds_arr), replace=True)
                    accs.append(float((preds_arr[idx] == targets_arr[idx]).mean()))
                ci_low, ci_high = np.percentile(accs, [2.5, 97.5])
                results["accuracy_ci_95"] = [round(ci_low, 4), round(ci_high, 4)]
            elif task_type == "regress":
                # Bootstrap MSE CI
                mses = []
                for _ in range(n_bootstrap):
                    idx = rng.choice(len(preds_arr), size=len(preds_arr), replace=True)
                    mses.append(float(np.mean((preds_arr[idx] - targets_arr[idx]) ** 2)))
                ci_low, ci_high = np.percentile(mses, [2.5, 97.5])
                results["mse_ci_95"] = [round(ci_low, 4), round(ci_high, 4)]
        except Exception:
            pass

    # --- Save metrics JSON ---
    metrics_to_save = {k: v for k, v in results.items()
                       if k not in ("predictions", "targets", "probabilities")}
    if run_info:
        metrics_to_save["run_info"] = run_info
    with open(os.path.join(report_dir, "metrics.json"), "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    print(f"\nReport saved to {report_dir}/")


def _generate_pretrain_report(results, training_history, report_dir):
    """Generate pre-training specific plots and report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Perplexity curve
    if training_history and "eval_steps" in training_history:
        eval_steps = training_history["eval_steps"]
        perplexities = training_history.get("eval_perplexities", [])
        if perplexities and len(perplexities) > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(eval_steps[:len(perplexities)], perplexities, "o-", color="#E91E63", markersize=5)
            ax.set_xlabel("Step")
            ax.set_ylabel("Perplexity")
            ax.set_title("Validation Perplexity")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(report_dir, "perplexity_curve.png"), dpi=150)
            plt.close(fig)

    # Token accuracy curve (MLM only)
    if training_history:
        token_accs = training_history.get("eval_token_accuracies", [])
        eval_steps = training_history.get("eval_steps", [])
        if token_accs and len(token_accs) > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(eval_steps[:len(token_accs)], token_accs, "o-", color="#009688", markersize=5)
            ax.set_xlabel("Step")
            ax.set_ylabel("Token Accuracy")
            ax.set_title("Masked Token Prediction Accuracy")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(report_dir, "token_accuracy_curve.png"), dpi=150)
            plt.close(fig)

    # Nucleotide confusion matrix (per-base prediction accuracy)
    per_pos_acc = results.get("per_position_accuracy")
    if per_pos_acc:
        try:
            positions = sorted(per_pos_acc.keys())
            accs = [per_pos_acc[p] for p in positions]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(positions, accs, width=1.0, color="#4CAF50", alpha=0.7)
            ax.set_xlabel("Position")
            ax.set_ylabel("Token Accuracy")
            ax.set_title("Per-Position Token Prediction Accuracy")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(os.path.join(report_dir, "per_position_accuracy.png"), dpi=150)
            plt.close(fig)
        except Exception:
            pass

    # Nucleotide-level confusion matrix
    nuc_confusion = results.get("nucleotide_confusion")
    if nuc_confusion is not None:
        try:
            cm = np.array(nuc_confusion)
            labels = ["A", "T", "C", "G", "N"][:cm.shape[0]]
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title("Nucleotide Prediction Confusion Matrix")
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(report_dir, "nucleotide_confusion.png"), dpi=150)
            plt.close(fig)
        except Exception:
            pass

    # Text report
    lines = ["Pre-training Report", "=" * 40]
    lines.append(f"Perplexity:     {results.get('val_perplexity', 'N/A'):.4f}")
    lines.append(f"Loss:           {results.get('val_loss', 'N/A'):.6f}")
    if "val_token_accuracy" in results:
        lines.append(f"Token Accuracy: {results['val_token_accuracy']:.4f}")
    lines.append(f"val_score:      {results['val_score']:.6f} (= -perplexity)")

    report_text = "\n".join(lines)
    with open(os.path.join(report_dir, "pretrain_report.txt"), "w") as f:
        f.write(report_text)
    print("\n" + report_text)


def _generate_classification_report(results, config, target_names, preds, targets, report_dir, probs=None):
    """Generate classification-specific plots and text report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(results["confusion_matrix"])
    n_classes = len(cm)

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(max(6, n_classes), max(5, n_classes * 0.8)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)

    labels = target_names[:n_classes] if len(target_names) >= n_classes else [str(i) for i in range(n_classes)]
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    short_labels = [l[:12] for l in labels]
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_yticklabels(short_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(report_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Per-class metrics
    per_class = results.get("per_class", {})
    if per_class:
        classes = sorted(per_class.keys())
        precisions = [per_class[c]["precision"] for c in classes]
        recalls = [per_class[c]["recall"] for c in classes]
        f1s = [per_class[c]["f1"] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.2), 5))
        ax.bar(x - width, precisions, width, label="Precision", color="#2196F3")
        ax.bar(x, recalls, width, label="Recall", color="#4CAF50")
        ax.bar(x + width, f1s, width, label="F1", color="#FF9800")
        ax.set_xlabel("Class")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[c] if c < len(labels) else str(c) for c in classes],
                           rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(report_dir, "per_class_metrics.png"), dpi=150)
        plt.close(fig)

    # ROC Curves (T102)
    if probs is not None and targets is not None and n_classes >= 2:
        try:
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize

            targets_arr = np.array(targets)
            probs_arr = np.array(probs)

            if n_classes == 2:
                # Binary: single ROC curve
                fpr, tpr, _ = roc_curve(targets_arr, probs_arr[:, 1])
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
                ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(report_dir, "roc_curve.png"), dpi=150)
                plt.close(fig)
            else:
                # Multi-class: one-vs-rest ROC
                targets_bin = label_binarize(targets_arr, classes=list(range(n_classes)))
                fig, ax = plt.subplots(figsize=(8, 8))
                colors = plt.cm.tab10(np.linspace(0, 1, min(n_classes, 10)))
                for i in range(min(n_classes, 10)):
                    fpr, tpr, _ = roc_curve(targets_bin[:, i], probs_arr[:, i])
                    roc_auc_i = auc(fpr, tpr)
                    lbl = labels[i] if i < len(labels) else str(i)
                    ax.plot(fpr, tpr, color=colors[i], lw=1.5,
                            label=f"{lbl} (AUC={roc_auc_i:.2f})")
                ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curves (One-vs-Rest)")
                ax.legend(fontsize=7, loc="lower right")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(report_dir, "roc_curve.png"), dpi=150)
                plt.close(fig)
        except (ImportError, Exception):
            pass

    # PR Curves (T103)
    if probs is not None and targets is not None and n_classes >= 2:
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            from sklearn.preprocessing import label_binarize

            targets_arr = np.array(targets)
            probs_arr = np.array(probs)

            if n_classes == 2:
                precision_vals, recall_vals, _ = precision_recall_curve(targets_arr, probs_arr[:, 1])
                ap = average_precision_score(targets_arr, probs_arr[:, 1])
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.plot(recall_vals, precision_vals, color="#4CAF50", lw=2,
                        label=f"PR (AP = {ap:.3f})")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall Curve")
                ax.legend(loc="lower left")
                ax.set_xlim(0, 1.05)
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(report_dir, "pr_curve.png"), dpi=150)
                plt.close(fig)
            else:
                targets_bin = label_binarize(targets_arr, classes=list(range(n_classes)))
                fig, ax = plt.subplots(figsize=(8, 8))
                colors = plt.cm.tab10(np.linspace(0, 1, min(n_classes, 10)))
                for i in range(min(n_classes, 10)):
                    prec_i, rec_i, _ = precision_recall_curve(targets_bin[:, i], probs_arr[:, i])
                    ap_i = average_precision_score(targets_bin[:, i], probs_arr[:, i])
                    lbl = labels[i] if i < len(labels) else str(i)
                    ax.plot(rec_i, prec_i, color=colors[i], lw=1.5,
                            label=f"{lbl} (AP={ap_i:.2f})")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall Curves (One-vs-Rest)")
                ax.legend(fontsize=7, loc="lower left")
                ax.set_xlim(0, 1.05)
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(report_dir, "pr_curve.png"), dpi=150)
                plt.close(fig)
        except (ImportError, Exception):
            pass

    # Calibration Plot (T104)
    if probs is not None and targets is not None and n_classes >= 2:
        try:
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import brier_score_loss

            targets_arr = np.array(targets)
            probs_arr = np.array(probs)

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")

            if n_classes == 2:
                prob_true, prob_pred = calibration_curve(targets_arr, probs_arr[:, 1], n_bins=10)
                brier = brier_score_loss(targets_arr, probs_arr[:, 1])
                ax.plot(prob_pred, prob_true, "o-", color="#2196F3",
                        label=f"Model (Brier={brier:.3f})")
            else:
                # Multi-class: one-vs-rest calibration for each class
                from sklearn.preprocessing import label_binarize
                targets_bin = label_binarize(targets_arr, classes=list(range(n_classes)))
                colors = plt.cm.tab10(np.linspace(0, 1, min(n_classes, 10)))
                for i in range(min(n_classes, 10)):
                    try:
                        prob_true, prob_pred = calibration_curve(targets_bin[:, i], probs_arr[:, i], n_bins=10)
                        lbl = labels[i] if i < len(labels) else str(i)
                        ax.plot(prob_pred, prob_true, "o-", color=colors[i], label=lbl, markersize=4)
                    except (ValueError, IndexError):
                        continue

            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title("Calibration Plot (Reliability Diagram)")
            ax.legend(fontsize=7, loc="lower right")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(report_dir, "calibration.png"), dpi=150)
            plt.close(fig)
        except (ImportError, Exception):
            pass

    # Text report
    lines = ["Classification Report", "=" * 60]
    lines.append(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 60)
    for c in sorted(per_class.keys()):
        m = per_class[c]
        label = labels[c] if c < len(labels) else str(c)
        lines.append(f"{label:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")
    lines.append("-" * 60)
    lines.append(f"{'Accuracy':<20} {'':>10} {'':>10} {results['val_accuracy']:>10.4f} {sum(v['support'] for v in per_class.values()):>10}")
    lines.append(f"{'Macro avg':<20} {results['val_precision_macro']:>10.4f} {results['val_recall_macro']:>10.4f} {results['val_f1_macro']:>10.4f}")
    lines.append(f"{'Weighted F1':<20} {'':>10} {'':>10} {results['val_f1_weighted']:>10.4f}")

    report_text = "\n".join(lines)
    with open(os.path.join(report_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)
    print("\n" + report_text)


def _generate_regression_report(results, config, preds, targets, report_dir):
    """Generate regression-specific plots and text report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Predicted vs Actual
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets, preds, alpha=0.5, s=15, color="#2196F3", edgecolors="none")
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "--", color="#F44336", alpha=0.7, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(report_dir, "predicted_vs_actual.png"), dpi=150)
    plt.close(fig)

    # Residuals
    residuals = preds - targets
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(preds, residuals, alpha=0.5, s=15, color="#4CAF50", edgecolors="none")
    axes[0].axhline(y=0, color="#F44336", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=30, color="#9C27B0", alpha=0.7, edgecolor="white")
    axes[1].axvline(x=0, color="#F44336", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    fig.tight_layout()
    fig.savefig(os.path.join(report_dir, "residuals.png"), dpi=150)
    plt.close(fig)

    # Text report
    lines = ["Regression Report", "=" * 40]
    lines.append(f"MSE:      {results['val_mse']:.6f}")
    lines.append(f"RMSE:     {results['val_rmse']:.6f}")
    lines.append(f"MAE:      {results['val_mae']:.6f}")
    lines.append(f"R²:       {results['val_r2']:.6f}")
    lines.append(f"val_score: {results['val_score']:.6f} (= -MSE)")

    report_text = "\n".join(lines)
    with open(os.path.join(report_dir, "regression_report.txt"), "w") as f:
        f.write(report_text)
    print("\n" + report_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare genomic data for training")
    parser.add_argument("--fasta", type=str, default=None, help="Path to FASTA/FASTQ file")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file")
    parser.add_argument("--seq-col", type=str, default=None, help="Sequence column name (CSV)")
    parser.add_argument("--id-col", type=str, default=None, help="ID column name")
    parser.add_argument("--task", type=str, default="pretrain",
                        choices=["pretrain", "classify", "regress"],
                        help="Task type (default: pretrain)")
    parser.add_argument("--tokenizer", type=str, default="char",
                        choices=["char", "kmer", "bpe"],
                        help="Tokenizer type (default: char)")
    parser.add_argument("--kmer-size", type=int, default=6, help="K-mer size (default: 6)")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length in tokens (default: 512)")
    parser.add_argument("--labels", type=str, default=None, help="Path to labels CSV (classify/regress)")
    parser.add_argument("--label-col", type=str, default=None, help="Label column name")
    parser.add_argument("--bpe-vocab-size", type=int, default=4096, help="BPE vocabulary size (default: 4096)")
    parser.add_argument("--sample-n", type=int, default=0, help="Subsample to N sequences (0 = no sampling)")
    parser.add_argument("--sample-frac", type=float, default=0.0, help="Subsample fraction 0-1 (0 = no sampling)")
    parser.add_argument("--rc-double", action="store_true", help="Double dataset with reverse complement sequences")
    parser.add_argument("--chunk-strategy", type=str, default="fixed",
                        choices=["fixed", "none", "random", "slide"],
                        help="Chunking strategy for long sequences (default: fixed)")
    parser.add_argument("--n-folds", type=int, default=1,
                        help="Number of CV folds (default: 1 = no CV, just train/val split)")
    args = parser.parse_args()

    seq_path = args.fasta or args.csv
    if seq_path is None:
        parser.error("Must specify --fasta or --csv")

    print(f"Cache directory: {CACHE_DIR}")
    print()

    config = prepare_data(
        seq_path=seq_path,
        task_type=args.task,
        tokenizer_type=args.tokenizer,
        kmer_size=args.kmer_size,
        max_length=args.max_length,
        labels_path=args.labels,
        label_col=args.label_col,
        seq_col=args.seq_col,
        id_col=args.id_col,
        bpe_vocab_size=args.bpe_vocab_size,
        sample_n=args.sample_n,
        sample_frac=args.sample_frac,
        rc_double=args.rc_double,
        chunk_strategy=args.chunk_strategy,
        n_folds=args.n_folds,
    )

    print()
    print("Done! Ready to train with: python train.py")
