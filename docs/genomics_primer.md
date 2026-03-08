# Genomics Primer

A brief introduction to genomics concepts relevant to this framework.

## DNA Basics

DNA (deoxyribonucleic acid) is a molecule that carries genetic information. It consists of two complementary strands forming a double helix.

**Nucleotides**: The building blocks of DNA, represented by four letters:
- **A** (Adenine) pairs with **T** (Thymine)
- **C** (Cytosine) pairs with **G** (Guanine)
- **N** represents any unknown nucleotide

**Reverse Complement**: The complementary strand read in reverse. For example:
```
5' → ATCGATCG → 3'
3' ← TAGCTAGC ← 5'

Reverse complement of ATCGATCG = CGATCGAT
```

## Codons and Proteins

DNA is read in groups of 3 nucleotides called **codons**. Each codon encodes one amino acid:
```
ATG → Methionine (Start codon)
TAA → Stop
TAG → Stop
TGA → Stop
GCT → Alanine
...
```

The flow of genetic information:
```
DNA → (transcription) → mRNA → (translation) → Protein
```

## GC Content

The proportion of G and C nucleotides in a sequence. Important because:
- GC-rich regions are more thermally stable
- GC content varies between organisms (e.g., 25-75%)
- Can indicate horizontal gene transfer if different from genome average

## Open Reading Frames (ORFs)

An ORF is a stretch of DNA that begins with a start codon (ATG) and ends with a stop codon. ORFs potentially encode proteins.

## Mutations

Types of genetic variation:
- **SNP** (Single Nucleotide Polymorphism): One base changed (A→T)
- **Insertion**: Extra bases added
- **Deletion**: Bases removed
- **Indel**: Insertion or deletion

## Why Pre-train on Genomic Data?

Pre-trained genomic language models learn:

1. **Sequence grammar**: Rules governing nucleotide patterns (promoters, splice sites, regulatory elements)
2. **Evolutionary conservation**: Which positions are functionally important
3. **Structural features**: Patterns related to DNA secondary structure
4. **Species-specific signatures**: Codon usage, GC content, repeat patterns

These learned representations transfer to downstream tasks:
- **Variant effect prediction**: Is this mutation harmful?
- **Gene classification**: What family does this gene belong to?
- **Promoter detection**: Where do genes start transcribing?
- **Antimicrobial resistance**: Does this organism resist antibiotics?

## Sequence Lengths in Biology

| Organism Type | Typical Length | Example |
|--------------|----------------|---------|
| Virus | 5-30 kb | SARS-CoV-2: ~30 kb |
| Plasmid | 1-200 kb | pBR322: 4.4 kb |
| Bacterium | 0.5-10 Mb | E. coli: 4.6 Mb |
| Yeast | 12 Mb | S. cerevisiae: 12 Mb |
| Human gene | 1-2000 kb | Average: ~27 kb |
| Human genome | 3.2 Gb | 3,200 Mb |

(kb = kilobases, Mb = megabases, Gb = gigabases)

## File Formats

| Format | Extension | Content |
|--------|-----------|---------|
| FASTA | .fasta, .fa, .fna | Sequences with headers |
| FASTQ | .fastq, .fq | Sequences + quality scores |
| GenBank | .gb, .gbk | Annotated sequences |
| GFF/GTF | .gff, .gtf | Genomic feature annotations |
| VCF | .vcf | Variant calls |
| CSV | .csv | Tabular sequence data |

## Tokenization for Genomics

| Strategy | Example | Tokens | Best For |
|----------|---------|--------|----------|
| Character | ATCG → [A][T][C][G] | 4 | SSM models, short sequences |
| K-mer (k=3) | ATCG → [ATC][TCG] | 64 | Balanced compression |
| K-mer (k=6) | ATCGATCG → [ATCGAT][TCGATC][CGATCG] | 4096 | Long sequences with Transformer |
| BPE | Learned subwords | Configurable | Large datasets |
| Codon | ATCGAT → [ATC][GAT] | 64 | Coding regions |
