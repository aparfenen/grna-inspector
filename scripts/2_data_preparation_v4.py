"""
gRNA Data Preparation Pipeline v4.0 - Part 1: Setup & Data Loading

CRITICAL CORRECTIONS from v3:
1. DATA LEAKAGE FIX: Group-based splitting (duplicates stay together)
2. ANCHOR REGION: positions 5-7, length 6-21 (was 4-6, 8-12)
3. GUIDING REGION: starts position 17 (was 15)
4. PALINDROME: fixed parameter handling
"""

import sys
import os
import warnings
import re
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Set, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from Bio import SeqIO
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (10, 6)
np.random.seed(42)

print("✓ Imports loaded successfully")
print(f"  NumPy: {np.__version__}")
print(f"  Pandas: {pd.__version__}")


# =============================================================================
# FILE PATHS - UPDATE THESE FOR YOUR ENVIRONMENT!
# =============================================================================

PROJECT_ROOT = Path.home() / 'projects' / 'grna-inspector'
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'gRNAs' / 'Cooper_2022'
PROCESSED_DIR = DATA_DIR / 'processed' / 'v4_pipeline'  # NEW VERSION!
PLOTS_DIR = DATA_DIR / 'plots' / 'data_prep_v4'

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Input files
GRNA_FILE = RAW_DIR / 'mOs.gRNA.final.fasta'
MINICIRCLE_FILE = RAW_DIR / 'mOs.Cooper.minicircle.fasta'
GTF_FILE = RAW_DIR / 'mOs.gRNA.final.gtf'

print("\nChecking input files...")
for filepath in [GRNA_FILE, MINICIRCLE_FILE, GTF_FILE]:
    if filepath.exists():
        print(f"  ✓ {filepath.name}")
    else:
        print(f"  ✗ {filepath.name} - NOT FOUND!")

print(f"\nOutput directory: {PROCESSED_DIR}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sequence_hash(seq: str) -> str:
    """Generate hash for sequence (for duplicate detection)."""
    return hashlib.md5(seq.upper().encode()).hexdigest()


def parse_gtf_file(gtf_file: Path) -> Dict[str, List[Tuple[int, int]]]:
    """Parse GTF file and extract gRNA coordinates."""
    grna_regions = defaultdict(list)
    
    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            minicircle_id = parts[0]
            start = int(parts[3]) - 1  # Convert to 0-indexed
            end = int(parts[4])
            grna_regions[minicircle_id].append((start, end))
    
    for mini_id in grna_regions:
        grna_regions[mini_id].sort()
    
    return dict(grna_regions)


def merge_overlapping_regions(regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping or adjacent regions."""
    if not regions:
        return []
    merged = [regions[0]]
    for start, end in regions[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def get_non_grna_regions(minicircle_id: str, minicircle_length: int,
                         grna_coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Calculate non-gRNA regions."""
    if not grna_coords:
        return [(0, minicircle_length)]
    
    non_grna = []
    if grna_coords[0][0] > 0:
        non_grna.append((0, grna_coords[0][0]))
    for i in range(len(grna_coords) - 1):
        gap_start = grna_coords[i][1]
        gap_end = grna_coords[i + 1][0]
        if gap_end > gap_start:
            non_grna.append((gap_start, gap_end))
    if grna_coords[-1][1] < minicircle_length:
        non_grna.append((grna_coords[-1][1], minicircle_length))
    return non_grna


# =============================================================================
# STAGE 1: LOAD POSITIVE SEQUENCES
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 1: LOAD & VALIDATE POSITIVE SEQUENCES")
print("=" * 80)

positive_data = []
for record in SeqIO.parse(GRNA_FILE, "fasta"):
    seq = str(record.seq).upper().replace('U', 'T')
    positive_data.append({
        'seq_id': record.id,
        'sequence': seq,
        'length': len(seq),
        'seq_hash': sequence_hash(seq),
    })

positive_df = pd.DataFrame(positive_data)

# Assign group IDs (duplicates get same group)
hash_to_group = {h: i for i, h in enumerate(positive_df['seq_hash'].unique())}
positive_df['group_id'] = positive_df['seq_hash'].map(hash_to_group)

print(f"\nLoaded {len(positive_df):,} canonical gRNA sequences")

# Duplicate analysis
n_unique = positive_df['seq_hash'].nunique()
n_duplicates = len(positive_df) - n_unique
dup_counts = positive_df.groupby('seq_hash').size()

print(f"\n🔍 Duplicate Analysis:")
print(f"  Total sequences: {len(positive_df):,}")
print(f"  Unique sequences: {n_unique:,}")
print(f"  Duplicate entries: {n_duplicates:,}")
print(f"  Max copies: {dup_counts.max()}")

# Statistics
unique_seqs = positive_df.drop_duplicates('seq_hash')['sequence'].tolist()
positive_lengths = [len(seq) for seq in unique_seqs]

print(f"\n📊 Statistics (unique):")
print(f"  Length range: {min(positive_lengths)}-{max(positive_lengths)} nt")
print(f"  Mean length: {np.mean(positive_lengths):.1f} ± {np.std(positive_lengths):.1f} nt")


# =============================================================================
# STAGE 2: PARSE GTF
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 2: PARSE GTF")
print("=" * 80)

grna_regions = parse_gtf_file(GTF_FILE)
print(f"\nFound gRNA annotations for {len(grna_regions)} minicircles")

total_before = sum(len(regions) for regions in grna_regions.values())
for mini_id in grna_regions:
    grna_regions[mini_id] = merge_overlapping_regions(grna_regions[mini_id])
total_after = sum(len(regions) for regions in grna_regions.values())

print(f"  Total annotations: {total_before}")
print(f"  After merging: {total_after}")

print("\n✅ Part 1 complete - run Part 2 next")
"""
gRNA Data Preparation Pipeline v4.0 - Part 2: Negative Sampling & Feature Extraction

Run after Part 1!
"""

# =============================================================================
# NEGATIVE SAMPLING FUNCTIONS
# =============================================================================

def generate_minicircle_negatives(minicircle_file: Path, grna_regions_dict: Dict,
                                   target_lengths: List[int], n_samples: int,
                                   existing_seqs: Set[str]) -> Dict[str, str]:
    """Generate negatives from minicircles, EXCLUDING gRNA regions."""
    minicircles = []
    non_grna_regions = {}
    
    for record in SeqIO.parse(minicircle_file, "fasta"):
        mini_id = record.id
        seq = str(record.seq).upper().replace('U', 'T')
        minicircles.append((mini_id, seq))
        grna_coords = grna_regions_dict.get(mini_id, [])
        non_grna = get_non_grna_regions(mini_id, len(seq), grna_coords)
        non_grna_regions[mini_id] = non_grna
    
    print(f"  Loaded {len(minicircles)} minicircles")
    
    negatives = {}
    attempts = 0
    max_attempts = n_samples * 20
    
    while len(negatives) < n_samples and attempts < max_attempts:
        attempts += 1
        target_len = np.random.choice(target_lengths)
        mini_id, mini_seq = minicircles[np.random.randint(len(minicircles))]
        available_regions = non_grna_regions[mini_id]
        
        if not available_regions:
            continue
        
        region = available_regions[np.random.randint(len(available_regions))]
        region_start, region_end = region
        region_len = region_end - region_start
        
        if region_len < target_len:
            continue
        
        frag_start = np.random.randint(region_start, region_end - target_len + 1)
        fragment = mini_seq[frag_start:frag_start + target_len]
        
        if 'N' in fragment or len(set(fragment)) == 1 or fragment in existing_seqs:
            continue
        
        neg_id = f"neg_mini_{len(negatives):04d}"
        negatives[neg_id] = fragment
        existing_seqs.add(fragment)
    
    return negatives


def generate_chimeric_negatives(minicircle_file: Path, target_lengths: List[int],
                                 n_samples: int, existing_seqs: Set[str]) -> Dict[str, str]:
    """Generate chimeric sequences."""
    all_seqs = []
    for record in SeqIO.parse(minicircle_file, "fasta"):
        all_seqs.append(str(record.seq).upper().replace('U', 'T'))
    
    negatives = {}
    attempts = 0
    max_attempts = n_samples * 20
    
    while len(negatives) < n_samples and attempts < max_attempts:
        attempts += 1
        target_len = np.random.choice(target_lengths)
        n_fragments = np.random.choice([2, 3])
        frag_len = target_len // n_fragments
        
        fragments = []
        for _ in range(n_fragments):
            src_seq = all_seqs[np.random.randint(len(all_seqs))]
            if len(src_seq) < frag_len:
                continue
            start = np.random.randint(0, len(src_seq) - frag_len + 1)
            fragments.append(src_seq[start:start + frag_len])
        
        if len(fragments) != n_fragments:
            continue
        
        chimera = ''.join(fragments)[:target_len]
        
        if 'N' in chimera or len(set(chimera)) == 1 or chimera in existing_seqs:
            continue
        
        neg_id = f"neg_chim_{len(negatives):04d}"
        negatives[neg_id] = chimera
        existing_seqs.add(chimera)
    
    return negatives


def generate_random_negatives(target_lengths: List[int], nucleotide_probs: Dict[str, float],
                               n_samples: int, existing_seqs: Set[str]) -> Dict[str, str]:
    """Generate random sequences."""
    nucleotides = list(nucleotide_probs.keys())
    probs = list(nucleotide_probs.values())
    
    negatives = {}
    attempts = 0
    max_attempts = n_samples * 10
    
    while len(negatives) < n_samples and attempts < max_attempts:
        attempts += 1
        target_len = np.random.choice(target_lengths)
        seq = ''.join(np.random.choice(nucleotides, size=target_len, p=probs))
        
        if len(set(seq)) == 1 or seq in existing_seqs:
            continue
        
        neg_id = f"neg_rand_{len(negatives):04d}"
        negatives[neg_id] = seq
        existing_seqs.add(seq)
    
    return negatives


# =============================================================================
# STAGE 3: GENERATE NEGATIVES
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 3: GENERATE LENGTH-MATCHED NEGATIVES")
print("=" * 80)

# Calculate nucleotide composition
all_pos_seq = ''.join(positive_df['sequence'].tolist())
total_nt = len(all_pos_seq)
nucleotide_probs = {
    'A': all_pos_seq.count('A') / total_nt,
    'T': all_pos_seq.count('T') / total_nt,
    'G': all_pos_seq.count('G') / total_nt,
    'C': all_pos_seq.count('C') / total_nt,
}

print(f"\n🧬 Positive composition: A={nucleotide_probs['A']:.3f} T={nucleotide_probs['T']:.3f} G={nucleotide_probs['G']:.3f} C={nucleotide_probs['C']:.3f}")

# Target counts
n_unique_pos = positive_df['seq_hash'].nunique()
n_total_neg = n_unique_pos
n_minicircle = int(n_total_neg * 0.50)
n_chimeric = int(n_total_neg * 0.30)
n_random = n_total_neg - n_minicircle - n_chimeric

print(f"\n📊 Negative plan: mini={n_minicircle} chim={n_chimeric} rand={n_random}")

existing_seqs = set(positive_df['sequence'].tolist())

print("\n[1/3] Generating minicircle negatives...")
neg_minicircle = generate_minicircle_negatives(
    MINICIRCLE_FILE, grna_regions, positive_lengths, n_minicircle, existing_seqs
)
print(f"  Generated: {len(neg_minicircle)}")

print("\n[2/3] Generating chimeric negatives...")
neg_chimeric = generate_chimeric_negatives(
    MINICIRCLE_FILE, positive_lengths, n_chimeric, existing_seqs
)
print(f"  Generated: {len(neg_chimeric)}")

print("\n[3/3] Generating random negatives...")
neg_random = generate_random_negatives(
    positive_lengths, nucleotide_probs, n_random, existing_seqs
)
print(f"  Generated: {len(neg_random)}")

# Combine
negative_sequences = {}
negative_sources = {}

for seq_id, seq in neg_minicircle.items():
    negative_sequences[seq_id] = seq
    negative_sources[seq_id] = 'minicircle'

for seq_id, seq in neg_chimeric.items():
    negative_sequences[seq_id] = seq
    negative_sources[seq_id] = 'chimeric'

for seq_id, seq in neg_random.items():
    negative_sequences[seq_id] = seq
    negative_sources[seq_id] = 'random'

print(f"\n✅ Total negatives: {len(negative_sequences)}")

# Verify length matching
neg_lengths = [len(seq) for seq in negative_sequences.values()]
ks_stat, ks_pval = stats.ks_2samp(positive_lengths, neg_lengths)
print(f"\n📏 KS test p-value: {ks_pval:.4f}")
if ks_pval > 0.05:
    print("  ✓ Length distributions match")

print("\n✅ Part 2 complete - run Part 3 next")
"""
gRNA Data Preparation Pipeline v4.0 - Part 3: CORRECTED Feature Extractor

CRITICAL CORRECTIONS per Cooper et al. 2022:
- Anchor: positions 5-7 (0-indexed: 4-6), length 6-21 nt
- Guiding: position 17+ (0-indexed: 16+)
- Palindrome: proper parameter handling
"""

@dataclass
class GrnaRegionBoundaries:
    """CORRECTED boundaries per Cooper 2022."""
    initiation_start: int = 0
    initiation_end: int = 5
    
    # CORRECTED: Figure 7A - "majority of anchors begin 5 to 7 nt along"
    anchor_start_min: int = 4   # Position 5 (0-indexed)
    anchor_start_max: int = 6   # Position 7 (0-indexed)
    
    # CORRECTED: "mean anchor length is 11.4 nt with a range from 6...to 21 nt"
    anchor_length_min: int = 6
    anchor_length_max: int = 21
    
    # CORRECTED: Figure 6 - "region from positions 17 to 43"
    guiding_start: int = 16     # Position 17 (0-indexed)
    guiding_end: int = 43
    
    # Molecular ruler: "3' end of anchor between nt positions 15 and 19"
    ruler_3prime_min: int = 15
    ruler_3prime_max: int = 19


class CorrectedGrnaFeatureExtractor:
    """Feature extractor with CORRECTED parameters."""
    
    def __init__(self):
        self.bounds = GrnaRegionBoundaries()
        
        self.initiation_patterns = {
            'ATATA': 'ATATA',
            'AWAHH': r'A[AT]A[ACT][ACT]',
            'AAAA': 'AAAA',
            'GAAA': 'GAAA',
            'AGAA': 'AGAA',
            'XAAA': r'[ATGC]AAA',
        }
        
        self.important_3mers = ['AAA', 'ATA', 'TAT', 'TTT', 'AAT', 'ATT', 'GAA', 'AGA']
        self.important_4mers = ['ATAT', 'TATA', 'AAAA', 'TTTT', 'AAAG', 'AAGA', 'GAAA', 'AGAA']
    
    def extract_features(self, sequence: str) -> Dict[str, float]:
        features = {}
        seq = sequence.upper().replace('U', 'T')
        
        features.update(self._extract_initiation_features(seq))
        features.update(self._extract_anchor_features(seq))
        features.update(self._extract_guiding_features(seq))
        features.update(self._extract_terminal_features(seq))
        features.update(self._extract_structure_features(seq))
        features.update(self._extract_kmer_features(seq))
        features.update(self._extract_dinucleotide_features(seq))
        features.update(self._extract_composition_features(seq))
        features.update(self._extract_advanced_features(seq))
        features.update(self._extract_meta_features(seq, features))
        
        return features
    
    def _extract_initiation_features(self, seq: str) -> Dict[str, float]:
        features = {}
        init_region = seq[:6] if len(seq) >= 6 else seq
        
        for pattern_name, pattern in self.initiation_patterns.items():
            features[f'init_has_{pattern_name}'] = float(bool(re.match(pattern, init_region)))
        
        if len(seq) > 0:
            features['init_starts_A'] = float(seq[0] == 'A')
            features['init_starts_G'] = float(seq[0] == 'G')
            features['init_starts_T'] = float(seq[0] == 'T')
            features['init_starts_C'] = float(seq[0] == 'C')
            features['init_starts_purine'] = float(seq[0] in 'AG')
        else:
            for f in ['init_starts_A', 'init_starts_G', 'init_starts_T', 'init_starts_C', 'init_starts_purine']:
                features[f] = 0.0
        
        first4 = seq[:4] if len(seq) >= 4 else seq
        if len(first4) > 0:
            features['init_4_A_count'] = float(first4.count('A'))
            features['init_4_T_count'] = float(first4.count('T'))
            features['init_4_G_count'] = float(first4.count('G'))
            features['init_4_C_count'] = float(first4.count('C'))
            features['init_4_A_rich'] = float(first4.count('A') >= 3)
        else:
            for f in ['init_4_A_count', 'init_4_T_count', 'init_4_G_count', 'init_4_C_count', 'init_4_A_rich']:
                features[f] = 0.0
        
        total_patterns = sum(1 for p in self.initiation_patterns if features.get(f'init_has_{p}', 0) > 0)
        features['init_pattern_count'] = float(total_patterns)
        features['init_any_known_pattern'] = float(total_patterns > 0)
        
        return features
    
    def _extract_anchor_features(self, seq: str) -> Dict[str, float]:
        """CORRECTED: positions 5-7, length 6-21."""
        features = {}
        best_anchor = None
        best_score = -1
        best_start = 0
        
        # CORRECTED range
        for start in range(self.bounds.anchor_start_min,
                          min(self.bounds.anchor_start_max + 1, len(seq) - 5)):
            for length in range(self.bounds.anchor_length_min,
                               min(self.bounds.anchor_length_max + 1, len(seq) - start + 1)):
                anchor = seq[start:start + length]
                if len(anchor) < 6:
                    continue
                
                ac_content = (anchor.count('A') + anchor.count('C')) / len(anchor)
                g_content = anchor.count('G') / len(anchor)
                score = ac_content - g_content
                
                if score > best_score:
                    best_score = score
                    best_anchor = anchor
                    best_start = start
        
        if best_anchor and len(best_anchor) > 0:
            n = len(best_anchor)
            for nt in 'ATGC':
                features[f'anchor_{nt}_freq'] = best_anchor.count(nt) / n
            
            features['anchor_AT_freq'] = (best_anchor.count('A') + best_anchor.count('T')) / n
            features['anchor_GC_freq'] = (best_anchor.count('G') + best_anchor.count('C')) / n
            features['anchor_purine_freq'] = (best_anchor.count('A') + best_anchor.count('G')) / n
            features['anchor_AC_content'] = (best_anchor.count('A') + best_anchor.count('C')) / n
            features['anchor_length'] = float(n)
            features['anchor_start_pos'] = float(best_start)
            features['anchor_G_depleted'] = float(features['anchor_G_freq'] < 0.15)
            features['anchor_AC_rich'] = float(features['anchor_AC_content'] > 0.60)
            features['anchor_AC_very_rich'] = float(features['anchor_AC_content'] > 0.70)
            
            anchor_3prime = best_start + n
            features['anchor_3prime_pos'] = float(anchor_3prime)
            features['init_anchor_total_len'] = float(anchor_3prime)
            features['in_molecular_ruler_range'] = float(
                self.bounds.ruler_3prime_min <= anchor_3prime <= self.bounds.ruler_3prime_max
            )
            features['anchor_entropy'] = self._calculate_entropy(best_anchor)
            features['anchor_unique_dinucs'] = float(len(set(
                best_anchor[i:i+2] for i in range(len(best_anchor) - 1)
            ))) if n > 1 else 0.0
        else:
            for ft in ['anchor_A_freq', 'anchor_T_freq', 'anchor_G_freq', 'anchor_C_freq',
                      'anchor_AT_freq', 'anchor_GC_freq', 'anchor_purine_freq', 'anchor_AC_content',
                      'anchor_length', 'anchor_start_pos', 'anchor_G_depleted', 'anchor_AC_rich',
                      'anchor_AC_very_rich', 'anchor_3prime_pos', 'init_anchor_total_len',
                      'in_molecular_ruler_range', 'anchor_entropy', 'anchor_unique_dinucs']:
                features[ft] = 0.0
        
        return features
    
    def _extract_guiding_features(self, seq: str) -> Dict[str, float]:
        """CORRECTED: starts at position 17 (index 16)."""
        features = {}
        
        # CORRECTED start
        guide_start = self.bounds.guiding_start  # 16 = position 17
        guide_end = min(self.bounds.guiding_end, len(seq))
        guide = seq[guide_start:guide_end] if len(seq) > guide_start else ""
        
        if len(guide) > 0:
            n = len(guide)
            for nt in 'ATGC':
                features[f'guide_{nt}_freq'] = guide.count(nt) / n
            
            features['guide_AT_freq'] = (guide.count('A') + guide.count('T')) / n
            features['guide_GC_freq'] = (guide.count('G') + guide.count('C')) / n
            features['guide_A_elevated'] = float(features['guide_A_freq'] > 0.40)
            features['guide_A_content_high'] = float(features['guide_A_freq'] > 0.45)
            features['guide_purine_freq'] = (guide.count('A') + guide.count('G')) / n
            features['guide_purine_rich'] = float(features['guide_purine_freq'] > 0.55)
            features['guide_pyrimidine_freq'] = (guide.count('T') + guide.count('C')) / n
            features['guide_C_count'] = float(guide.count('C'))
            features['guide_T_count'] = float(guide.count('T'))
            features['guide_edit_potential'] = (guide.count('C') + guide.count('T')) / n
        else:
            for ft in ['guide_A_freq', 'guide_T_freq', 'guide_G_freq', 'guide_C_freq',
                      'guide_AT_freq', 'guide_GC_freq', 'guide_A_elevated', 'guide_A_content_high',
                      'guide_purine_freq', 'guide_purine_rich', 'guide_pyrimidine_freq',
                      'guide_C_count', 'guide_T_count', 'guide_edit_potential']:
                features[ft] = 0.0
        
        return features
    
    def _extract_terminal_features(self, seq: str) -> Dict[str, float]:
        features = {}
        if len(seq) > 0:
            features['ends_with_T'] = float(seq[-1] == 'T')
            features['ends_with_A'] = float(seq[-1] == 'A')
            features['ends_with_G'] = float(seq[-1] == 'G')
            features['ends_with_C'] = float(seq[-1] == 'C')
            
            last3 = seq[-3:] if len(seq) >= 3 else seq
            features['last3_T_count'] = float(last3.count('T'))
            features['last3_A_count'] = float(last3.count('A'))
            features['last3_TT'] = float(last3.endswith('TT')) if len(last3) >= 2 else 0.0
            features['last3_AT'] = float('AT' in last3) if len(last3) >= 2 else 0.0
            
            last5 = seq[-5:] if len(seq) >= 5 else seq
            if len(last5) > 0:
                features['last5_T_freq'] = last5.count('T') / len(last5)
                features['last5_A_freq'] = last5.count('A') / len(last5)
                features['last5_AT_freq'] = (last5.count('A') + last5.count('T')) / len(last5)
            else:
                features['last5_T_freq'] = features['last5_A_freq'] = features['last5_AT_freq'] = 0.0
            
            features['ends_poly_T_2'] = float(seq[-2:] == 'TT') if len(seq) >= 2 else 0.0
            features['ends_poly_T_3'] = float(seq[-3:] == 'TTT') if len(seq) >= 3 else 0.0
            features['ends_single_T'] = float(seq[-1] == 'T' and (len(seq) < 2 or seq[-2] != 'T'))
        else:
            for ft in ['ends_with_T', 'ends_with_A', 'ends_with_G', 'ends_with_C',
                      'last3_T_count', 'last3_A_count', 'last3_TT', 'last3_AT',
                      'last5_T_freq', 'last5_A_freq', 'last5_AT_freq',
                      'ends_poly_T_2', 'ends_poly_T_3', 'ends_single_T']:
                features[ft] = 0.0
        return features
    
    def _extract_structure_features(self, seq: str) -> Dict[str, float]:
        features = {}
        n = len(seq)
        
        if n == 0:
            for ft in ['entropy', 'complexity_ratio', 'max_homopolymer', 'n_homopolymers_3plus',
                      'has_palindrome_4bp', 'has_palindrome_5bp', 'has_palindrome_6bp', 'no_palindrome_5bp']:
                features[ft] = 0.0
            features['no_palindrome_5bp'] = 1.0
            return features
        
        features['entropy'] = self._calculate_entropy(seq)
        
        if n >= 3:
            unique_3mers = len(set(seq[i:i+3] for i in range(n - 2)))
            possible = min(n - 2, 64)
            features['complexity_ratio'] = unique_3mers / possible if possible > 0 else 0.0
        else:
            features['complexity_ratio'] = 0.0
        
        # Homopolymers
        max_run = 1
        n_runs = 0
        current = 1
        for i in range(1, n):
            if seq[i] == seq[i - 1]:
                current += 1
            else:
                if current >= 3:
                    n_runs += 1
                max_run = max(max_run, current)
                current = 1
        if current >= 3:
            n_runs += 1
        max_run = max(max_run, current)
        features['max_homopolymer'] = float(max_run)
        features['n_homopolymers_3plus'] = float(n_runs)
        
        # FIXED palindrome detection with explicit params
        features['has_palindrome_4bp'] = float(self._has_palindrome(seq, min_stem=4, min_loop=3))
        features['has_palindrome_5bp'] = float(self._has_palindrome(seq, min_stem=5, min_loop=3))
        features['has_palindrome_6bp'] = float(self._has_palindrome(seq, min_stem=6, min_loop=3))
        features['no_palindrome_5bp'] = 1.0 - features['has_palindrome_5bp']
        
        return features
    
    def _has_palindrome(self, seq: str, min_stem: int = 5, min_loop: int = 3) -> bool:
        """FIXED: properly uses min_stem and min_loop."""
        comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        n = len(seq)
        min_total = 2 * min_stem + min_loop
        if n < min_total:
            return False
        
        for i in range(n - min_total + 1):
            stem = seq[i:i + min_stem]
            rc = ''.join(comp.get(nt, nt) for nt in reversed(stem))
            if rc in seq[i + min_stem + min_loop:]:
                return True
        return False
    
    def _extract_kmer_features(self, seq: str) -> Dict[str, float]:
        features = {}
        n = len(seq)
        
        if n < 3:
            for kmer in self.important_3mers:
                features[f'kmer3_{kmer}_count'] = 0.0
                features[f'kmer3_{kmer}_freq'] = 0.0
            for kmer in self.important_4mers:
                features[f'kmer4_{kmer}_present'] = 0.0
            return features
        
        kmer3_counts = Counter(seq[i:i+3] for i in range(n - 2))
        total_3mers = n - 2
        for kmer in self.important_3mers:
            count = kmer3_counts.get(kmer, 0)
            features[f'kmer3_{kmer}_count'] = float(count)
            features[f'kmer3_{kmer}_freq'] = count / total_3mers if total_3mers > 0 else 0.0
        
        if n >= 4:
            for kmer in self.important_4mers:
                features[f'kmer4_{kmer}_present'] = float(kmer in seq)
        else:
            for kmer in self.important_4mers:
                features[f'kmer4_{kmer}_present'] = 0.0
        
        return features
    
    def _extract_dinucleotide_features(self, seq: str) -> Dict[str, float]:
        features = {}
        n = len(seq)
        important_dinucs = ['AA', 'AT', 'TA', 'TT', 'GC', 'CG', 'AC', 'CA']
        
        if n < 2:
            for dn in important_dinucs:
                features[f'dinuc_{dn}_freq'] = 0.0
            features['dinuc_bias_AT'] = 0.0
            return features
        
        dinuc_counts = Counter(seq[i:i+2] for i in range(n - 1))
        total = n - 1
        for dn in important_dinucs:
            features[f'dinuc_{dn}_freq'] = dinuc_counts.get(dn, 0) / total
        
        at_dinucs = sum(dinuc_counts.get(d, 0) for d in ['AA', 'AT', 'TA', 'TT'])
        features['dinuc_bias_AT'] = at_dinucs / total
        
        return features
    
    def _extract_composition_features(self, seq: str) -> Dict[str, float]:
        features = {}
        n = len(seq)
        
        if n == 0:
            for nt in 'ATGC':
                features[f'global_{nt}_freq'] = 0.0
            features['global_AT_content'] = 0.0
            features['global_GC_content'] = 0.0
            features['global_purine_content'] = 0.0
            return features
        
        for nt in 'ATGC':
            features[f'global_{nt}_freq'] = seq.count(nt) / n
        features['global_AT_content'] = (seq.count('A') + seq.count('T')) / n
        features['global_GC_content'] = (seq.count('G') + seq.count('C')) / n
        features['global_purine_content'] = (seq.count('A') + seq.count('G')) / n
        
        return features
    
    def _extract_advanced_features(self, seq: str) -> Dict[str, float]:
        features = {}
        n = len(seq)
        
        if n == 0:
            features['skew_AT'] = features['skew_GC'] = features['balance_ratio'] = features['gc_middle_third'] = 0.0
            return features
        
        a, t, g, c = seq.count('A'), seq.count('T'), seq.count('G'), seq.count('C')
        features['skew_AT'] = (a - t) / (a + t) if a + t > 0 else 0.0
        features['skew_GC'] = (g - c) / (g + c) if g + c > 0 else 0.0
        freqs = [a, t, g, c]
        features['balance_ratio'] = min(freqs) / max(freqs) if max(freqs) > 0 else 0.0
        
        third = n // 3
        if third > 0:
            middle = seq[third:2 * third]
            features['gc_middle_third'] = (middle.count('G') + middle.count('C')) / len(middle) if middle else 0.0
        else:
            features['gc_middle_third'] = 0.0
        
        return features
    
    def _extract_meta_features(self, seq: str, existing: Dict) -> Dict[str, float]:
        features = {}
        
        init_score = existing.get('init_any_known_pattern', 0)
        anchor_score = existing.get('anchor_AC_rich', 0)
        guide_score = existing.get('guide_A_elevated', 0)
        terminal_score = existing.get('ends_with_T', 0)
        structure_score = existing.get('no_palindrome_5bp', 0)
        
        features['grna_signature_count'] = init_score + anchor_score + guide_score + terminal_score + structure_score
        features['grna_signature_all'] = float(init_score > 0 and anchor_score > 0 and guide_score > 0 and terminal_score > 0)
        features['init_anchor_quality'] = (
            existing.get('init_any_known_pattern', 0) * 0.3 +
            existing.get('anchor_AC_rich', 0) * 0.4 +
            existing.get('in_molecular_ruler_range', 0) * 0.3
        )
        features['utail_ready'] = float(
            existing.get('ends_single_T', 0) > 0 and existing.get('no_palindrome_5bp', 0) > 0
        )
        
        return features
    
    def _calculate_entropy(self, seq: str) -> float:
        if len(seq) == 0:
            return 0.0
        counts = Counter(seq)
        probs = [c / len(seq) for c in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)


print("✓ CorrectedGrnaFeatureExtractor defined")
print("\n  CORRECTED parameters:")
bounds = GrnaRegionBoundaries()
print(f"    Anchor start: positions {bounds.anchor_start_min + 1}-{bounds.anchor_start_max + 1}")
print(f"    Anchor length: {bounds.anchor_length_min}-{bounds.anchor_length_max} nt")
print(f"    Guiding start: position {bounds.guiding_start + 1}")

print("\n✅ Part 3 complete - run Part 4 next")
"""
gRNA Data Preparation Pipeline v4.0 - Part 4: Feature Extraction, Splitting & Export

CRITICAL: Uses GroupShuffleSplit to prevent data leakage!
"""

# =============================================================================
# STAGE 4: FEATURE EXTRACTION
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 4: FEATURE EXTRACTION")
print("=" * 80)

extractor = CorrectedGrnaFeatureExtractor()

# Positives
print("\n[1/2] Extracting features from positives...")
positive_features = []
for idx, row in positive_df.iterrows():
    feats = extractor.extract_features(row['sequence'])
    feats['seq_id'] = row['seq_id']
    feats['sequence'] = row['sequence']
    feats['label'] = 1
    feats['source'] = 'canonical_grna'
    feats['group_id'] = row['group_id']
    positive_features.append(feats)
print(f"  Extracted: {len(positive_features)}")

# Negatives
print("\n[2/2] Extracting features from negatives...")
negative_features = []
next_group_id = positive_df['group_id'].max() + 1

for seq_id, seq in negative_sequences.items():
    feats = extractor.extract_features(seq)
    feats['seq_id'] = seq_id
    feats['sequence'] = seq
    feats['label'] = 0
    feats['source'] = negative_sources[seq_id]
    feats['group_id'] = next_group_id
    negative_features.append(feats)
    next_group_id += 1
print(f"  Extracted: {len(negative_features)}")

# Combine
df_pos = pd.DataFrame(positive_features)
df_neg = pd.DataFrame(negative_features)
df_all = pd.concat([df_pos, df_neg], ignore_index=True)

non_feature_cols = ['seq_id', 'sequence', 'label', 'source', 'group_id']
feature_cols = [c for c in df_all.columns if c not in non_feature_cols]

print(f"\n📊 Dataset: {len(df_all):,} samples, {len(feature_cols)} features")


# =============================================================================
# STAGE 5: QUALITY CONTROL
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 5: QUALITY CONTROL")
print("=" * 80)

# NaN
nan_features = [c for c in feature_cols if df_all[c].isna().any()]
if nan_features:
    df_all[feature_cols] = df_all[feature_cols].fillna(0)
    print(f"  Fixed NaN in {len(nan_features)} features")
else:
    print("  ✓ No NaN")

# Inf
inf_features = [c for c in feature_cols if np.isinf(df_all[c]).any()]
if inf_features:
    df_all[feature_cols] = df_all[feature_cols].replace([np.inf, -np.inf], 0)
    print(f"  Fixed Inf in {len(inf_features)} features")
else:
    print("  ✓ No Inf")

# Constant
constant_features = [c for c in feature_cols if df_all[c].nunique() <= 1]
if constant_features:
    feature_cols = [c for c in feature_cols if c not in constant_features]
    print(f"  Removed {len(constant_features)} constant features")
else:
    print("  ✓ No constant features")

print(f"\n📊 Final features: {len(feature_cols)}")


# =============================================================================
# STAGE 6: GROUP-BASED SPLITTING (CRITICAL FIX!)
# =============================================================================

def verify_no_leakage(train_df, val_df, test_df):
    train_seqs = set(train_df['sequence'])
    val_seqs = set(val_df['sequence'])
    test_seqs = set(test_df['sequence'])
    
    return {
        'train_val': len(train_seqs & val_seqs),
        'train_test': len(train_seqs & test_seqs),
        'val_test': len(val_seqs & test_seqs),
        'is_clean': len(train_seqs & val_seqs) == 0 and len(train_seqs & test_seqs) == 0 and len(val_seqs & test_seqs) == 0
    }


print("\n" + "=" * 80)
print("STAGE 6: GROUP-BASED SPLITTING")
print("=" * 80)

# Split 1: train+val vs test (15%)
print("\n[1/3] train+val vs test...")
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_val_idx, test_idx = next(gss1.split(df_all, df_all['label'], groups=df_all['group_id']))

train_val_df = df_all.iloc[train_val_idx].copy()
test_df = df_all.iloc[test_idx].copy()
print(f"  Train+Val: {len(train_val_df):,}, Test: {len(test_df):,}")

# Split 2: train vs val (~18% of remaining)
print("\n[2/3] train vs val...")
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=42)
train_idx, val_idx = next(gss2.split(train_val_df, train_val_df['label'], groups=train_val_df['group_id']))

train_df = train_val_df.iloc[train_idx].copy()
val_df = train_val_df.iloc[val_idx].copy()
print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}")

# Verify
print("\n[3/3] Verifying no leakage...")
leakage = verify_no_leakage(train_df, val_df, test_df)
print(f"  Train↔Val overlap: {leakage['train_val']}")
print(f"  Train↔Test overlap: {leakage['train_test']}")
print(f"  Val↔Test overlap: {leakage['val_test']}")

if leakage['is_clean']:
    print("  ✅ NO DATA LEAKAGE!")
else:
    print("  ❌ DATA LEAKAGE DETECTED!")

# Stats
print("\n📊 Final split:")
total = len(df_all)
for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    pos = sum(df['label'] == 1)
    neg = sum(df['label'] == 0)
    ratio = min(pos, neg) / max(pos, neg) if max(pos, neg) > 0 else 0
    print(f"  {name}: {len(df):,} ({len(df)/total*100:.1f}%) - pos={pos} neg={neg} balance={ratio:.3f}")


# =============================================================================
# STAGE 7: EXPORT
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 7: EXPORT")
print("=" * 80)

train_file = PROCESSED_DIR / 'train_data.csv'
val_file = PROCESSED_DIR / 'val_data.csv'
test_file = PROCESSED_DIR / 'test_data.csv'

train_df.to_csv(train_file, index=False)
print(f"  ✓ {train_file}")

val_df.to_csv(val_file, index=False)
print(f"  ✓ {val_file}")

test_df.to_csv(test_file, index=False)
print(f"  ✓ {test_file}")

# Feature names
feature_file = PROCESSED_DIR / 'feature_names.txt'
with open(feature_file, 'w') as f:
    for feat in feature_cols:
        f.write(feat + '\n')
print(f"  ✓ {feature_file}")

# Metadata
metadata = {
    'pipeline_version': '4.0',
    'creation_date': pd.Timestamp.now().isoformat(),
    'critical_fixes': [
        'Group-based splitting (no duplicate leakage)',
        'Anchor region: positions 5-7, length 6-21',
        'Guiding region: starts position 17',
        'Palindrome detection: fixed parameters'
    ],
    'splits': {
        'train': len(train_df),
        'val': len(val_df),
        'test': len(test_df)
    },
    'features': len(feature_cols),
    'leakage_check': leakage
}

meta_file = PROCESSED_DIR / 'dataset_summary.json'
with open(meta_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ {meta_file}")


print("\n" + "=" * 80)
print("✅ DATA PREPARATION v4.0 COMPLETE!")
print("=" * 80)
print(f"\n📁 Output: {PROCESSED_DIR}")
print(f"📊 Samples: {len(df_all):,} | Features: {len(feature_cols)}")
print(f"✅ Ready for training with 3_model_training_v4.py!")
