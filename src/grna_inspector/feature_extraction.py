"""
Comprehensive gRNA feature extraction module.

Implements 120+ features based on Cooper et al. 2022 RNA and supplemental analysis.
Features are organized into 9 categories for biological interpretability.

Reference:
- Cooper et al. 2022: Assembly and annotation of the mitochondrial minicircle 
  genome of a differentiation-competent strain of Trypanosoma brucei
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from collections import Counter


class gRNAFeatureExtractor:
    """Extract comprehensive features from gRNA sequences."""
    
    def __init__(self):
        """Initialize feature extractor with pattern matchers and k-mer lists."""
        # IUPAC nucleotide codes for pattern matching
        self.iupac = {
            'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'U': 'T',
            'R': '[AG]', 'Y': '[CT]', 'W': '[AT]', 'S': '[GC]',
            'M': '[AC]', 'K': '[GT]', 'B': '[CGT]', 'D': '[AGT]',
            'H': '[ACT]', 'V': '[ACG]', 'N': '[ACGT]'
        }
        
        # Important k-mers from Cooper 2022 analysis
        self.important_3mers = ['ATT', 'TAA', 'AAT', 'TTA', 'ATA', 'AAA', 
                               'TAT', 'TTC', 'AAG', 'AGA']
        
        # Region boundaries (approximate, will adjust per sequence)
        self.INIT_LENGTH = 5  # Initiation sequence: first 5 nt
        self.ANCHOR_START = 5  # Anchor typically starts after init
        self.ANCHOR_TYPICAL_LEN = 11  # Median anchor length
        
    def matches_pattern(self, seq: str, pattern: str) -> bool:
        """
        Match IUPAC pattern to sequence.
        
        Example: 'AWAHH' matches sequences like 'AAAAT', 'ATACT', etc.
        where W=[AT], H=[ACT]
        """
        regex = ''.join(self.iupac.get(c.upper(), c) for c in pattern)
        return bool(re.match(regex, seq.upper()))
    
    # =================================================================
    # 1. INITIATION SEQUENCE FEATURES (15 features)
    # =================================================================
    
    def _extract_init_features(self, seq: str) -> Dict[str, float]:
        """
        Extract initiation sequence features (first 5 nt).
        
        Key insights from Cooper 2022:
        - 96% of canonical gRNAs start with AWA (A/T at pos 1-2)
        - ATATA found in 60% canonical, 42% noncanonical
        - AWAHH pattern covers 95% of canonical gRNAs
        """
        features = {}
        
        # Ensure we have at least 5 nt
        if len(seq) < 5:
            return {f'init_{k}': 0 for k in range(20)}  # Return zeros
        
        init = seq[:5].upper()
        
        # 1.1 Exact motif matches
        features['has_ATATA'] = int(init == 'ATATA')
        features['has_AAAAT'] = int('AAAA' in init)
        features['starts_ATA'] = int(seq[:3] == 'ATA')
        features['starts_AAA'] = int(seq[:3] == 'AAA')
        
        # 1.2 Consensus patterns (CRITICAL!)
        features['matches_AWAHH'] = int(self.matches_pattern(init, 'AWAHH'))
        features['matches_B_AWAHH'] = int(len(seq) >= 6 and 
                                         self.matches_pattern(seq[:6], 'BAWAHH'))
        features['matches_RYAYA'] = int(self.matches_pattern(init, 'RYAYA'))
        features['starts_AWA'] = int(self.matches_pattern(seq[:3], 'AWA'))
        
        # 1.3 Position-specific nucleotides
        features['init_pos1_is_A'] = int(init[0] == 'A')
        features['init_pos2_is_T'] = int(init[1] == 'T')
        features['init_pos3_is_A'] = int(init[2] == 'A')
        features['init_pos4_is_T'] = int(init[3] == 'T')
        features['init_pos5_is_A'] = int(init[4] == 'A')
        
        # 1.4 TA repeats (alternating pattern characteristic)
        features['has_TA_repeat_init'] = init.count('TA')
        features['init_alternating'] = int(self._is_alternating_AT(init))
        
        return features
    
    def _is_alternating_AT(self, seq: str) -> bool:
        """Check if sequence has alternating A/T pattern."""
        for i in range(len(seq) - 1):
            if not ((seq[i] in 'AT' and seq[i+1] in 'AT' and seq[i] != seq[i+1]) or
                    seq[i] not in 'AT'):
                return False
        return True
    
    # =================================================================
    # 2. ANCHOR REGION FEATURES (15 features)
    # =================================================================
    
    def _extract_anchor_features(self, seq: str) -> Dict[str, float]:
        """
        Extract anchor region features (typically positions 5-16).
        
        Key insights from Cooper 2022:
        - Anchor is AC-rich and GT-poor (prevents GU wobble pairing)
        - Median anchor length = 11 nt (range 6-21, 90% are 7-16)
        - Low G nucleotide frequency is key discriminator
        """
        features = {}
        
        # Define anchor region (approximate)
        if len(seq) < 15:
            anchor = seq[5:] if len(seq) > 5 else seq
        else:
            anchor = seq[5:16]  # Typical anchor region
        
        if len(anchor) == 0:
            return {f'anchor_{k}': 0 for k in range(15)}
        
        # 2.1 Anchor composition (CRITICAL!)
        features['anchor_A_freq'] = anchor.count('A') / len(anchor)
        features['anchor_C_freq'] = anchor.count('C') / len(anchor)
        features['anchor_G_freq'] = anchor.count('G') / len(anchor)
        features['anchor_T_freq'] = anchor.count('T') / len(anchor)
        
        features['anchor_AC_content'] = (anchor.count('A') + anchor.count('C')) / len(anchor)
        features['anchor_GT_content'] = (anchor.count('G') + anchor.count('T')) / len(anchor)
        
        # 2.2 Key discriminators
        features['anchor_G_is_low'] = int(features['anchor_G_freq'] < 0.15)
        features['anchor_T_is_low'] = int(features['anchor_T_freq'] < 0.25)
        features['anchor_AC_rich'] = int(features['anchor_AC_content'] > 0.50)
        
        # 2.3 Anchor structure
        features['anchor_length'] = len(anchor)
        features['anchor_length_canonical'] = int(7 <= len(anchor) <= 16)
        
        # 2.4 Combined init+anchor length (molecular ruler!)
        features['init_anchor_combined'] = 5 + len(anchor)
        features['init_anchor_in_range'] = int(15 <= features['init_anchor_combined'] <= 19)
        
        # 2.5 Anchor patterns
        features['anchor_has_polyA'] = int('AAA' in anchor)
        features['anchor_has_polyC'] = int('CCC' in anchor)
        
        return features
    
    # =================================================================
    # 3. GUIDING REGION FEATURES (15 features)
    # =================================================================
    
    def _extract_guide_features(self, seq: str) -> Dict[str, float]:
        """
        Extract guiding region features (typically positions 17-43).
        
        Key insights from Cooper 2022:
        - A frequency remains constant at ~40%
        - G frequency elevated at ~25%
        - T frequency rises from 20% to 30%
        - C frequency declines to ~2%
        """
        features = {}
        
        # Define guide region (positions after anchor)
        if len(seq) < 17:
            guide = seq[5:] if len(seq) > 5 else ''
        else:
            guide = seq[16:-2] if len(seq) > 18 else seq[16:]
        
        if len(guide) == 0:
            return {f'guide_{k}': 0 for k in range(15)}
        
        # 3.1 Guiding region composition
        features['guide_A_freq'] = guide.count('A') / len(guide)
        features['guide_C_freq'] = guide.count('C') / len(guide)
        features['guide_G_freq'] = guide.count('G') / len(guide)
        features['guide_T_freq'] = guide.count('T') / len(guide)
        
        # 3.2 Key patterns from Cooper 2022
        features['guide_A_is_high'] = int(features['guide_A_freq'] > 0.35)
        features['guide_G_is_elevated'] = int(0.20 <= features['guide_G_freq'] <= 0.30)
        features['guide_C_is_low'] = int(features['guide_C_freq'] < 0.10)
        
        # 3.3 Guide structure
        features['guide_length'] = len(guide)
        features['guide_length_canonical'] = int(20 <= len(guide) <= 40)
        
        # 3.4 AT content in guide
        features['guide_AT_content'] = (guide.count('A') + guide.count('T')) / len(guide)
        features['guide_GC_content'] = (guide.count('G') + guide.count('C')) / len(guide)
        
        # 3.5 Editing sites estimation (non-T intervals)
        features['num_editing_sites'] = self._count_editing_sites(guide)
        features['editing_sites_canonical'] = int(13 <= features['num_editing_sites'] <= 22)
        
        # 3.6 Purine/pyrimidine balance
        purines = guide.count('A') + guide.count('G')
        pyrimidines = guide.count('C') + guide.count('T')
        features['guide_purine_freq'] = purines / len(guide) if len(guide) > 0 else 0
        
        return features
    
    def _count_editing_sites(self, seq: str) -> int:
        """Estimate number of editing sites (rough approximation)."""
        # Count runs of non-T nucleotides as potential editing sites
        editing_sites = 0
        in_site = False
        for nt in seq:
            if nt != 'T':
                if not in_site:
                    editing_sites += 1
                    in_site = True
            else:
                in_site = False
        return editing_sites
    
    # =================================================================
    # 4. 3' END FEATURES (10 features)
    # =================================================================
    
    def _extract_3prime_features(self, seq: str) -> Dict[str, float]:
        """
        Extract 3' end features.
        
        Key insights from Cooper 2022:
        - 90% of canonical gRNAs end with T (vs 78% noncanonical)
        - T frequency increases to ~50% in last 7 nt
        - G frequency drops to ~10% at 3' end
        """
        features = {}
        
        # 4.1 Terminal nucleotide (CRITICAL!)
        features['ends_with_T'] = int(seq[-1].upper() == 'T')
        features['ends_with_A'] = int(seq[-1].upper() == 'A')
        features['ends_with_G'] = int(seq[-1].upper() == 'G')
        features['ends_with_C'] = int(seq[-1].upper() == 'C')
        
        # 4.2 Last 7 nucleotides composition
        if len(seq) >= 7:
            end7 = seq[-7:].upper()
            features['3prime_T_freq'] = end7.count('T') / 7
            features['3prime_G_freq'] = end7.count('G') / 7
            features['3prime_A_freq'] = end7.count('A') / 7
            
            # Key patterns
            features['3prime_T_high'] = int(features['3prime_T_freq'] > 0.45)
            features['3prime_G_low'] = int(features['3prime_G_freq'] < 0.15)
        else:
            features['3prime_T_freq'] = 0
            features['3prime_G_freq'] = 0
            features['3prime_A_freq'] = 0
            features['3prime_T_high'] = 0
            features['3prime_G_low'] = 0
        
        # 4.3 Poly-T at end
        features['has_polyT_end'] = int('TTT' in seq[-10:] if len(seq) >= 10 else False)
        
        return features
    
    # =================================================================
    # 5. GLOBAL COMPOSITION FEATURES (20 features)
    # =================================================================
    
    def _extract_composition_features(self, seq: str) -> Dict[str, float]:
        """Extract global nucleotide composition features."""
        features = {}
        
        seq_upper = seq.upper()
        length = len(seq)
        
        # 5.1 Basic frequencies
        features['A_freq'] = seq_upper.count('A') / length
        features['C_freq'] = seq_upper.count('C') / length
        features['G_freq'] = seq_upper.count('G') / length
        features['T_freq'] = seq_upper.count('T') / length
        
        # 5.2 Combined frequencies
        features['AT_content'] = (seq_upper.count('A') + seq_upper.count('T')) / length
        features['GC_content'] = (seq_upper.count('G') + seq_upper.count('C')) / length
        
        # 5.3 Key indicators
        features['is_AT_rich'] = int(features['AT_content'] > 0.60)
        features['is_GC_poor'] = int(features['GC_content'] < 0.40)
        
        # 5.4 Purine/Pyrimidine
        purines = seq_upper.count('A') + seq_upper.count('G')
        pyrimidines = seq_upper.count('C') + seq_upper.count('T')
        features['purine_freq'] = purines / length
        features['pyrimidine_freq'] = pyrimidines / length
        
        # 5.5 Region-specific composition
        # First 10 nt (5' region)
        if length >= 10:
            five_prime = seq_upper[:10]
            features['5prime_AT_content'] = (five_prime.count('A') + five_prime.count('T')) / 10
            features['5prime_GC_content'] = (five_prime.count('G') + five_prime.count('C')) / 10
        else:
            features['5prime_AT_content'] = features['AT_content']
            features['5prime_GC_content'] = features['GC_content']
        
        # Middle region
        if length > 20:
            middle = seq_upper[10:-10]
            features['middle_A_freq'] = middle.count('A') / len(middle)
            features['middle_G_freq'] = middle.count('G') / len(middle)
            features['middle_T_freq'] = middle.count('T') / len(middle)
        else:
            features['middle_A_freq'] = features['A_freq']
            features['middle_G_freq'] = features['G_freq']
            features['middle_T_freq'] = features['T_freq']
        
        # 5.6 Complexity measures
        features['num_unique_nucleotides'] = len(set(seq_upper))
        features['sequence_entropy'] = self._calculate_entropy(seq_upper)
        
        # 5.7 Runs and repeats
        features['max_homopolymer_length'] = self._max_homopolymer(seq_upper)
        features['has_long_run'] = int(features['max_homopolymer_length'] >= 4)
        
        return features
    
    def _calculate_entropy(self, seq: str) -> float:
        """Calculate Shannon entropy of sequence."""
        if len(seq) == 0:
            return 0.0
        counts = Counter(seq)
        probs = [count / len(seq) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)
    
    def _max_homopolymer(self, seq: str) -> int:
        """Find maximum homopolymer run length."""
        if len(seq) == 0:
            return 0
        max_len = 1
        current_len = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 1
        return max_len
    
    # =================================================================
    # 6. K-MER FEATURES (25 features)
    # =================================================================
    
    def _extract_kmer_features(self, seq: str) -> Dict[str, float]:
        """Extract k-mer based features."""
        features = {}
        
        seq_upper = seq.upper()
        
        # 6.1 Important 3-mers from Cooper analysis
        for kmer in self.important_3mers:
            features[f'has_3mer_{kmer}'] = int(kmer in seq_upper)
            features[f'count_3mer_{kmer}'] = seq_upper.count(kmer)
        
        # 6.2 K-mer diversity
        if len(seq) >= 2:
            dimers = [seq_upper[i:i+2] for i in range(len(seq)-1)]
            features['num_unique_2mers'] = len(set(dimers))
            features['2mer_diversity'] = len(set(dimers)) / len(dimers)
        else:
            features['num_unique_2mers'] = 0
            features['2mer_diversity'] = 0
        
        if len(seq) >= 3:
            trimers = [seq_upper[i:i+3] for i in range(len(seq)-2)]
            features['num_unique_3mers'] = len(set(trimers))
            features['3mer_diversity'] = len(set(trimers)) / len(trimers)
        else:
            features['num_unique_3mers'] = 0
            features['3mer_diversity'] = 0
        
        # 6.3 CpG dinucleotide (should be rare)
        if len(seq) >= 2:
            features['CpG_freq'] = seq_upper.count('CG') / (len(seq) - 1)
            features['CpG_is_rare'] = int(features['CpG_freq'] < 0.03)
        else:
            features['CpG_freq'] = 0
            features['CpG_is_rare'] = 1
        
        return features
    
    # =================================================================
    # 7. MOTIF AND PATTERN FEATURES (15 features)
    # =================================================================
    
    def _extract_motif_features(self, seq: str) -> Dict[str, float]:
        """Extract motif and pattern features."""
        features = {}
        
        seq_upper = seq.upper()
        
        # 7.1 Key motifs
        features['has_ATATA_motif'] = int('ATATA' in seq_upper)
        features['has_AAAA_motif'] = int('AAAA' in seq_upper)
        features['has_TTTT_motif'] = int('TTTT' in seq_upper)
        features['has_TTTTT_motif'] = int('TTTTT' in seq_upper)
        
        # 7.2 Poly tracts
        features['has_polyA'] = int('AAA' in seq_upper or 'AAAA' in seq_upper)
        features['has_polyT'] = int('TTTT' in seq_upper or 'TTTTT' in seq_upper)
        features['has_polyG'] = int('GGG' in seq_upper or 'GGGG' in seq_upper)
        features['has_polyC'] = int('CCC' in seq_upper or 'CCCC' in seq_upper)
        
        # 7.3 Alternating patterns
        features['has_TA_alternation'] = int('TATA' in seq_upper or 'ATAT' in seq_upper)
        features['has_CG_alternation'] = int('CGCG' in seq_upper or 'GCGC' in seq_upper)
        
        # 7.4 TA dinucleotide frequency (important in init region)
        if len(seq) >= 2:
            features['dinuc_TA_freq'] = seq_upper.count('TA') / (len(seq) - 1)
            features['dinuc_AT_freq'] = seq_upper.count('AT') / (len(seq) - 1)
        else:
            features['dinuc_TA_freq'] = 0
            features['dinuc_AT_freq'] = 0
        
        # 7.5 Repeat patterns
        features['num_TA_dinucs'] = seq_upper.count('TA')
        features['num_AT_dinucs'] = seq_upper.count('AT')
        features['num_AA_dinucs'] = seq_upper.count('AA')
        
        return features
    
    # =================================================================
    # 8. STRUCTURAL FEATURES (10 features)
    # =================================================================
    
    def _extract_structural_features(self, seq: str) -> Dict[str, float]:
        """Extract structural and complexity features."""
        features = {}
        
        seq_upper = seq.upper()
        
        # 8.1 Sequence complexity
        features['linguistic_complexity'] = self._linguistic_complexity(seq_upper)
        
        # 8.2 GC skew and AT skew
        g = seq_upper.count('G')
        c = seq_upper.count('C')
        a = seq_upper.count('A')
        t = seq_upper.count('T')
        
        features['GC_skew'] = (g - c) / (g + c) if (g + c) > 0 else 0
        features['AT_skew'] = (a - t) / (a + t) if (a + t) > 0 else 0
        
        # 8.3 Position-specific patterns
        # Check for AT-rich windows
        features['has_AT_rich_window'] = int(self._has_AT_rich_window(seq_upper))
        
        # 8.4 Length-independent features
        features['normalized_polyT_content'] = seq_upper.count('TTTT') / max(len(seq) - 3, 1)
        features['normalized_polyA_content'] = seq_upper.count('AAAA') / max(len(seq) - 3, 1)
        
        # 8.5 Stem-loop potential (simple estimate)
        features['potential_hairpin'] = int(self._has_hairpin_potential(seq_upper))
        
        # 8.6 Wobble pairing potential
        features['GU_pair_potential'] = (g + t) / len(seq) if len(seq) > 0 else 0
        
        # 8.7 Nucleotide diversity
        features['nucleotide_diversity'] = len(set(seq_upper)) / 4.0
        
        # 8.8 Runs of same type (purine/pyrimidine)
        features['max_purine_run'] = self._max_purine_run(seq_upper)
        
        return features
    
    def _linguistic_complexity(self, seq: str) -> float:
        """Calculate linguistic complexity (Trifonov measure)."""
        if len(seq) < 2:
            return 0.0
        
        observed = len(set([seq[i:i+2] for i in range(len(seq)-1)]))
        maximum = min(16, len(seq) - 1)  # 16 possible dinucleotides
        return observed / maximum if maximum > 0 else 0
    
    def _has_AT_rich_window(self, seq: str, window_size: int = 10, 
                           threshold: float = 0.70) -> bool:
        """Check for AT-rich sliding window."""
        if len(seq) < window_size:
            return False
        
        for i in range(len(seq) - window_size + 1):
            window = seq[i:i+window_size]
            at_content = (window.count('A') + window.count('T')) / window_size
            if at_content >= threshold:
                return True
        return False
    
    def _has_hairpin_potential(self, seq: str) -> bool:
        """Simple check for potential hairpin structure."""
        if len(seq) < 8:
            return False
        
        # Check for inverted repeats (simple check)
        for i in range(len(seq) - 7):
            segment = seq[i:i+4]
            reverse_comp = self._reverse_complement(segment)
            if reverse_comp in seq[i+4:i+8]:
                return True
        return False
    
    def _reverse_complement(self, seq: str) -> str:
        """Get reverse complement of sequence."""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        return ''.join(complement.get(nt, nt) for nt in reversed(seq))
    
    def _max_purine_run(self, seq: str) -> int:
        """Find maximum run of purines (A or G)."""
        max_run = 0
        current_run = 0
        for nt in seq:
            if nt in 'AG':
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run
    
    # =================================================================
    # 9. POSITIONAL FEATURES (10 features) - Optional
    # =================================================================
    
    def _extract_positional_features(self, seq: str, 
                                    cassette_info: Dict) -> Dict[str, float]:
        """
        Extract cassette-position dependent features.
        Requires cassette_info dict with keys: 'position', 'cassette_size', etc.
        """
        features = {}
        
        # Cassette position effects (from Cooper Table 3)
        position = cassette_info.get('position', 'unknown')
        features['cassette_position_I'] = int(position == 'I')
        features['cassette_position_II'] = int(position == 'II')
        features['cassette_position_IV'] = int(position == 'IV')
        features['cassette_position_V'] = int(position == 'V')
        
        # Cassette size
        cassette_size = cassette_info.get('cassette_size', 0)
        features['cassette_size'] = cassette_size
        features['cassette_size_canonical'] = int(139 <= cassette_size <= 146)
        
        # Distance from forward repeat
        distance_from_repeat = cassette_info.get('distance_from_repeat', 0)
        features['distance_from_repeat'] = distance_from_repeat
        features['in_init_region'] = int(30 <= distance_from_repeat <= 32)
        
        # Strand
        strand = cassette_info.get('strand', '+')
        features['is_sense_strand'] = int(strand == '+')
        
        # Expression status (if available)
        expression = cassette_info.get('expression', 'unknown')
        features['is_expressed'] = int(expression == 'expressed')
        
        return features
    
    # =================================================================
    # 10. INTERACTION FEATURES (10 features)
    # =================================================================
    
    def _extract_interaction_features(self, seq: str, 
                                     basic_features: Dict) -> Dict[str, float]:
        """
        Extract feature interactions/combinations.
        These capture important biological combinations.
        """
        features = {}
        
        # Key combinations from Cooper findings
        features['init_ATATA_and_T_end'] = int(
            basic_features.get('has_ATATA', 0) and 
            basic_features.get('ends_with_T', 0)
        )
        
        features['AC_anchor_and_AG_guide'] = int(
            basic_features.get('anchor_AC_rich', 0) and 
            basic_features.get('guide_A_is_high', 0)
        )
        
        features['correct_init_and_anchor_len'] = int(
            basic_features.get('matches_AWAHH', 0) and 
            basic_features.get('anchor_length_canonical', 0)
        )
        
        features['correct_init_anchor_combined'] = int(
            basic_features.get('matches_AWAHH', 0) and 
            basic_features.get('init_anchor_in_range', 0)
        )
        
        # Canonical structure signature
        features['has_canonical_structure'] = int(
            (basic_features.get('has_ATATA', 0) or basic_features.get('matches_AWAHH', 0)) and
            basic_features.get('anchor_AC_rich', 0) and
            basic_features.get('guide_A_is_high', 0) and
            basic_features.get('ends_with_T', 0)
        )
        
        # AT-rich init with AT-rich overall
        features['AT_rich_init_and_global'] = int(
            basic_features.get('5prime_AT_content', 0) > 0.70 and
            basic_features.get('AT_content', 0) > 0.60
        )
        
        # Low G in anchor and guide
        features['low_G_anchor_and_guide'] = int(
            basic_features.get('anchor_G_is_low', 0) and
            basic_features.get('guide_G_is_elevated', 0)
        )
        
        # Terminal T with poly-T
        features['T_end_with_polyT'] = int(
            basic_features.get('ends_with_T', 0) and
            basic_features.get('has_polyT', 0)
        )
        
        # Expressed profile (empirical combination)
        features['matches_expressed_profile'] = int(
            basic_features.get('matches_AWAHH', 0) and
            basic_features.get('anchor_length', 0) >= 7 and
            basic_features.get('guide_length', 0) >= 20 and
            basic_features.get('AT_content', 0) > 0.55
        )
        
        # Anti-pattern: signs of non-gRNA
        features['has_anti_pattern'] = int(
            basic_features.get('anchor_G_freq', 0) > 0.25 or
            basic_features.get('GC_content', 0) > 0.50 or
            (not basic_features.get('ends_with_T', 0))
        )
        
        return features
    
    # =================================================================
    # MAIN EXTRACTION METHOD
    # =================================================================
    
    def extract_all_features(self, seq: str, 
                            cassette_info: Optional[Dict] = None) -> Dict[str, float]:
        """
        Extract all features from a gRNA sequence.
        
        Args:
            seq: Nucleotide sequence (DNA or RNA)
            cassette_info: Optional dict with cassette metadata
            
        Returns:
            Dictionary with ~120-140 features
        """
        # Convert to uppercase and handle U->T
        seq = seq.upper().replace('U', 'T')
        
        # Initialize feature dict
        all_features = {}
        
        # Extract feature groups
        all_features.update(self._extract_init_features(seq))
        all_features.update(self._extract_anchor_features(seq))
        all_features.update(self._extract_guide_features(seq))
        all_features.update(self._extract_3prime_features(seq))
        all_features.update(self._extract_composition_features(seq))
        all_features.update(self._extract_kmer_features(seq))
        all_features.update(self._extract_motif_features(seq))
        all_features.update(self._extract_structural_features(seq))
        
        # Optional positional features
        if cassette_info is not None:
            all_features.update(self._extract_positional_features(seq, cassette_info))
        
        # Interaction features (use previously extracted features)
        all_features.update(self._extract_interaction_features(seq, all_features))
        
        return all_features
    
    def extract_features_batch(self, sequences: Dict[str, str],
                              cassette_info_dict: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
        """
        Extract features for multiple sequences.
        
        Args:
            sequences: Dict mapping sequence_id -> sequence
            cassette_info_dict: Optional dict mapping sequence_id -> cassette_info
            
        Returns:
            DataFrame with features for all sequences
        """
        feature_list = []
        
        for seq_id, seq in sequences.items():
            cassette_info = None
            if cassette_info_dict is not None:
                cassette_info = cassette_info_dict.get(seq_id)
            
            features = self.extract_all_features(seq, cassette_info)
            features['sequence_id'] = seq_id
            feature_list.append(features)
        
        return pd.DataFrame(feature_list)


if __name__ == "__main__":
    # Example usage
    extractor = gRNAFeatureExtractor()
    
    # Test sequence
    test_seq = "ATATAAAGAUUGAAGAAUGUGAUGUUAGACAAGUGAAACUACAAAAUACA"
    
    features = extractor.extract_all_features(test_seq)
    
    print(f"Extracted {len(features)} features")
    print("\nKey features:")
    for key in ['has_ATATA', 'matches_AWAHH', 'ends_with_T', 'anchor_AC_content', 
                'guide_A_freq', 'AT_content']:
        if key in features:
            print(f"  {key}: {features[key]:.3f}")
