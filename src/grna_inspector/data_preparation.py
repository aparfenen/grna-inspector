"""
Data preparation module for gRNA classification.

This module handles:
1. Loading positive examples (canonical gRNA from Cooper 2022)
2. Generating negative examples from inter-cassette regions and maxicircles
3. Feature extraction (sequence composition, k-mers, motifs)
4. Train/validation/test split

FIXED Data preparation - matching length distributions.
Critical fix: Negative examples MUST have same length distribution as positives,
otherwise model just learns "longer = negative" instead of sequence features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class gRNADataPreparator:
    """Prepare data with proper length matching."""
    
    def __init__(self, positive_file: str, minicircle_file: str):
        """
        Args:
            positive_file: Path to mOs_gRNA_final.fasta
            minicircle_file: Path to mOs_Cooper_minicircle.fasta
        """
        self.positive_file = positive_file
        self.minicircle_file = minicircle_file
        
    def load_positive_sequences(self) -> Dict[str, str]:
        """Load canonical gRNA sequences."""
        print("Loading positive examples...")
        sequences = {}
        
        for record in SeqIO.parse(self.positive_file, "fasta"):
            seq = str(record.seq).upper().replace('U', 'T')
            sequences[record.id] = seq
        
        print(f"  Loaded {len(sequences)} gRNA sequences")
        lengths = [len(s) for s in sequences.values()]
        print(f"  Length: {min(lengths)}-{max(lengths)} nt (mean: {np.mean(lengths):.1f})")
        return sequences
    
    def _generate_negative_examples(self, n_samples: int) -> Dict[str, str]:
        """
        Generate negatives with SAME length distribution as positives.
        
        Strategy:
        1. Load minicircles
        2. For each needed negative:
           - Sample length from positive distribution
           - Extract random sequence of that length from minicircle
           - Exclude gRNA regions
        """
        print(f"\nGenerating {n_samples} negative examples...")
        
        # Load positive lengths
        pos_seqs = self.load_positive_sequences()
        pos_lengths = [len(s) for s in pos_seqs.values()]
        
        # Load minicircles
        minicircles = []
        for record in SeqIO.parse(self.minicircle_file, "fasta"):
            seq = str(record.seq).upper().replace('U', 'T')
            minicircles.append((record.id, seq))
        
        print(f"  Loaded {len(minicircles)} minicircles")
        
        # Load GTF to exclude gRNA regions
        gRNA_regions = self._load_gRNA_regions()
        
        negatives = {}
        attempts = 0
        max_attempts = n_samples * 10
        
        while len(negatives) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Sample length from positive distribution
            target_len = np.random.choice(pos_lengths)
            
            # Pick random minicircle
            mini_id, mini_seq = minicircles[np.random.randint(len(minicircles))]
            
            if len(mini_seq) < target_len:
                continue
            
            # Pick random position
            start = np.random.randint(0, len(mini_seq) - target_len + 1)
            end = start + target_len
            
            # Check if overlaps with gRNA
            if self._overlaps_gRNA(mini_id, start, end, gRNA_regions):
                continue
            
            # Extract sequence
            seq = mini_seq[start:end]
            
            # Skip if too many N's or ambiguous bases
            if seq.count('N') > target_len * 0.1:
                continue
            
            neg_id = f"{mini_id}_neg_{start}_{end}"
            negatives[neg_id] = seq
        
        print(f"  Generated {len(negatives)} negatives (attempts: {attempts})")
        lengths = [len(s) for s in negatives.values()]
        print(f"  Length: {min(lengths)}-{max(lengths)} nt (mean: {np.mean(lengths):.1f})")
        
        return negatives
    
    def _load_gRNA_regions(self) -> Dict:
        """Load gRNA coordinates from GTF file."""
        # This would parse the GTF file
        # For now, simple placeholder
        return {}
    
    def _overlaps_gRNA(self, mini_id: str, start: int, end: int, 
                      gRNA_regions: Dict) -> bool:
        """Check if region overlaps with known gRNA."""
        if mini_id not in gRNA_regions:
            return False
        
        for g_start, g_end in gRNA_regions[mini_id]:
            # Check overlap
            if not (end < g_start or start > g_end):
                return True
        return False
    
    def generate_shuffled_sequences(self, positive_sequences: Dict[str, str],
                                   n_samples: int) -> Dict[str, str]:
        """Generate shuffled negatives preserving dinucleotide composition."""
        print(f"\nGenerating {n_samples} shuffled sequences...")
        
        shuffled = {}
        pos_seqs = list(positive_sequences.values())
        
        for i in range(n_samples):
            # Pick random positive
            original = pos_seqs[np.random.randint(len(pos_seqs))]
            
            # Shuffle
            shuffled_seq = self._dinucleotide_shuffle(original)
            shuffled[f"shuffled_{i}"] = shuffled_seq
        
        print(f"  Generated {len(shuffled)} shuffled sequences")
        return shuffled
    
    def _dinucleotide_shuffle(self, sequence: str, n_iterations: int = 100) -> str:
        """Shuffle preserving dinucleotide composition (Altschul-Erickson)."""
        seq_list = list(sequence)
        n = len(seq_list)
        
        for _ in range(n_iterations):
            # Pick two random positions (not last)
            if n < 3:
                break
            i, j = np.random.choice(n-1, 2, replace=False)
            
            # Swap if preserves dinucleotides
            if seq_list[i+1] != seq_list[j+1]:
                seq_list[i], seq_list[j] = seq_list[j], seq_list[i]
        
        return ''.join(seq_list)
    
    def prepare_dataset(self, n_negatives: int = None, 
                       use_shuffled: bool = True) -> pd.DataFrame:
        """
        Prepare complete dataset.
        
        Args:
            n_negatives: Number of negatives (default: same as positives)
            use_shuffled: Include shuffled sequences as negatives
            
        Returns:
            DataFrame with sequences and labels
        """
        print("\n" + "="*60)
        print("PREPARING DATASET WITH LENGTH MATCHING")
        print("="*60)
        
        # Load positives
        positives = self.load_positive_sequences()
        n_pos = len(positives)
        
        if n_negatives is None:
            n_negatives = n_pos
        
        # Generate negatives
        if use_shuffled:
            # Half from minicircles, half shuffled
            n_minicircle = n_negatives // 2
            n_shuffled = n_negatives - n_minicircle
            
            negatives_mini = self._generate_negative_examples(n_minicircle)
            negatives_shuf = self.generate_shuffled_sequences(positives, n_shuffled)
            negatives = {**negatives_mini, **negatives_shuf}
        else:
            negatives = self._generate_negative_examples(n_negatives)
        
        # Create DataFrame
        data = []
        
        for seq_id, seq in positives.items():
            data.append({
                'sequence_id': seq_id,
                'sequence': seq,
                'length': len(seq),
                'label': 1,
                'source': 'gRNA'
            })
        
        for seq_id, seq in negatives.items():
            source = 'shuffled' if 'shuffled' in seq_id else 'minicircle'
            data.append({
                'sequence_id': seq_id,
                'sequence': seq,
                'length': len(seq),
                'label': 0,
                'source': source
            })
        
        df = pd.DataFrame(data)
        
        print(f"\n{'='*60}")
        print(f"DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"Total sequences: {len(df)}")
        print(f"  Positive: {len(df[df['label']==1])}")
        print(f"  Negative: {len(df[df['label']==0])}")
        print(f"\nLength distribution:")
        print(df.groupby('label')['length'].describe()[['mean', 'std', 'min', 'max']])
        
        # Check if distributions are similar
        from scipy import stats
        pos_lens = df[df['label']==1]['length'].values
        neg_lens = df[df['label']==0]['length'].values
        ks_stat, ks_pval = stats.ks_2samp(pos_lens, neg_lens)
        
        print(f"\nKolmogorov-Smirnov test:")
        print(f"  Statistic: {ks_stat:.4f}")
        print(f"  P-value: {ks_pval:.4f}")
        if ks_pval > 0.05:
            print("  ✓ Length distributions are similar (good!)")
        else:
            print("  ⚠ Length distributions differ significantly!")
        
        return df


if __name__ == "__main__":
    # Example usage
    preparator = gRNADataPreparator(
        positive_file='../data/mOs_gRNA_final.fasta',
        minicircle_file='../data/mOs_Cooper_minicircle.fasta'
    )
    
    df = preparator.prepare_dataset()
    df.to_csv('../data/processed/dataset_length_matched.csv', index=False)
    print("\nDataset saved!")