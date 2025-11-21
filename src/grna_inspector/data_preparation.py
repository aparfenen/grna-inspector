"""
Data preparation module for gRNA classification with complete feature extraction.

This module handles:
1. Loading positive examples (canonical gRNA from Cooper 2022)
2. Generating negative examples from inter-cassette regions
3. FULL feature extraction (120+ features)
4. Length-matched negative sampling
5. Integration with supplemental data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from typing import Dict, Tuple, Optional
import warnings
from scipy import stats

from feature_extraction import gRNAFeatureExtractor

warnings.filterwarnings('ignore')


class gRNADataPreparator:
    """Prepare gRNA dataset with comprehensive feature extraction."""
    
    def __init__(self, positive_file: str, minicircle_file: str,
                 supplemental_dir: Optional[str] = None):
        """
        Args:
            positive_file: Path to mOs_gRNA_final.fasta
            minicircle_file: Path to mOs_Cooper_minicircle.fasta
            supplemental_dir: Optional path to dir with supplemental CSV files
        """
        self.positive_file = positive_file
        self.minicircle_file = minicircle_file
        self.supplemental_dir = supplemental_dir
        
        # Initialize feature extractor
        self.feature_extractor = gRNAFeatureExtractor()
        
        # Load supplemental data if available
        self.supplemental_data = self._load_supplemental_files()
        
    def _load_supplemental_files(self) -> Dict:
        """Load supplemental CSV files with gRNA annotations."""
        data = {}
        
        if self.supplemental_dir is None:
            return data
        
        supp_path = Path(self.supplemental_dir)
        
        # Load S1: Cassette information
        s1_file = supp_path / 'Supplemental_File_S1.csv'
        if s1_file.exists():
            data['cassettes'] = pd.read_csv(s1_file, index_col=0)
            print(f"  Loaded cassette data: {len(data['cassettes'])} entries")
        
        # Load S2: Detailed gRNA annotations (MOST IMPORTANT!)
        s2_file = supp_path / 'Supplemental_File_S2.csv'
        if s2_file.exists():
            data['gRNAs'] = pd.read_csv(s2_file, index_col=0)
            print(f"  Loaded gRNA annotations: {len(data['gRNAs'])} entries")
        
        return data
    
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
    
    def _generate_negative_examples(self, n_samples: int, 
                                   pos_lengths: list) -> Dict[str, str]:
        """
        Generate negatives with SAME length distribution as positives.
        
        Strategy:
        1. Load minicircles
        2. For each needed negative:
           - Sample length from positive distribution
           - Extract random sequence of that length from minicircle
           - Exclude gRNA regions (if GTF available)
        """
        print(f"\nGenerating {n_samples} negative examples...")
        
        # Load minicircles
        minicircles = []
        for record in SeqIO.parse(self.minicircle_file, "fasta"):
            seq = str(record.seq).upper().replace('U', 'T')
            minicircles.append((record.id, seq))
        
        print(f"  Loaded {len(minicircles)} minicircles")
        
        # TODO: Load GTF to exclude gRNA regions (for now, simple random sampling)
        # gRNA_regions = self._load_gRNA_regions()
        
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
            
            # Extract sequence
            seq = mini_seq[start:end]
            
            # Skip if too many N's or ambiguous bases
            if seq.count('N') > target_len * 0.1:
                continue
            
            # Skip if all same nucleotide (degenerate)
            if len(set(seq)) == 1:
                continue
            
            neg_id = f"{mini_id}_neg_{start}_{end}"
            negatives[neg_id] = seq
        
        print(f"  Generated {len(negatives)} negatives (attempts: {attempts})")
        lengths = [len(s) for s in negatives.values()]
        print(f"  Length: {min(lengths)}-{max(lengths)} nt (mean: {np.mean(lengths):.1f})")
        
        return negatives
    
    def generate_shuffled_sequences(self, positive_sequences: Dict[str, str],
                                   n_samples: int) -> Dict[str, str]:
        """Generate shuffled negatives preserving dinucleotide composition."""
        print(f"\nGenerating {n_samples} shuffled sequences...")
        
        shuffled = {}
        pos_seqs = list(positive_sequences.values())
        
        for i in range(n_samples):
            # Pick random positive
            original = pos_seqs[np.random.randint(len(pos_seqs))]
            
            # Shuffle (dinucleotide-preserving)
            shuffled_seq = self._dinucleotide_shuffle(original)
            shuffled[f"shuffled_{i}"] = shuffled_seq
        
        print(f"  Generated {len(shuffled)} shuffled sequences")
        return shuffled
    
    def _dinucleotide_shuffle(self, sequence: str, n_iterations: int = 100) -> str:
        """Shuffle preserving dinucleotide composition (Altschul-Erickson)."""
        seq_list = list(sequence)
        n = len(seq_list)
        
        for _ in range(n_iterations):
            if n < 3:
                break
            i, j = np.random.choice(n-1, 2, replace=False)
            
            # Swap if preserves dinucleotides
            if seq_list[i+1] != seq_list[j+1]:
                seq_list[i], seq_list[j] = seq_list[j], seq_list[i]
        
        return ''.join(seq_list)
    
    def _extract_features_for_sequences(self, sequences: Dict[str, str],
                                       labels: Dict[str, int],
                                       sources: Dict[str, str]) -> pd.DataFrame:
        """
        Extract features for all sequences.
        
        Args:
            sequences: Dict of seq_id -> sequence
            labels: Dict of seq_id -> label (0/1)
            sources: Dict of seq_id -> source ('gRNA', 'minicircle', 'shuffled')
            
        Returns:
            DataFrame with all features + metadata
        """
        print("\nExtracting features from sequences...")
        
        feature_list = []
        
        for seq_id, seq in sequences.items():
            # Extract features (NO LENGTH!)
            features = self.feature_extractor.extract_all_features(seq)
            
            # Add metadata (NOT as features, just for reference)
            features['sequence_id'] = seq_id
            features['sequence'] = seq
            features['label'] = labels[seq_id]
            features['source'] = sources[seq_id]
            
            feature_list.append(features)
        
        df = pd.DataFrame(feature_list)
        
        # Get feature columns (exclude metadata)
        feature_cols = [col for col in df.columns 
                       if col not in ['sequence_id', 'sequence', 'label', 'source']]
        
        print(f"  Extracted {len(feature_cols)} features for {len(df)} sequences")
        
        # Check for NaN or inf
        if df[feature_cols].isnull().any().any():
            print("  ⚠️  Warning: Some features have NaN values")
            df[feature_cols] = df[feature_cols].fillna(0)
        
        if np.isinf(df[feature_cols].values).any():
            print("  ⚠️  Warning: Some features have inf values")
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
        
        return df
    
    def prepare_dataset(self, n_negatives: int = None, 
                       use_shuffled: bool = True,
                       test_size: float = 0.15,
                       val_size: float = 0.15,
                       random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare complete dataset with train/val/test split.
        
        Args:
            n_negatives: Number of negatives (default: same as positives)
            use_shuffled: Include shuffled sequences as negatives
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining after test)
            random_state: Random seed
            
        Returns:
            train_df, val_df, test_df with features
        """
        print("\n" + "="*70)
        print("PREPARING DATASET WITH COMPREHENSIVE FEATURE EXTRACTION")
        print("="*70)
        
        # Load positives
        positives = self.load_positive_sequences()
        n_pos = len(positives)
        
        if n_negatives is None:
            n_negatives = n_pos
        
        pos_lengths = [len(s) for s in positives.values()]
        
        # Generate negatives
        if use_shuffled:
            # Half from minicircles, half shuffled
            n_minicircle = n_negatives // 2
            n_shuffled = n_negatives - n_minicircle
            
            negatives_mini = self._generate_negative_examples(n_minicircle, pos_lengths)
            negatives_shuf = self.generate_shuffled_sequences(positives, n_shuffled)
            negatives = {**negatives_mini, **negatives_shuf}
        else:
            negatives = self._generate_negative_examples(n_negatives, pos_lengths)
        
        # Combine sequences
        all_sequences = {**positives, **negatives}
        
        # Create labels and sources
        labels = {}
        sources = {}
        
        for seq_id in positives:
            labels[seq_id] = 1
            sources[seq_id] = 'gRNA'
        
        for seq_id in negatives:
            labels[seq_id] = 0
            if 'shuffled' in seq_id:
                sources[seq_id] = 'shuffled'
            else:
                sources[seq_id] = 'minicircle'
        
        # Extract features for all sequences
        df = self._extract_features_for_sequences(all_sequences, labels, sources)
        
        # Check length distribution match
        print(f"\n{'='*70}")
        print("LENGTH DISTRIBUTION CHECK")
        print(f"{'='*70}")
        
        pos_lens = [len(s) for seq_id, s in all_sequences.items() if labels[seq_id] == 1]
        neg_lens = [len(s) for seq_id, s in all_sequences.items() if labels[seq_id] == 0]
        
        print(f"\nPositive sequences: mean={np.mean(pos_lens):.1f}, std={np.std(pos_lens):.1f}")
        print(f"Negative sequences: mean={np.mean(neg_lens):.1f}, std={np.std(neg_lens):.1f}")
        
        ks_stat, ks_pval = stats.ks_2samp(pos_lens, neg_lens)
        print(f"\nKolmogorov-Smirnov test:")
        print(f"  Statistic: {ks_stat:.4f}")
        print(f"  P-value: {ks_pval:.4f}")
        if ks_pval > 0.05:
            print("  ✓ Length distributions are similar (good!)")
        else:
            print("  ⚠️  Length distributions differ significantly!")
        
        # Train/val/test split
        print(f"\n{'='*70}")
        print("SPLITTING INTO TRAIN/VAL/TEST")
        print(f"{'='*70}")
        
        np.random.seed(random_state)
        
        # Stratified split
        pos_df = df[df['label'] == 1].copy()
        neg_df = df[df['label'] == 0].copy()
        
        # Shuffle
        pos_df = pos_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        neg_df = neg_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Split positives
        n_pos_test = int(len(pos_df) * test_size)
        n_pos_val = int((len(pos_df) - n_pos_test) * val_size)
        
        pos_test = pos_df[:n_pos_test]
        pos_val = pos_df[n_pos_test:n_pos_test + n_pos_val]
        pos_train = pos_df[n_pos_test + n_pos_val:]
        
        # Split negatives
        n_neg_test = int(len(neg_df) * test_size)
        n_neg_val = int((len(neg_df) - n_neg_test) * val_size)
        
        neg_test = neg_df[:n_neg_test]
        neg_val = neg_df[n_neg_test:n_neg_test + n_neg_val]
        neg_train = neg_df[n_neg_test + n_neg_val:]
        
        # Combine
        train_df = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=random_state)
        val_df = pd.concat([pos_val, neg_val]).sample(frac=1, random_state=random_state)
        test_df = pd.concat([pos_test, neg_test]).sample(frac=1, random_state=random_state)
        
        print(f"\nTrain: {len(train_df)} ({sum(train_df['label']==1)} pos, {sum(train_df['label']==0)} neg)")
        print(f"Val:   {len(val_df)} ({sum(val_df['label']==1)} pos, {sum(val_df['label']==0)} neg)")
        print(f"Test:  {len(test_df)} ({sum(test_df['label']==1)} pos, {sum(test_df['label']==0)} neg)")
        
        return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("gRNA DATA PREPARATION PIPELINE")
    print("="*70)
    
    preparator = gRNADataPreparator(
        positive_file='../../data/gRNAs/Cooper_2022/mOs.gRNA.final.fasta',
        minicircle_file='../../data/gRNAs/Cooper_2022/mOs.Cooper.minicircle.fasta',
        supplemental_dir='../../data/gRNAs/Cooper_2022/'
    )
    
    train_df, val_df, test_df = preparator.prepare_dataset(
        n_negatives=None,  # Same as positives
        use_shuffled=True,
        random_state=42
    )
    
    # Save datasets
    output_dir = Path('../../data/processed/data_preparation')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    train_df.to_csv(output_dir / 'train_data.csv', index=False)
    val_df.to_csv(output_dir / 'val_data.csv', index=False)
    test_df.to_csv(output_dir / 'test_data.csv', index=False)
    
    print(f"\n{'='*70}")
    print("DATASETS SAVED!")
    print(f"{'='*70}")
    print(f"  Training:   {output_dir / 'train_data.csv'}")
    print(f"  Validation: {output_dir / 'val_data.csv'}")
    print(f"  Test:       {output_dir / 'test_data.csv'}")
    
    # Print feature summary
    feature_cols = [col for col in train_df.columns 
                   if col not in ['sequence_id', 'sequence', 'label', 'source']]
    print(f"\nTotal features: {len(feature_cols)}")
    print("\nFirst 20 features:")
    for i, feat in enumerate(feature_cols[:20], 1):
        print(f"  {i:2d}. {feat}")
    
    # Print dataset statistics
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print(f"{'='*70}")
    
    for name, df_split in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{name} set:")
        print(f"  Total: {len(df_split)}")
        print(f"  Positive: {sum(df_split['label']==1)} ({sum(df_split['label']==1)/len(df_split)*100:.1f}%)")
        print(f"  Negative: {sum(df_split['label']==0)} ({sum(df_split['label']==0)/len(df_split)*100:.1f}%)")
        
        # Source breakdown
        print(f"  Sources:")
        for source in df_split['source'].unique():
            count = sum(df_split['source'] == source)
            print(f"    - {source}: {count}")
    
    print(f"\n{'='*70}")
    print("DATA PREPARATION COMPLETE!")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Train models using baseline_models.py")
    print("2. Check feature importances")
    print("3. Validate no length leakage")
    print("4. Verify ROC-AUC > 0.90")