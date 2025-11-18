"""
Data preparation module for gRNA classification.

This module handles:
1. Loading positive examples (canonical gRNA from Cooper 2022)
2. Generating negative examples from inter-cassette regions and maxicircles
3. Feature extraction (sequence composition, k-mers, motifs)
4. Train/validation/test split
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from collections import Counter
import re
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class gRNADataPreparator:
    """Prepare data for gRNA classification."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize data preparator.
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir)
        self.cooper_dir = self.data_dir / "gRNAs" / "Cooper_2022"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
        # Load annotations
        self.s2 = pd.read_csv(self.cooper_dir / "Supplemental_File_S2.csv", index_col=0)
        self.s4 = pd.read_csv(self.cooper_dir / "Supplemental_File_S4.csv", index_col=0)
        
    def load_positive_sequences(self) -> Dict[str, str]:
        """
        Load canonical gRNA sequences.
        
        Returns:
            Dictionary mapping sequence ID to sequence
        """
        print("Loading positive examples (canonical gRNA)...")
        sequences = {}
        
        fasta_file = self.cooper_dir / "mOs.gRNA.final.fasta"
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences[record.id] = str(record.seq).upper()
        
        print(f"  Loaded {len(sequences)} canonical gRNA sequences")
        return sequences
    
    def extract_inter_cassette_regions(self, min_length: int = 30, 
                                       max_length: int = 60) -> Dict[str, str]:
        """
        Extract sequences from inter-cassette regions as negative examples.
        
        Args:
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            
        Returns:
            Dictionary of negative sequences
        """
        print("Extracting inter-cassette regions...")
        
        negatives = {}
        minicircle_file = self.cooper_dir / "mOs.Cooper.minicircle.fasta"
        
        # Load GTF annotations to find cassette positions
        gtf_file = self.cooper_dir / "mOs.gRNA.final.gtf"
        annotations = self._parse_gtf(gtf_file)
        
        # For each minicircle, extract regions between cassettes
        for record in SeqIO.parse(minicircle_file, "fasta"):
            minicircle_id = record.id.split()[0]
            sequence = str(record.seq).upper()
            
            # Get all gRNA positions on this minicircle
            if minicircle_id not in annotations:
                continue
                
            positions = sorted([(int(ann['start']), int(ann['end'])) 
                              for ann in annotations[minicircle_id]])
            
            # Extract inter-cassette regions
            for i in range(len(positions) - 1):
                start = positions[i][1] + 1
                end = positions[i + 1][0] - 1
                
                if end - start >= min_length:
                    # Extract random fragments from this region
                    for _ in range(2):  # 2 fragments per inter-cassette region
                        frag_len = np.random.randint(min_length, 
                                                    min(max_length, end - start))
                        frag_start = np.random.randint(start, end - frag_len)
                        fragment = sequence[frag_start:frag_start + frag_len]
                        
                        neg_id = f"{minicircle_id}_inter_{frag_start}_{frag_start+frag_len}"
                        negatives[neg_id] = fragment
        
        print(f"  Extracted {len(negatives)} inter-cassette sequences")
        return negatives
    
    def extract_maxicircle_sequences(self, num_sequences: int = 500,
                                    min_length: int = 30,
                                    max_length: int = 60) -> Dict[str, str]:
        """
        Extract random sequences from maxicircles as negative examples.
        
        Args:
            num_sequences: Number of sequences to extract
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            
        Returns:
            Dictionary of negative sequences
        """
        print("Extracting maxicircle sequences...")
        
        negatives = {}
        maxicircle_files = list((self.data_dir.parent / "data_dump" / "data-deposit" / "maxcircle").glob("*.fasta"))
        
        if not maxicircle_files:
            print("  Warning: No maxicircle files found")
            return negatives
        
        all_sequences = []
        for maxi_file in maxicircle_files:
            for record in SeqIO.parse(maxi_file, "fasta"):
                all_sequences.append((maxi_file.stem, str(record.seq).upper()))
        
        # Extract random fragments
        for i in range(num_sequences):
            source, sequence = all_sequences[np.random.randint(len(all_sequences))]
            frag_len = np.random.randint(min_length, max_length)
            
            if len(sequence) < frag_len:
                continue
                
            start = np.random.randint(0, len(sequence) - frag_len)
            fragment = sequence[start:start + frag_len]
            
            neg_id = f"maxi_{source}_{i}_{start}_{start+frag_len}"
            negatives[neg_id] = fragment
        
        print(f"  Extracted {len(negatives)} maxicircle sequences")
        return negatives
    
    def generate_shuffled_sequences(self, positive_sequences: Dict[str, str],
                                   num_sequences: int = 300) -> Dict[str, str]:
        """
        Generate shuffled sequences preserving dinucleotide composition.
        
        Args:
            positive_sequences: Dictionary of positive sequences
            num_sequences: Number of shuffled sequences to generate
            
        Returns:
            Dictionary of shuffled sequences
        """
        print("Generating shuffled sequences...")
        
        shuffled = {}
        pos_seqs = list(positive_sequences.values())
        
        for i in range(num_sequences):
            # Pick a random positive sequence
            original = pos_seqs[np.random.randint(len(pos_seqs))]
            
            # Shuffle preserving dinucleotide frequency
            shuffled_seq = self._dinucleotide_shuffle(original)
            shuffled[f"shuffled_{i}"] = shuffled_seq
        
        print(f"  Generated {len(shuffled)} shuffled sequences")
        return shuffled
    
    def _dinucleotide_shuffle(self, sequence: str, max_iterations: int = 100) -> str:
        """
        Shuffle sequence preserving dinucleotide composition.
        Uses Altschul-Erickson algorithm.
        """
        seq_list = list(sequence)
        n = len(seq_list)
        
        for _ in range(max_iterations):
            # Pick two random positions
            i, j = np.random.choice(n-1, 2, replace=False)
            
            # Swap if it preserves dinucleotide composition
            if seq_list[i+1] == seq_list[j+1]:
                continue
            
            # Swap nucleotides at positions i and j
            seq_list[i], seq_list[j] = seq_list[j], seq_list[i]
        
        return ''.join(seq_list)
    
    def _parse_gtf(self, gtf_file: Path) -> Dict[str, List[Dict]]:
        """Parse GTF file to extract gRNA positions."""
        annotations = {}
        
        with open(gtf_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                
                minicircle_id = fields[0]
                start = fields[3]
                end = fields[4]
                
                if minicircle_id not in annotations:
                    annotations[minicircle_id] = []
                
                annotations[minicircle_id].append({
                    'start': start,
                    'end': end
                })
        
        return annotations
    
    def compute_sequence_features(self, sequences: Dict[str, str]) -> pd.DataFrame:
        """
        Compute features for sequences.
        
        Features include:
        - Length
        - Nucleotide composition (A, T, G, C percentages)
        - GC content
        - Di-nucleotide frequencies
        - Tri-nucleotide frequencies
        - Presence of initiation motifs
        - Entropy
        
        Args:
            sequences: Dictionary of sequences
            
        Returns:
            DataFrame with features
        """
        print("Computing sequence features...")
        
        features_list = []
        
        for seq_id, sequence in sequences.items():
            features = self._extract_features(sequence)
            features['sequence_id'] = seq_id
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        print(f"  Computed {len(df)} feature vectors with {len(df.columns)-1} features each")
        
        return df
    
    def _extract_features(self, sequence: str) -> Dict:
        """Extract features from a single sequence."""
        length = len(sequence)
        
        # Nucleotide counts
        counts = Counter(sequence)
        a_count = counts.get('A', 0)
        t_count = counts.get('T', 0)
        g_count = counts.get('G', 0)
        c_count = counts.get('C', 0)
        
        # Basic features
        features = {
            'length': length,
            'a_content': a_count / length * 100,
            't_content': t_count / length * 100,
            'g_content': g_count / length * 100,
            'c_content': c_count / length * 100,
            'gc_content': (g_count + c_count) / length * 100,
            'at_content': (a_count + t_count) / length * 100,
            'purine_content': (a_count + g_count) / length * 100,
        }
        
        # Di-nucleotide frequencies
        dinucs = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC',
                  'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
        for dinuc in dinucs:
            count = sequence.count(dinuc)
            features[f'dinuc_{dinuc}'] = count / (length - 1) * 100 if length > 1 else 0
        
        # Tri-nucleotide frequencies (selected important ones)
        trinucs = ['ATA', 'TAT', 'AAA', 'TTT', 'GGG', 'CCC', 
                   'ATG', 'TGA', 'CAT', 'GCA']
        for trinuc in trinucs:
            count = sequence.count(trinuc)
            features[f'trinuc_{trinuc}'] = count / (length - 2) * 100 if length > 2 else 0
        
        # Initiation motifs (from Cooper paper)
        motifs = ['ATATA', 'ATACA', 'AAATA', 'ATAAA', 'ATATT']
        for motif in motifs:
            features[f'has_motif_{motif}'] = 1 if motif in sequence else 0
        
        # Position-specific composition (first 10, middle, last 10 nt)
        if length >= 20:
            first10 = sequence[:10]
            last10 = sequence[-10:]
            middle = sequence[10:-10] if length > 20 else ''
            
            features['first10_gc'] = (first10.count('G') + first10.count('C')) / 10 * 100
            features['last10_gc'] = (last10.count('G') + last10.count('C')) / 10 * 100
            if middle:
                features['middle_gc'] = (middle.count('G') + middle.count('C')) / len(middle) * 100
            else:
                features['middle_gc'] = 0
        else:
            features['first10_gc'] = 0
            features['last10_gc'] = 0
            features['middle_gc'] = 0
        
        # Entropy
        features['entropy'] = self._calculate_entropy(sequence)
        
        return features
    
    def _calculate_entropy(self, sequence: str) -> float:
        """Calculate Shannon entropy of sequence."""
        if not sequence:
            return 0.0
        
        counts = Counter(sequence)
        probs = [count / len(sequence) for count in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        return entropy
    
    def prepare_dataset(self, test_size: float = 0.15, 
                       val_size: float = 0.15,
                       balance_ratio: float = 1.0,
                       random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare complete dataset with train/val/test splits.
        
        Args:
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set
            balance_ratio: Ratio of negatives to positives (1.0 = balanced)
            random_state: Random seed
            
        Returns:
            train_df, val_df, test_df
        """
        np.random.seed(random_state)
        
        print("\n" + "="*60)
        print("PREPARING gRNA CLASSIFICATION DATASET")
        print("="*60 + "\n")
        
        # Load positive sequences
        positive_seqs = self.load_positive_sequences()
        
        # Generate negative sequences
        inter_cassette = self.extract_inter_cassette_regions()
        maxicircle = self.extract_maxicircle_sequences()
        shuffled = self.generate_shuffled_sequences(positive_seqs)
        
        # Combine all negatives
        all_negatives = {**inter_cassette, **maxicircle, **shuffled}
        
        # Balance dataset
        n_positives = len(positive_seqs)
        n_negatives = int(n_positives * balance_ratio)
        
        if len(all_negatives) > n_negatives:
            # Randomly sample negatives
            negative_ids = np.random.choice(list(all_negatives.keys()), 
                                           n_negatives, replace=False)
            negatives = {k: all_negatives[k] for k in negative_ids}
        else:
            negatives = all_negatives
        
        print(f"\nDataset composition:")
        print(f"  Positives: {len(positive_seqs)}")
        print(f"  Negatives: {len(negatives)}")
        print(f"  Ratio: 1:{len(negatives)/len(positive_seqs):.2f}")
        
        # Compute features
        pos_features = self.compute_sequence_features(positive_seqs)
        pos_features['label'] = 1
        pos_features['source'] = 'canonical_gRNA'
        
        neg_features = self.compute_sequence_features(negatives)
        neg_features['label'] = 0
        neg_features['source'] = neg_features['sequence_id'].apply(
            lambda x: 'inter_cassette' if 'inter' in x 
                     else ('maxicircle' if 'maxi' in x else 'shuffled')
        )
        
        # Combine
        all_features = pd.concat([pos_features, neg_features], ignore_index=True)
        
        # Add sequences
        all_sequences = {**positive_seqs, **negatives}
        all_features['sequence'] = all_features['sequence_id'].map(all_sequences)
        
        # Shuffle
        all_features = all_features.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Split into train/val/test
        n_total = len(all_features)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val
        
        test_df = all_features[:n_test].copy()
        val_df = all_features[n_test:n_test+n_val].copy()
        train_df = all_features[n_test+n_val:].copy()
        
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_df)} ({len(train_df[train_df['label']==1])} pos, {len(train_df[train_df['label']==0])} neg)")
        print(f"  Val:   {len(val_df)} ({len(val_df[val_df['label']==1])} pos, {len(val_df[val_df['label']==0])} neg)")
        print(f"  Test:  {len(test_df)} ({len(test_df[test_df['label']==1])} pos, {len(test_df[test_df['label']==0])} neg)")
        
        # Save datasets
        train_df.to_csv(self.processed_dir / "train_data.csv", index=False)
        val_df.to_csv(self.processed_dir / "val_data.csv", index=False)
        test_df.to_csv(self.processed_dir / "test_data.csv", index=False)
        
        print(f"\nDatasets saved to {self.processed_dir}")
        print("="*60 + "\n")
        
        return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    data_dir = Path("../data")
    preparator = gRNADataPreparator(data_dir)
    train_df, val_df, test_df = preparator.prepare_dataset()
    
    print("Data preparation complete!")
