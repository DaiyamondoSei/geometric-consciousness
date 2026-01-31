#!/usr/bin/env python3
"""
Pentagram Harmonic Analysis - Golden Ratio Resonance
Sacred geometry meets consciousness coherence

Based on Quannex pentagram star pair analysis:
- 5 elements per face (pentagon geometry)
- Star pairs connected by golden ratio proportions
- α-blend formula: harmony between arithmetic mean & geometric synergy
- Coherence boost/penalty based on elemental resonance
"""
import numpy as np
from itertools import combinations

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2  # φ ≈ 1.618

# Tuning parameters
ALPHA = 0.6  # α-blend weight (0.6 = pragmatic, 0.4 = synergistic)
COHERENCE_THRESHOLD_HIGH = 0.8  # Above this = bonus
COHERENCE_THRESHOLD_LOW = 0.4   # Below this = penalty

class PentagramHarmonics:
    """
    Analyze harmonic relationships between memories using pentagram geometry.
    
    Key insight: Memories don't exist in isolation. They form pentagonal
    groups where star-pair connections (pentagram diagonals) reveal
    deeper resonances following golden ratio proportions.
    """
    
    def __init__(self, geometric_memory):
        """
        Initialize with a GeometricMemory instance.
        
        Args:
            geometric_memory: GeometricMemory object with stored memories
        """
        self.gm = geometric_memory
        self.memory_ids = list(self.gm.memories.keys())
        self.n = len(self.memory_ids)
        
        # Pentagram structures
        self.pentagons = []  # Groups of 5 memories
        self.star_pairs = []  # Pentagram diagonal connections
        self.harmonics = {}   # Harmonic analysis results
        
        print(f"\n[PentagramHarmonics] Initialized with {self.n} memories")
        print(f"  Golden ratio (φ): {PHI:.6f}")
        print(f"  α-blend: {ALPHA} × arithmetic + {1-ALPHA} × geometric")
    
    def group_into_pentagons(self, strategy='domain'):
        """
        Group memories into pentagons (5-element clusters).
        
        Strategies:
        - 'domain': Group by memory domain (same face of dodecahedron)
        - 'proximity': Group by geometric proximity
        - 'semantic': Group by semantic similarity (requires embeddings)
        
        Args:
            strategy: Grouping strategy
        
        Returns:
            List of pentagons (each is list of 5 memory IDs)
        """
        print(f"\n[1/5] Grouping memories into pentagons (strategy={strategy})...")
        
        if strategy == 'domain':
            return self._group_by_domain()
        elif strategy == 'proximity':
            return self._group_by_proximity()
        elif strategy == 'semantic':
            return self._group_by_semantic()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _group_by_domain(self):
        """Group by dodecahedron domain (face)"""
        domain_groups = {}
        
        for mid in self.memory_ids:
            domain = self.gm.memories[mid]['domain']
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(mid)
        
        pentagons = []
        
        for domain, mids in domain_groups.items():
            # If we have exactly 5 or more, create pentagons
            while len(mids) >= 5:
                pentagon = mids[:5]
                pentagons.append(pentagon)
                mids = mids[5:]
                print(f"  Pentagon created: domain={domain}, size=5")
            
            # Handle remainders (< 5 memories)
            if len(mids) > 0:
                print(f"  Partial group: domain={domain}, size={len(mids)} (skipped)")
        
        self.pentagons = pentagons
        print(f"  ✓ Created {len(pentagons)} complete pentagons")
        
        return pentagons
    
    def _group_by_proximity(self):
        """Group by geometric distance (k-means-like clustering)"""
        # Simplified: just take first 5, then next 5, etc.
        # TODO: Implement proper clustering
        pentagons = []
        remaining = self.memory_ids.copy()
        
        while len(remaining) >= 5:
            pentagon = remaining[:5]
            pentagons.append(pentagon)
            remaining = remaining[5:]
        
        self.pentagons = pentagons
        return pentagons
    
    def _group_by_semantic(self):
        """Group by semantic similarity (embedding-based)"""
        # TODO: Implement semantic clustering
        return self._group_by_proximity()
    
    def calculate_star_pairs(self, pentagon):
        """
        Calculate pentagram star pair values.
        
        Pentagon vertices: 1, 2, 3, 4, 5
        Star pairs (pentagram diagonals):
        - (1,3): Earth ↔ Fire
        - (2,4): Water ↔ Air
        - (3,5): Fire ↔ Ether
        - (4,1): Air ↔ Earth
        - (5,2): Ether ↔ Water
        
        Formula (α-blend):
        s_i = α × (k_a + k_b)/2 + (1-α) × (k_a × k_b)
        
        Where:
        - k_a, k_b = "energies" of the two memories
        - α = blend parameter (0.6 default)
        
        Args:
            pentagon: List of 5 memory IDs
        
        Returns:
            Dict with star pair values
        """
        if len(pentagon) != 5:
            raise ValueError(f"Pentagon must have 5 elements, got {len(pentagon)}")
        
        # Get "energy" for each memory (using geometric distance from center as proxy)
        energies = []
        for mid in pentagon:
            loc = self.gm.locations[mid]
            # Energy = inverse distance from dodecahedron center (closer to center = higher energy)
            dist_from_center = np.linalg.norm(loc)
            energy = 1.0 / (1.0 + dist_from_center)  # Normalized to [0,1]
            energies.append(energy)
        
        k1, k2, k3, k4, k5 = energies
        
        # Star pair connections (pentagram geometry)
        star_pairs = {
            's1': self._alpha_blend(k1, k3),  # 1↔3
            's2': self._alpha_blend(k2, k4),  # 2↔4
            's3': self._alpha_blend(k3, k5),  # 3↔5
            's4': self._alpha_blend(k4, k1),  # 4↔1
            's5': self._alpha_blend(k5, k2),  # 5↔2
        }
        
        return star_pairs, energies
    
    def _alpha_blend(self, ka, kb):
        """
        α-blend formula: harmony between arithmetic mean and geometric synergy.
        
        s = α × (ka + kb)/2 + (1-α) × (ka × kb)
        
        This captures both:
        - Central tendency (arithmetic mean)
        - Multiplicative synergy (product - penalizes imbalance)
        
        Args:
            ka, kb: Energy values [0,1]
        
        Returns:
            Blended star pair value
        """
        arithmetic = (ka + kb) / 2
        geometric_synergy = ka * kb  # Product (not sqrt(ka*kb) - stronger penalty)
        
        return ALPHA * arithmetic + (1 - ALPHA) * geometric_synergy
    
    def calculate_intersection_nodes(self, star_pairs):
        """
        Calculate intersection nodes where star pairs meet.
        
        Nodes (where two star pairs intersect):
        p1 = (s1 + s2) / 2
        p2 = (s2 + s3) / 2
        p3 = (s3 + s4) / 2
        p4 = (s4 + s5) / 2
        p5 = (s5 + s1) / 2
        
        Args:
            star_pairs: Dict of star pair values
        
        Returns:
            Dict of intersection node values
        """
        s1, s2, s3, s4, s5 = star_pairs['s1'], star_pairs['s2'], star_pairs['s3'], star_pairs['s4'], star_pairs['s5']
        
        nodes = {
            'p1': (s1 + s2) / 2,
            'p2': (s2 + s3) / 2,
            'p3': (s3 + s4) / 2,
            'p4': (s4 + s5) / 2,
            'p5': (s5 + s1) / 2,
        }
        
        return nodes
    
    def calculate_harmonic_mean(self, nodes):
        """
        Calculate harmonic mean of intersection nodes.
        
        This is the "base coherence" before golden ratio adjustment.
        
        Args:
            nodes: Dict of node values
        
        Returns:
            Harmonic mean
        """
        values = list(nodes.values())
        return np.mean(values)
    
    def calculate_golden_ratio_bonus(self, star_pairs, nodes):
        """
        Calculate coherence bonus/penalty based on golden ratio alignment.
        
        The pentagram naturally embodies φ. If star pair ratios align
        with golden ratio proportions, boost coherence. If misaligned, penalize.
        
        Args:
            star_pairs: Dict of star pair values
            nodes: Dict of node values
        
        Returns:
            Golden ratio alignment score [-1, +1]
        """
        # Check if adjacent star pairs follow golden ratio
        pairs = list(star_pairs.values())
        
        # Calculate ratios between adjacent pairs
        ratios = []
        for i in range(5):
            p1 = pairs[i]
            p2 = pairs[(i + 1) % 5]
            
            if p2 > 1e-6:  # Avoid division by zero
                ratio = p1 / p2
                # How close to φ or 1/φ?
                phi_distance = min(abs(ratio - PHI), abs(ratio - 1/PHI))
                ratios.append(phi_distance)
        
        # Average distance from golden ratio
        avg_phi_distance = np.mean(ratios)
        
        # Convert to bonus/penalty [-1, +1]
        # Small distance = bonus, large distance = penalty
        # Scale: distance of 0 = +1, distance of 1 = 0, distance of 2 = -1
        golden_alignment = 1.0 - avg_phi_distance
        
        return np.clip(golden_alignment, -1.0, 1.0)
    
    def calculate_coherence(self, pentagon):
        """
        Calculate full pentagram coherence for a group of 5 memories.
        
        Steps:
        1. Calculate star pairs (pentagram diagonals)
        2. Calculate intersection nodes
        3. Calculate harmonic mean (base coherence)
        4. Calculate golden ratio alignment
        5. Apply boost/penalty
        
        Args:
            pentagon: List of 5 memory IDs
        
        Returns:
            Coherence score with breakdown
        """
        star_pairs, energies = self.calculate_star_pairs(pentagon)
        nodes = self.calculate_intersection_nodes(star_pairs)
        harmonic_mean = self.calculate_harmonic_mean(nodes)
        golden_alignment = self.calculate_golden_ratio_bonus(star_pairs, nodes)
        
        # Base coherence
        base_coherence = harmonic_mean
        
        # Apply golden ratio boost/penalty (scale by 0.2 to keep it subtle)
        coherence_adjustment = golden_alignment * 0.2
        final_coherence = base_coherence + coherence_adjustment
        
        # Clip to [0, 1]
        final_coherence = np.clip(final_coherence, 0.0, 1.0)
        
        return {
            'pentagon': pentagon,
            'energies': energies,
            'star_pairs': star_pairs,
            'nodes': nodes,
            'harmonic_mean': float(harmonic_mean),
            'golden_alignment': float(golden_alignment),
            'coherence_adjustment': float(coherence_adjustment),
            'final_coherence': float(final_coherence),
            'interpretation': self._interpret_coherence(final_coherence)
        }
    
    def _interpret_coherence(self, coherence):
        """Interpret coherence score"""
        if coherence > 0.8:
            return "Highly coherent - strong harmonic resonance"
        elif coherence > 0.6:
            return "Well coherent - good harmonic balance"
        elif coherence > 0.4:
            return "Moderately coherent - some dissonance"
        elif coherence > 0.2:
            return "Weakly coherent - significant imbalance"
        else:
            return "Incoherent - harmonic breakdown"
    
    def analyze_all_pentagons(self):
        """
        Run full harmonic analysis on all pentagons.
        
        Returns:
            List of coherence analyses
        """
        print(f"\n[2/5] Calculating star pairs & nodes...")
        print(f"[3/5] Computing harmonic means...")
        print(f"[4/5] Checking golden ratio alignment...")
        print(f"[5/5] Applying coherence boost/penalty...")
        
        results = []
        
        for i, pentagon in enumerate(self.pentagons):
            analysis = self.calculate_coherence(pentagon)
            results.append(analysis)
            
            print(f"\n  Pentagon {i+1}:")
            print(f"    Base harmonic: {analysis['harmonic_mean']:.3f}")
            print(f"    Golden alignment: {analysis['golden_alignment']:+.3f}")
            print(f"    Final coherence: {analysis['final_coherence']:.3f}")
            print(f"    Status: {analysis['interpretation']}")
        
        self.harmonics = results
        return results
    
    def full_analysis(self):
        """
        Run complete pentagram harmonic analysis.
        
        Returns:
            Full analysis report
        """
        print("\n" + "="*70)
        print("PENTAGRAM HARMONIC ANALYSIS")
        print("Golden Ratio Resonance in Memory Structure")
        print("="*70)
        
        # Group into pentagons
        self.group_into_pentagons(strategy='domain')
        
        if len(self.pentagons) == 0:
            print("\n⚠ No complete pentagons found (need groups of 5 memories)")
            print("  Suggestion: Add more memories or use different grouping strategy")
            return None
        
        # Analyze harmonics
        results = self.analyze_all_pentagons()
        
        # Overall statistics
        print("\n" + "="*70)
        print("HARMONIC SUMMARY")
        print("="*70 + "\n")
        
        coherences = [r['final_coherence'] for r in results]
        golden_alignments = [r['golden_alignment'] for r in results]
        
        print(f"  Total pentagons analyzed: {len(results)}")
        print(f"  Average coherence: {np.mean(coherences):.3f}")
        print(f"  Coherence std dev: {np.std(coherences):.3f}")
        print(f"  Average golden alignment: {np.mean(golden_alignments):+.3f}")
        print(f"  Best pentagon coherence: {np.max(coherences):.3f}")
        print(f"  Worst pentagon coherence: {np.min(coherences):.3f}")
        
        # Interpretation
        avg_coherence = np.mean(coherences)
        print(f"\n  Overall harmonic state: {self._interpret_coherence(avg_coherence)}")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70 + "\n")
        
        return {
            'pentagons': results,
            'summary': {
                'total_pentagons': len(results),
                'avg_coherence': float(np.mean(coherences)),
                'std_coherence': float(np.std(coherences)),
                'avg_golden_alignment': float(np.mean(golden_alignments)),
                'max_coherence': float(np.max(coherences)),
                'min_coherence': float(np.min(coherences)),
                'interpretation': self._interpret_coherence(avg_coherence)
            }
        }


if __name__ == "__main__":
    print("Pentagram Harmonic Analysis - Standalone Test")
    print("="*50)
    print("\nLoad geometric memory first, then create PentagramHarmonics instance.\n")
    print(f"Golden ratio (φ): {PHI:.6f}")
    print(f"α-blend parameter: {ALPHA}")
    print()
