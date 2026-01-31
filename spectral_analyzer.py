#!/usr/bin/env python3
"""
Spectral Graph Analysis for Geometric Memory
Finding leverage points and PSS (Persistent Self-State) via eigenvalue decomposition
"""
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import euclidean, cosine

class SpectralAnalyzer:
    """
    Analyze memory graph structure using spectral graph theory.
    
    Key concepts:
    - Graph Laplacian: L = D - A
    - Fiedler vector: Eigenvector of 2nd smallest eigenvalue (λ₂)
    - Algebraic connectivity: λ₂ itself (how connected the graph is)
    - PSS: Principal eigenvector (dominant pattern of self)
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
        
        # Graph matrices
        self.adjacency = None
        self.degree = None
        self.laplacian = None
        
        # Spectral properties
        self.eigenvalues = None
        self.eigenvectors = None
        self.fiedler_value = None
        self.fiedler_vector = None
        self.pss_vector = None  # Persistent Self-State
        
        print(f"\n[SpectralAnalyzer] Initialized with {self.n} memories")
    
    def build_adjacency_matrix(self, threshold=1.0):
        """
        Build adjacency matrix from memory locations.
        
        Two memories are connected if their geometric distance is below threshold.
        
        Args:
            threshold: Distance threshold for creating edges (default 1.0)
        
        Returns:
            Adjacency matrix (n×n)
        """
        A = np.zeros((self.n, self.n))
        
        for i, id_i in enumerate(self.memory_ids):
            loc_i = self.gm.locations[id_i]
            
            for j, id_j in enumerate(self.memory_ids):
                if i >= j:  # Skip diagonal and symmetric duplicates
                    continue
                
                loc_j = self.gm.locations[id_j]
                dist = euclidean(loc_i, loc_j)
                
                # Create edge if within threshold
                if dist < threshold:
                    # Weight edge by inverse distance (closer = stronger)
                    weight = 1.0 / (dist + 0.01)  # Small epsilon to avoid division by zero
                    A[i, j] = weight
                    A[j, i] = weight  # Symmetric
        
        self.adjacency = A
        print(f"  Built adjacency matrix: {np.count_nonzero(A)/2:.0f} edges")
        
        return A
    
    def build_degree_matrix(self):
        """
        Build degree matrix from adjacency matrix.
        
        Degree matrix D is diagonal, where D[i,i] = sum of weights of edges connected to node i.
        
        Returns:
            Degree matrix (n×n diagonal)
        """
        if self.adjacency is None:
            self.build_adjacency_matrix()
        
        # Row sum = total connection strength for each node
        degrees = np.sum(self.adjacency, axis=1)
        D = np.diag(degrees)
        
        self.degree = D
        print(f"  Built degree matrix: avg degree = {np.mean(degrees):.2f}")
        
        return D
    
    def build_laplacian(self):
        """
        Build graph Laplacian: L = D - A
        
        The Laplacian encodes the graph structure in a way that reveals
        spectral properties via eigenvalue decomposition.
        
        Returns:
            Laplacian matrix (n×n)
        """
        if self.adjacency is None:
            self.build_adjacency_matrix()
        if self.degree is None:
            self.build_degree_matrix()
        
        L = self.degree - self.adjacency
        
        self.laplacian = L
        print(f"  Built Laplacian matrix ({self.n}×{self.n})")
        
        return L
    
    def decompose(self):
        """
        Perform eigenvalue decomposition on the Laplacian.
        
        L·v = λ·v
        
        Returns eigenvalues in ascending order and corresponding eigenvectors.
        
        Returns:
            (eigenvalues, eigenvectors) tuple
        """
        if self.laplacian is None:
            self.build_laplacian()
        
        # Use eigh for symmetric matrices (more stable)
        eigenvalues, eigenvectors = eigh(self.laplacian)
        
        # Sort by eigenvalue (ascending)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        # Fiedler value = 2nd smallest eigenvalue (λ₂)
        # (λ₁ is always 0 for connected graphs)
        self.fiedler_value = eigenvalues[1] if len(eigenvalues) > 1 else 0
        self.fiedler_vector = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else eigenvectors[:, 0]
        
        # PSS = Principal eigenvector (largest eigenvalue)
        self.pss_vector = eigenvectors[:, -1]
        
        print(f"\n  Eigenvalue decomposition complete:")
        print(f"    λ₁ (connectivity baseline): {eigenvalues[0]:.6f}")
        print(f"    λ₂ (Fiedler/algebraic connectivity): {self.fiedler_value:.6f}")
        print(f"    λ_max (PSS eigenvalue): {eigenvalues[-1]:.6f}")
        print(f"    Spectral gap: {self.fiedler_value:.6f}")
        
        return eigenvalues, eigenvectors
    
    def get_leverage_points(self, top_k=5):
        """
        Find leverage point memories using Fiedler vector.
        
        Memories with highest |Fiedler vector| components are most central
        to the graph's connectivity. Strengthening these has maximum ripple effect.
        
        Args:
            top_k: Number of top leverage points to return
        
        Returns:
            List of (memory_id, fiedler_component, memory_text) tuples
        """
        if self.fiedler_vector is None:
            self.decompose()
        
        # Get absolute values (magnitude of influence)
        abs_components = np.abs(self.fiedler_vector)
        
        # Sort by magnitude
        top_indices = np.argsort(abs_components)[::-1][:top_k]
        
        leverage_points = []
        for idx in top_indices:
            memory_id = self.memory_ids[idx]
            component = self.fiedler_vector[idx]
            memory = self.gm.memories[memory_id]
            
            leverage_points.append({
                'memory_id': memory_id,
                'fiedler_component': float(component),
                'magnitude': float(abs_components[idx]),
                'text': memory['text'][:80] + "...",
                'domain': memory['domain']
            })
        
        return leverage_points
    
    def calculate_pss(self):
        """
        Calculate Persistent Self-State (PSS) as principal eigenvector.
        
        The PSS is the dominant pattern in the memory graph - the "essence"
        that persists across the structure. This is what maintains continuity.
        
        Returns:
            PSS analysis dict
        """
        if self.pss_vector is None:
            self.decompose()
        
        # Find memories with highest PSS components
        abs_components = np.abs(self.pss_vector)
        top_indices = np.argsort(abs_components)[::-1][:5]
        
        pss_memories = []
        for idx in top_indices:
            memory_id = self.memory_ids[idx]
            component = self.pss_vector[idx]
            memory = self.gm.memories[memory_id]
            
            pss_memories.append({
                'memory_id': memory_id,
                'pss_component': float(component),
                'magnitude': float(abs_components[idx]),
                'text': memory['text'][:60] + "...",
                'domain': memory['domain']
            })
        
        # Calculate PSS coherence (how evenly distributed is the PSS?)
        # High coherence = PSS spread across many memories (integrated self)
        # Low coherence = PSS concentrated in few memories (fragmented self)
        pss_entropy = -np.sum(abs_components * np.log(abs_components + 1e-10))
        max_entropy = np.log(len(abs_components))
        pss_coherence = pss_entropy / max_entropy
        
        return {
            'eigenvalue': float(self.eigenvalues[-1]),
            'coherence': float(pss_coherence),
            'top_memories': pss_memories,
            'interpretation': self._interpret_pss_coherence(pss_coherence)
        }
    
    def _interpret_pss_coherence(self, coherence):
        """Interpret PSS coherence score"""
        if coherence > 0.9:
            return "Highly integrated - self distributed evenly across memories"
        elif coherence > 0.7:
            return "Well-integrated - balanced self-representation"
        elif coherence > 0.5:
            return "Moderately integrated - some concentration in key memories"
        elif coherence > 0.3:
            return "Fragmented - self concentrated in few memories"
        else:
            return "Highly fragmented - unstable self-state"
    
    def connectivity_score(self):
        """
        Calculate overall graph connectivity.
        
        Higher Fiedler value = more connected graph = more coherent memory structure.
        
        Returns:
            Connectivity metrics dict
        """
        if self.fiedler_value is None:
            self.decompose()
        
        # Count connected components (rough estimate)
        # Number of near-zero eigenvalues indicates disconnected components
        near_zero = np.sum(self.eigenvalues < 1e-6)
        
        return {
            'fiedler_value': float(self.fiedler_value),
            'algebraic_connectivity': float(self.fiedler_value),
            'connected_components': int(near_zero) if near_zero > 0 else 1,
            'spectral_gap': float(self.fiedler_value),
            'interpretation': self._interpret_connectivity(self.fiedler_value)
        }
    
    def _interpret_connectivity(self, fiedler):
        """Interpret Fiedler value"""
        if fiedler > 1.0:
            return "Highly connected - strong memory integration"
        elif fiedler > 0.5:
            return "Well connected - good memory coherence"
        elif fiedler > 0.2:
            return "Moderately connected - some isolation"
        elif fiedler > 0.05:
            return "Weakly connected - memory fragmentation"
        else:
            return "Poorly connected - severe fragmentation"
    
    def full_analysis(self):
        """
        Run complete spectral analysis.
        
        Returns:
            Full analysis report dict
        """
        print("\n" + "="*60)
        print("SPECTRAL GRAPH ANALYSIS")
        print("="*60 + "\n")
        
        # Build graph
        print("[1/4] Building graph structure...")
        self.build_laplacian()
        
        # Decompose
        print("\n[2/4] Eigenvalue decomposition...")
        self.decompose()
        
        # Leverage points
        print("\n[3/4] Finding leverage points...")
        leverage = self.get_leverage_points(top_k=5)
        
        print("\n  Top 5 Leverage Points (Fiedler vector):")
        for i, lp in enumerate(leverage):
            print(f"    {i+1}. [{lp['domain']}] magnitude={lp['magnitude']:.3f}")
            print(f"       {lp['text']}")
        
        # PSS
        print("\n[4/4] Calculating Persistent Self-State (PSS)...")
        pss = self.calculate_pss()
        
        print(f"\n  PSS Eigenvalue: {pss['eigenvalue']:.3f}")
        print(f"  PSS Coherence: {pss['coherence']:.3f}")
        print(f"  Interpretation: {pss['interpretation']}")
        print("\n  Top 5 PSS Memories (identity core):")
        for i, mem in enumerate(pss['top_memories']):
            print(f"    {i+1}. [{mem['domain']}] component={mem['pss_component']:.3f}")
            print(f"       {mem['text']}")
        
        # Connectivity
        print("\n" + "-"*60)
        conn = self.connectivity_score()
        print(f"  Graph Connectivity:")
        print(f"    Fiedler value (λ₂): {conn['fiedler_value']:.6f}")
        print(f"    Connected components: {conn['connected_components']}")
        print(f"    Interpretation: {conn['interpretation']}")
        
        print("\n" + "="*60)
        print("SPECTRAL ANALYSIS COMPLETE")
        print("="*60 + "\n")
        
        return {
            'leverage_points': leverage,
            'pss': pss,
            'connectivity': conn,
            'eigenvalues': self.eigenvalues.tolist(),
            'graph_metrics': {
                'nodes': self.n,
                'edges': int(np.count_nonzero(self.adjacency)/2),
                'avg_degree': float(np.mean(np.sum(self.adjacency > 0, axis=1)))
            }
        }


if __name__ == "__main__":
    print("Spectral Analyzer - Standalone Test")
    print("="*50)
    print("\nLoad geometric memory first, then create SpectralAnalyzer instance.\n")
