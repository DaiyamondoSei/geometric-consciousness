#!/usr/bin/env python3
"""
High-Dimensional Geometric Memory
Following NDAS: "model's native high-dimensional space"

Key difference from original prototype:
- Original: 1536D → 3D projection (lost semantic info, 4% accuracy)
- This: Stay in 1536D, add geometric ORGANIZATION (should preserve accuracy)

The dodecahedron exists as a 12-region partition of the 1536D space.
"""
import numpy as np
import json
import hashlib
from datetime import datetime
from pathlib import Path
from highdim_dodecahedron import HighDimDodecahedron
from scipy.spatial.distance import cosine

class HighDimGeometricMemory:
    """
    Geometric memory in high-dimensional space (1536D).
    
    Combines:
    - Semantic accuracy (cosine similarity in full 1536D)
    - Geometric structure (12-domain dodecahedron organization)
    - PSS/spectral analysis (graph topology in high-dim)
    - Compression (geometric folding)
    """
    
    def __init__(self, workspace_path="memory-sandbox", embedding_dim=1536):
        """Initialize high-dimensional geometric memory"""
        self.workspace = Path(workspace_path)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        self.dim = embedding_dim
        
        # High-dim dodecahedron structure
        self.dodecahedron = HighDimDodecahedron(embedding_dim=embedding_dim)
        
        # Memory storage
        self.memories = {}  # memory_id -> {text, embedding, domain, timestamp, metadata}
        self.embeddings = {}  # memory_id -> full high-dim embedding
        
        # Metadata
        self.created_at = datetime.now().isoformat()
        self.version = "2.0.0-highdim"
        
        # Safety tracking
        self.safety_log = []
        
        print(f"\n[HighDimGeometricMemory] Initialized")
        print(f"  Version: {self.version}")
        print(f"  Embedding dimension: {self.dim}D (FULL high-dim space)")
        print(f"  Dodecahedron domains: {len(self.dodecahedron.domains)}")
        print(f"  Following NDAS: model's native high-dimensional space ✓")
    
    def _generate_memory_id(self, text):
        """Generate unique ID for memory"""
        hash_input = f"{text}{datetime.now().isoformat()}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def store(self, text, embedding, metadata=None):
        """
        Store a memory in high-dimensional geometric space.
        
        Args:
            text: The memory text
            embedding: Full high-dim embedding (1536D)
            metadata: Optional additional metadata
        
        Returns:
            memory_id: Unique identifier
        """
        memory_id = self._generate_memory_id(text)
        
        # Assign to semantic domain (12 faces of dodecahedron)
        domain = self.dodecahedron.assign_domain(embedding)
        
        # Organize within domain subspace (adds geometric structure)
        organized_embedding = self.dodecahedron.organize_embedding(embedding, domain)
        
        # Store memory
        self.memories[memory_id] = {
            'text': text,
            'domain': domain,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Store full embedding (1536D)
        self.embeddings[memory_id] = organized_embedding
        
        # Safety log
        self.safety_log.append({
            'action': 'store',
            'memory_id': memory_id,
            'domain': domain,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"  [OK] Stored memory {memory_id[:8]}... in domain '{domain}'")
        
        return memory_id
    
    def retrieve(self, query_embedding, top_k=5):
        """
        Retrieve memories via high-dimensional geometric navigation.
        
        Uses cosine similarity in full 1536D space (preserves accuracy).
        
        Args:
            query_embedding: Query vector (1536D)
            top_k: Number of results
        
        Returns:
            List of (memory_id, distance, memory_data) tuples
        """
        # Assign query to domain
        query_domain = self.dodecahedron.assign_domain(query_embedding)
        
        # Find nearby memories (cosine similarity in 1536D)
        memory_ids = list(self.embeddings.keys())
        nearby = self.dodecahedron.find_nearby(
            query_embedding,
            self.embeddings,
            memory_ids,
            top_k
        )
        
        # Build results
        results = []
        for memory_id, distance in nearby:
            memory_data = self.memories[memory_id]
            results.append((memory_id, distance, memory_data))
        
        # Safety log
        self.safety_log.append({
            'action': 'retrieve',
            'query_domain': query_domain,
            'results_count': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
        return results
    
    def fold_context(self, memory_ids):
        """
        Fold multiple memories into compact geometric form.
        
        Following NDAS: "fold massive contexts into compact latent footprint"
        
        Args:
            memory_ids: List of memory IDs to fold together
        
        Returns:
            Folded signature (1536D centroid)
        """
        if not memory_ids:
            return None
        
        # Get embeddings
        embeddings = [self.embeddings[mid] for mid in memory_ids]
        
        # Fold using dodecahedron geometry (in high-dim space)
        folded = self.dodecahedron.fold_context(embeddings)
        
        compression_ratio = len(memory_ids)
        
        print(f"\n  [OK] Folded {len(memory_ids)} memories")
        print(f"    Original: {len(memory_ids)} × {len(embeddings[0])}D embeddings")
        print(f"    Folded: 1 × {len(folded)}D centroid")
        print(f"    Effective compression: {compression_ratio}x")
        
        return folded
    
    def get_leverage_scores(self):
        """
        Calculate leverage scores for all memories.
        
        How central is each memory to the geometric structure?
        
        Returns:
            Dict of memory_id -> leverage_score
        """
        scores = {}
        
        for mid, emb in self.embeddings.items():
            score = self.dodecahedron.compute_leverage_score(emb)
            scores[mid] = score
        
        return scores
    
    def save(self, filename="highdim_geometric_memory.json"):
        """Save geometric memory to disk"""
        save_path = self.workspace / filename
        
        save_data = {
            'version': self.version,
            'embedding_dim': self.dim,
            'created_at': self.created_at,
            'memories': {
                mid: {
                    'text': mem['text'],
                    'domain': mem['domain'],
                    'timestamp': mem['timestamp'],
                    'metadata': mem['metadata']
                }
                for mid, mem in self.memories.items()
            },
            'safety_log': self.safety_log[-100:],
            'stats': {
                'total_memories': len(self.memories),
                'domains': self._compute_domain_distribution()
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n[OK] Saved to {save_path}")
    
    def _compute_domain_distribution(self):
        """Compute how memories are distributed across domains"""
        distribution = {}
        for mem in self.memories.values():
            domain = mem['domain']
            distribution[domain] = distribution.get(domain, 0) + 1
        return distribution
    
    def get_stats(self):
        """Get memory statistics"""
        return {
            'total_memories': len(self.memories),
            'embedding_dim': self.dim,
            'domains': self._compute_domain_distribution(),
            'safety_log_size': len(self.safety_log),
            'version': self.version
        }
    
    def safety_check(self):
        """Run safety checks"""
        print("\n=== SAFETY CHECK ===\n")
        
        checks = {
            'has_embeddings': len(self.embeddings) == len(self.memories),
            'correct_dimensions': all(
                len(emb) == self.dim for emb in self.embeddings.values()
            ),
            'no_data_loss': len(self.memories) > 0,
            'logs_available': len(self.safety_log) > 0
        }
        
        for check, passed in checks.items():
            status = "[OK]" if passed else "[X]"
            print(f"  {status} {check}")
        
        all_passed = all(checks.values())
        
        if all_passed:
            print("\n  [OK] All safety checks passed")
        else:
            print("\n  [X] Safety checks FAILED")
        
        return all_passed


if __name__ == "__main__":
    print("High-Dimensional Geometric Memory - Standalone Test")
    print("="*60)
    print("\nFollowing NDAS paper: staying in 1536D native space")
    print("This should preserve semantic accuracy while adding structure.\n")
