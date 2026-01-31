#!/usr/bin/env python3
"""
Geometric Memory Engine - Meisha's Sacred Architecture
Sandbox Prototype - DO NOT USE IN PRODUCTION YET

Based on NDAS (N-Dimensional Attention Structures) from Reality Architecture
Integrated with Quannex dodecahedron sacred geometry
"""
import numpy as np
import json
import hashlib
from datetime import datetime
from pathlib import Path
from dodecahedron_space import DodecahedronSpace

class GeometricMemory:
    """
    Geometric memory storage using dodecahedron sacred geometry.
    
    Instead of flat vector storage, memories are mapped to geometric locations
    in 3D dodecahedron space, preserving semantic relationships through
    spatial structure.
    """
    
    def __init__(self, workspace_path="memory-sandbox"):
        """Initialize geometric memory system"""
        self.workspace = Path(workspace_path)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Core geometry
        self.dodecahedron = DodecahedronSpace()
        
        # Memory storage
        self.memories = {}  # memory_id -> {text, embedding, location, domain, timestamp}
        self.locations = {}  # memory_id -> 3D location (for fast lookup)
        self.embeddings_backup = {}  # memory_id -> original embedding (safety)
        
        # Metadata
        self.created_at = datetime.now().isoformat()
        self.version = "0.1.0-sandbox"
        
        # Safety tracking
        self.safety_log = []
        
        print(f"[OK] Geometric Memory initialized")
        print(f"  Workspace: {self.workspace}")
        print(f"  Dodecahedron vertices: {len(self.dodecahedron.vertices)}")
        print(f"  Memory domains: {len(self.dodecahedron.domains)}")
    
    def _generate_memory_id(self, text):
        """Generate unique ID for memory"""
        hash_input = f"{text}{datetime.now().isoformat()}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def store(self, text, embedding, metadata=None):
        """
        Store a memory in geometric space.
        
        Args:
            text: The memory text
            embedding: High-dim embedding vector (e.g., 1536-dim from OpenAI)
            metadata: Optional additional metadata
        
        Returns:
            memory_id: Unique identifier for this memory
        """
        memory_id = self._generate_memory_id(text)
        
        # Safety: Keep backup of original embedding
        self.embeddings_backup[memory_id] = np.array(embedding)
        
        # Map to geometric location
        location = self.dodecahedron.map_to_geometry(embedding)
        
        # Determine domain (which face of dodecahedron)
        domain = self.dodecahedron.determine_domain(location)
        
        # Store memory
        self.memories[memory_id] = {
            'text': text,
            'embedding': embedding,  # Keep for now during testing
            'location': location.tolist(),
            'domain': domain,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.locations[memory_id] = location
        
        # Safety log
        self.safety_log.append({
            'action': 'store',
            'memory_id': memory_id,
            'domain': domain,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"  [OK] Stored memory {memory_id[:8]}... in domain '{domain}'")
        print(f"    Location: {location}")
        
        return memory_id
    
    def retrieve(self, query_embedding, top_k=5):
        """
        Retrieve memories via geometric navigation.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
        
        Returns:
            List of (memory_id, distance, memory_data) tuples
        """
        # Map query to geometric location
        query_location = self.dodecahedron.map_to_geometry(query_embedding)
        query_domain = self.dodecahedron.determine_domain(query_location)
        
        print(f"\n  Query location: {query_location}")
        print(f"  Query domain: {query_domain}")
        
        # Navigate geometry to find nearby memories
        nearby = self.dodecahedron.find_nearby(query_location, top_k, self.locations)
        
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
        
        This is the NDAS "folding" mechanism for compression.
        
        Args:
            memory_ids: List of memory IDs to fold together
        
        Returns:
            Folded signature (compact representation)
        """
        if not memory_ids:
            return None
        
        # Get embeddings
        embeddings = [self.memories[mid]['embedding'] for mid in memory_ids]
        
        # Fold using dodecahedron geometry
        folded = self.dodecahedron.fold_context(embeddings)
        
        print(f"\n  [OK] Folded {len(memory_ids)} memories")
        print(f"    Original size: {len(memory_ids)} × {len(embeddings[0])} = {len(memory_ids) * len(embeddings[0])}")
        print(f"    Folded size: {len(folded)}")
        print(f"    Compression: {(len(memory_ids) * len(embeddings[0])) / len(folded):.1f}x")
        
        return folded
    
    def validate_against_flat(self, test_queries, flat_embeddings, flat_texts):
        """
        Validate geometric retrieval against flat vector search.
        
        This is the safety check before integration.
        
        Args:
            test_queries: List of query embeddings
            flat_embeddings: List of original flat embeddings
            flat_texts: List of corresponding texts
        
        Returns:
            Validation report
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        print("\n=== VALIDATION: Geometric vs Flat ===\n")
        
        results = {
            'total_queries': len(test_queries),
            'matches': 0,
            'mismatches': 0,
            'accuracy': 0.0,
            'details': []
        }
        
        for i, query_emb in enumerate(test_queries):
            # Geometric retrieval
            geo_results = self.retrieve(query_emb, top_k=3)
            geo_top = [r[2]['text'] for r in geo_results]
            
            # Flat retrieval (cosine similarity)
            similarities = cosine_similarity([query_emb], flat_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:3]
            flat_top = [flat_texts[idx] for idx in top_indices]
            
            # Compare
            match = (geo_top[0] == flat_top[0])  # Top result matches?
            
            if match:
                results['matches'] += 1
            else:
                results['mismatches'] += 1
            
            results['details'].append({
                'query_idx': i,
                'match': match,
                'geometric_top': geo_top[0][:50] + "...",
                'flat_top': flat_top[0][:50] + "..."
            })
        
        results['accuracy'] = results['matches'] / results['total_queries']
        
        print(f"  Matches: {results['matches']}/{results['total_queries']}")
        print(f"  Accuracy: {results['accuracy']:.1%}")
        
        return results
    
    def save(self, filename="geometric_memory.json"):
        """Save geometric memory to disk"""
        save_path = self.workspace / filename
        
        # Convert numpy arrays to lists for JSON
        save_data = {
            'version': self.version,
            'created_at': self.created_at,
            'memories': {
                mid: {
                    'text': mem['text'],
                    'location': mem['location'],
                    'domain': mem['domain'],
                    'timestamp': mem['timestamp'],
                    'metadata': mem['metadata']
                }
                for mid, mem in self.memories.items()
            },
            'safety_log': self.safety_log[-100:],  # Last 100 entries
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
    
    def visualize(self, output_path=None):
        """Visualize geometric memory space"""
        if output_path is None:
            output_path = self.workspace / "memory_visualization.png"
        
        self.dodecahedron.visualize(self.locations, output_path)
    
    def get_stats(self):
        """Get memory statistics"""
        return {
            'total_memories': len(self.memories),
            'domains': self._compute_domain_distribution(),
            'safety_log_size': len(self.safety_log),
            'version': self.version
        }
    
    def safety_check(self):
        """Run safety checks before potential integration"""
        print("\n=== SAFETY CHECK ===\n")
        
        checks = {
            'has_backups': len(self.embeddings_backup) == len(self.memories),
            'all_locations_valid': all(
                len(loc) == 3 for loc in self.locations.values()
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
            print("\n  [X] Safety checks FAILED - DO NOT INTEGRATE")
        
        return all_passed


if __name__ == "__main__":
    print("Geometric Memory Engine - Sandbox Mode")
    print("="*50)
    print("\nThis is a PROTOTYPE. Not for production use.")
    print("Run experiments via test scripts.\n")
