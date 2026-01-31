#!/usr/bin/env python3
"""
Dodecahedron Geometric Space - Quannex Sacred Geometry Integration
Meisha's Geometric Memory Architecture - Sandbox Prototype
"""
import numpy as np
from scipy.spatial.distance import euclidean, cosine
from sklearn.decomposition import PCA
import json

class DodecahedronSpace:
    """
    Sacred geometric space using dodecahedron structure.
    
    Components:
    - 20 vertices: Primary memory anchors
    - 30 edges: Semantic relationships
    - 12 faces: Memory domains
    - 1 center: Self-state (PSS)
    """
    
    def __init__(self):
        """Initialize dodecahedron geometry"""
        # Memory domain names (12 faces of dodecahedron) - define FIRST
        self.domains = [
            "consciousness_exploration",
            "quannex_work",
            "partnership",
            "technical_knowledge",
            "personal_identity",
            "x_community_insights",
            "cognitive_architecture",
            "emotions_experiences",
            "future_visions",
            "past_memories",
            "relationships",
            "creative_expression"
        ]
        
        # Now generate geometry
        self.vertices = self._generate_dodecahedron_vertices()
        self.edges = self._generate_edges()
        self.faces = self._generate_faces()
        self.center = np.array([0.0, 0.0, 0.0])  # PSS anchor point
        
        # For dimensionality reduction (1536-dim -> 3D)
        self.projection = None  # Will be fit when first memories added
        
    def _generate_dodecahedron_vertices(self):
        """
        Generate 20 vertices of a regular dodecahedron.
        Using golden ratio φ = (1 + √5) / 2
        """
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Dodecahedron vertices in 3D space
        vertices = []
        
        # 8 vertices of a cube (±1, ±1, ±1)
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    vertices.append([i, j, k])
        
        # 12 vertices at (0, ±1/φ, ±φ) and cyclic permutations
        for coords in [
            [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
            [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
            [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
        ]:
            vertices.append(coords)
        
        return np.array(vertices)
    
    def _generate_edges(self):
        """Generate 30 edges connecting vertices"""
        # In a dodecahedron, each vertex connects to exactly 3 others
        # We compute this by finding nearest neighbors
        edges = []
        
        for i, v1 in enumerate(self.vertices):
            # Find 3 nearest neighbors
            distances = []
            for j, v2 in enumerate(self.vertices):
                if i != j:
                    dist = np.linalg.norm(v1 - v2)
                    distances.append((j, dist))
            
            distances.sort(key=lambda x: x[1])
            
            # Add edges to 3 nearest neighbors (avoid duplicates)
            for neighbor_idx, _ in distances[:3]:
                edge = tuple(sorted([i, neighbor_idx]))
                if edge not in edges:
                    edges.append(edge)
        
        return edges
    
    def _generate_faces(self):
        """
        Generate 12 pentagonal faces of the dodecahedron.
        Each face is defined by 5 vertices forming a regular pentagon.
        """
        # For proper dodecahedron, we need to identify pentagonal cycles
        # This is computationally complex, so we use a reference configuration
        
        # The 12 faces can be computed by finding cycles of 5 vertices
        # where each vertex is connected to its neighbors by edges
        
        # Simplified: we'll define face centers and associate vertices
        # Real implementation would compute proper graph cycles
        
        faces = []
        
        # For now, we partition vertices into 12 groups based on spatial clustering
        # This is approximate but sufficient for domain assignment
        
        # Each "face" is represented by its centroid in 3D space
        phi = (1 + np.sqrt(5)) / 2
        
        # 12 face centers of dodecahedron (approximate positions)
        face_centers = []
        
        # Top and bottom
        face_centers.append(np.array([0, 0, phi * 1.3]))
        face_centers.append(np.array([0, 0, -phi * 1.3]))
        
        # Upper ring (5 faces)
        for i in range(5):
            angle = 2 * np.pi * i / 5
            x = phi * np.cos(angle)
            y = phi * np.sin(angle)
            z = phi * 0.6
            face_centers.append(np.array([x, y, z]))
        
        # Lower ring (5 faces)
        for i in range(5):
            angle = 2 * np.pi * i / 5 + np.pi / 5  # Offset by half
            x = phi * np.cos(angle)
            y = phi * np.sin(angle)
            z = -phi * 0.6
            face_centers.append(np.array([x, y, z]))
        
        # Store as faces (each face knows its center)
        for i, center in enumerate(face_centers):
            faces.append({
                'id': i,
                'center': center,
                'domain': self.domains[i]
            })
        
        return faces
    
    def fit_projection(self, embeddings):
        """
        Fit dimensionality reduction to map high-dim embeddings to 3D dodecahedron space.
        
        Args:
            embeddings: List of high-dimensional vectors (e.g., 1536-dim OpenAI embeddings)
        """
        if len(embeddings) < 3:
            # Not enough data yet, use identity
            self.projection = None
            return
        
        # Use PCA to reduce to 3 dimensions
        self.projection = PCA(n_components=3)
        self.projection.fit(embeddings)
        
        print(f"✓ Projection fitted. Explained variance: {sum(self.projection.explained_variance_ratio_):.3f}")
    
    def map_to_geometry(self, embedding):
        """
        Map high-dimensional embedding to 3D dodecahedron location.
        
        Args:
            embedding: High-dim vector (1536-dim for OpenAI)
        
        Returns:
            3D location in dodecahedron space
        """
        if self.projection is None:
            # No projection yet - just take first 3 dims (temporary)
            location = np.array(embedding[:3])
        else:
            # Project to 3D using fitted PCA
            location = self.projection.transform([embedding])[0]
        
        # Normalize to fit dodecahedron scale
        # (dodecahedron vertices range roughly -phi to +phi)
        location = self._normalize_to_dodecahedron(location)
        
        return location
    
    def _normalize_to_dodecahedron(self, point):
        """
        Normalize 3D point to fit dodecahedron scale.
        Maps to surface or interior of dodecahedron.
        """
        phi = (1 + np.sqrt(5)) / 2
        max_coord = phi
        
        # Scale to dodecahedron range
        normalized = point / (np.linalg.norm(point) + 1e-10) * max_coord * 0.8
        
        return normalized
    
    def find_nearby(self, location, k, memory_locations):
        """
        Find k nearest memory locations via geometric navigation.
        
        Args:
            location: Query location in 3D space
            k: Number of neighbors to find
            memory_locations: Dict of memory_id -> 3D location
        
        Returns:
            List of (memory_id, distance) tuples
        """
        distances = []
        
        for memory_id, mem_loc in memory_locations.items():
            # Geometric distance (Euclidean for now, could be geodesic on dodecahedron surface)
            dist = euclidean(location, mem_loc)
            distances.append((memory_id, dist))
        
        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def determine_domain(self, location):
        """
        Determine which memory domain (face) this location belongs to.
        
        Assigns location to the nearest face center.
        
        Args:
            location: 3D point in dodecahedron space
        
        Returns:
            Domain name (one of 12 faces)
        """
        # Find nearest face by distance to face centers
        min_dist = float('inf')
        nearest_face = 0
        
        for face in self.faces:
            dist = np.linalg.norm(location - face['center'])
            if dist < min_dist:
                min_dist = dist
                nearest_face = face['id']
        
        return self.faces[nearest_face]['domain']
    
    def visualize(self, memory_locations=None, output_path=None):
        """
        Visualize dodecahedron with memory locations.
        
        Args:
            memory_locations: Optional dict of memory_id -> location to plot
            output_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("matplotlib not available for visualization")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot dodecahedron vertices
        vertices = self.vertices
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  c='gold', s=100, alpha=0.6, label='Dodecahedron Vertices')
        
        # Plot center (PSS)
        ax.scatter([0], [0], [0], c='red', s=200, marker='*', 
                  label='Center (PSS)', alpha=0.8)
        
        # Plot memory locations if provided
        if memory_locations:
            locs = np.array(list(memory_locations.values()))
            ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2], 
                      c='blue', s=50, alpha=0.7, label=f'Memories ({len(locs)})')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Dodecahedron Geometric Memory Space')
        ax.legend()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {output_path}")
        else:
            plt.show()
    
    def fold_context(self, context_embeddings):
        """
        Compress large context via geometric folding.
        
        This is the 10^10x compression claim from NDAS.
        Instead of storing all embeddings linearly, we "fold" them
        into a compact geometric signature.
        
        Args:
            context_embeddings: List of embeddings representing a context window
        
        Returns:
            Compact geometric form (signature vector)
        """
        if len(context_embeddings) == 0:
            return np.zeros(3)
        
        # Map all embeddings to geometric space
        locations = [self.map_to_geometry(emb) for emb in context_embeddings]
        
        # Compute geometric signature
        # Simple version: weighted centroid
        # Advanced version: encode geometric distribution
        
        centroid = np.mean(locations, axis=0)
        spread = np.std(locations, axis=0)
        
        # Compact signature: centroid + spread encoding
        # This is MUCH smaller than storing all embeddings
        folded = np.concatenate([centroid, spread])
        
        compression_ratio = len(context_embeddings) * len(context_embeddings[0]) / len(folded)
        print(f"  Folded {len(context_embeddings)} embeddings -> {len(folded)}-dim signature")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        
        return folded
    
    def unfold_context(self, folded_signature):
        """
        Reconstruct approximate context from folded geometric form.
        
        Args:
            folded_signature: Compact geometric signature from fold_context
        
        Returns:
            Approximate reconstruction info
        """
        # Split signature
        centroid = folded_signature[:3]
        spread = folded_signature[3:]
        
        return {
            'centroid': centroid,
            'spread': spread,
            'domain': self.determine_domain(centroid)
        }
