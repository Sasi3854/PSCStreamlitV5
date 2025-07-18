import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Dict
import itertools
from collections import Counter


class NetworkAnalyzer:
    """Handles network analysis and visualization for N-grams."""
    
    def __init__(self):
        """Initialize the network analyzer."""
        self.graph = None
        
    def create_ngram_network(self, ngrams: List[str], min_frequency: int = 2) -> nx.Graph:
        """Create a network graph from N-grams."""
        # Count N-gram frequencies
        ngram_counts = Counter(ngrams)
        
        # Filter by minimum frequency
        filtered_ngrams = [ngram for ngram, count in ngram_counts.items() if count >= min_frequency]
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes and edges
        for ngram in filtered_ngrams:
            words = ngram.split()
            
            # Add nodes
            for word in words:
                if not G.has_node(word):
                    G.add_node(word, frequency=0)
                G.nodes[word]['frequency'] += ngram_counts[ngram]
            
            # Add edges between consecutive words
            for i in range(len(words) - 1):
                word1, word2 = words[i], words[i + 1]
                if G.has_edge(word1, word2):
                    G[word1][word2]['weight'] += ngram_counts[ngram]
                else:
                    G.add_edge(word1, word2, weight=ngram_counts[ngram])
        
        self.graph = G
        return G
    
    def get_network_stats(self) -> Dict:
        """Get basic network statistics."""
        if self.graph is None:
            return {}
        
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_clustering': nx.average_clustering(self.graph),
            'connected_components': nx.number_connected_components(self.graph)
        }
    
    def get_central_nodes(self, centrality_type: str = 'degree', top_n: int = 10) -> List[Tuple[str, float]]:
        """Get most central nodes based on centrality measure."""
        if self.graph is None:
            return []
        
        if centrality_type == 'degree':
            centrality = nx.degree_centrality(self.graph)
        elif centrality_type == 'betweenness':
            centrality = nx.betweenness_centrality(self.graph)
        elif centrality_type == 'closeness':
            centrality = nx.closeness_centrality(self.graph)
        elif centrality_type == 'eigenvector':
            try:
                centrality = nx.eigenvector_centrality(self.graph)
            except:
                centrality = nx.degree_centrality(self.graph)  # Fallback
        else:
            centrality = nx.degree_centrality(self.graph)
        
        # Sort by centrality value
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]
    
    def create_plotly_network(self, layout_type: str = 'spring') -> go.Figure:
        """Create an interactive network visualization using Plotly."""
        if self.graph is None:
            return go.Figure()
        
        # Calculate layout
        if layout_type == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout_type == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout_type == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Extract node and edge information
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node size based on frequency
            frequency = self.graph.nodes[node].get('frequency', 1)
            node_size.append(max(10, min(50, frequency * 2)))  # Scale size
            
            # Node text
            adjacencies = list(self.graph.neighbors(node))
            node_text.append(f'{node}<br>Connections: {len(adjacencies)}<br>Frequency: {frequency}')
        
        # Create edges
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = self.graph[edge[0]][edge[1]].get('weight', 1)
            edge_weights.append(weight)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in self.graph.nodes()],
            textposition="middle center",
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                reversescale=True,
                color=node_size,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.02,
                    title="Node<br>Frequency"
                ),
                line=dict(width=2, color='black')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='N-gram Network Visualization',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Network shows relationships between words in N-grams",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor="left", yanchor="bottom",
                        font=dict(color="#888", size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        
        return fig
    
    def get_communities(self) -> Dict[int, List[str]]:
        """Detect communities in the network."""
        if self.graph is None:
            return {}
        
        try:
            # Use Louvain community detection
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph)
            
            # Group nodes by community
            communities = {}
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)
            
            return communities
        except ImportError:
            # Fallback to connected components if community package not available
            components = nx.connected_components(self.graph)
            return {i: list(component) for i, component in enumerate(components)}
