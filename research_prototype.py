"""
Temporal World Memory Research Prototype
========================================

Objective:
Evaluate a new AI memory architecture for long-horizon reasoning over evolving
relational data vs. standard RAG-style retrieval.

This is a single-file Python prototype representing a research implementation.
Dataset: relbench (rel-salt)

Architecture Stages:
1. Dataset Loader
2. Temporal World Graph
3. Hierarchical World Memory
4. Dynamic State Memory
5. Temporal Retrieval Engine
6. Baseline RAG
"""

# ==============================================================================
# IMPORTS AND CONFIGURATION
# ==============================================================================
import time
import datetime
import collections
import pathlib
import typing
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd
import networkx as nx

import relbench
from relbench.datasets import get_dataset

CONFIG = {
    "dataset_name": "rel-salt",
    "alpha_ema": 0.1,
    "eval_query_fraction": 0.05,
    "max_rows_per_table": 5000, # Cap for prototype performance
}

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class EntitySummary:
    entity_id: str
    entity_type: str
    attributes: Dict[str, Any]

@dataclass
class EventRecord:
    event_id: str
    event_type: str
    timestamp: pd.Timestamp
    attributes: Dict[str, Any]
    foreign_keys: Dict[str, str]

@dataclass
class RetrievalContext:
    entity_summary: Dict[str, Any]
    continuous_state: Dict[str, float]
    recent_events: List[Dict[str, Any]]
    neighbor_entities: List[Dict[str, Any]]
    historical_trends: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Entity Summary": self.entity_summary,
            "Continuous State": self.continuous_state,
            "Recent Events": self.recent_events,
            "Neighbor Entities": self.neighbor_entities,
            "Historical Trends": self.historical_trends
        }

# ==============================================================================
# STAGE 1: DATASET LOADER
# ==============================================================================
class DatasetLoader:
    """
    Responsibilities:
    - Load rel-salt dataset via relbench API.
    - Convert timestamps.
    - Automatically detect primary keys, foreign keys, timestamp columns, entity/event tables.
    - Report statistics.
    """
    def __init__(self, dataset_name: str = "rel-salt"):
        self.dataset_name = dataset_name
        self.dataset = None
        self.db = None
        
        self.entity_tables: Dict[str, Any] = {}
        self.event_tables: Dict[str, Any] = {}
        
    def load(self):
        print(f"[*] Loading dataset: {self.dataset_name}...")
        start_time = time.time()
        self.dataset = get_dataset(self.dataset_name)
        self.db = self.dataset.get_db()
        
        # Categorize tables
        for table_name, table in self.db.table_dict.items():
            if table.time_col is None:
                self.entity_tables[table_name] = table
            else:
                self.event_tables[table_name] = table
                
        print(f"[*] Dataset loaded in {time.time() - start_time:.2f} seconds.")
        self.print_schema_summary()
        
    def print_schema_summary(self):
        print("\n" + "="*50)
        print("SCHEMA SUMMARY")
        print("="*50)
        print(f"Total Tables: {len(self.db.table_dict)}")
        print(f"Entity Tables: {list(self.entity_tables.keys())}")
        print(f"Event Tables: {list(self.event_tables.keys())}")
        
        for table_name, table in self.db.table_dict.items():
            table_type = "Event" if table.time_col else "Entity"
            print(f"\n--- Table: {table_name} ({table_type}) ---")
            print(f"Rows: {len(table.df)}")
            print(f"Columns: {table.df.columns.tolist()}")
            print(f"Primary Key: {table.pkey_col}")
            print(f"Foreign Keys: {table.fkey_col_to_pkey_table}")
            print(f"Time Column: {table.time_col}")
        print("="*50 + "\n")

# ==============================================================================
# STAGE 2: TEMPORAL WORLD GRAPH
# ==============================================================================
class TemporalWorldGraph:
    """
    Builds a temporal knowledge graph using NetworkX.
    Nodes: Business entities (e.g., customers, materials).
    Edges: Relationships/events with timestamp and metadata.
    Preserves chronology.
    """
    def __init__(self, loader: DatasetLoader):
        self.loader = loader
        self.graph = nx.MultiDiGraph()
        
    def build(self):
        print("[*] Building Temporal World Graph...")
        start_time = time.time()
        
        # 1. Add Entity Nodes
        for table_name, table in self.loader.entity_tables.items():
            df = table.df.head(CONFIG["max_rows_per_table"])
            pkey = table.pkey_col
            fkeys = table.fkey_col_to_pkey_table
            
            for _, row in df.iterrows():
                node_id = f"{table_name}::{row[pkey]}"
                attrs = row.to_dict()
                self.graph.add_node(node_id, type=table_name, **attrs)
                
                # Link foreign keys within entities (structural edges)
                for fk_col, target_table in fkeys.items():
                    if pd.notna(row[fk_col]):
                        target_id = f"{target_table}::{row[fk_col]}"
                        # Ensure target node exists (might be implicit)
                        if not self.graph.has_node(target_id):
                            self.graph.add_node(target_id, type=target_table)
                        self.graph.add_edge(node_id, target_id, type='structural_fk', fk_col=fk_col, timestamp=pd.Timestamp.min)
        
        # 2. Add Event Edges
        # Events connect entities. If an event has multiple foreign keys, we link the event's primary entity to others,
        # or we create an event node and link entities to it. We will use the event node approach for richer metadata.
        for table_name, table in self.loader.event_tables.items():
            df = table.df.head(CONFIG["max_rows_per_table"])
            pkey = table.pkey_col
            fkeys = table.fkey_col_to_pkey_table
            time_col = table.time_col
            
            # Sort by time to ensure chronological processing if needed
            df = df.sort_values(by=time_col)
            
            for _, row in df.iterrows():
                event_node_id = f"Event_{table_name}::{row[pkey]}"
                timestamp = pd.to_datetime(row[time_col])
                attrs = row.to_dict()
                
                # Add event as a node to represent n-ary relationships
                self.graph.add_node(event_node_id, type=f"event_{table_name}", timestamp=timestamp, **attrs)
                
                # Connect foreign keys to the event node
                for fk_col, target_table in fkeys.items():
                    if pd.notna(row[fk_col]):
                        target_id = f"{target_table}::{row[fk_col]}"
                        if not self.graph.has_node(target_id):
                            self.graph.add_node(target_id, type=target_table)
                        # Bi-directional edge for traversal
                        self.graph.add_edge(event_node_id, target_id, type=fk_col, timestamp=timestamp)
                        self.graph.add_edge(target_id, event_node_id, type=f"involved_in_{fk_col}", timestamp=timestamp)
                        
        print(f"[*] Graph built in {time.time() - start_time:.2f} seconds.")
        print(f"    Nodes: {self.graph.number_of_nodes()}")
        print(f"    Edges: {self.graph.number_of_edges()}")

# ==============================================================================
# STAGE 3: HIERARCHICAL WORLD MEMORY
# ==============================================================================
class HierarchicalWorldMemory:
    """
    Compresses lower-level information into daily, weekly, monthly, and lifetime summaries.
    Level 0: Raw tables
    Level 1: Daily
    Level 2: Weekly
    Level 3: Monthly
    Level 4: Lifetime
    """
    def __init__(self, graph: TemporalWorldGraph):
        self.graph = graph
        self.memory = collections.defaultdict(lambda: collections.defaultdict(dict))
        
    def build_hierarchy(self):
        print("[*] Building Hierarchical World Memory...")
        start_time = time.time()
        
        # We will iterate over all events in the graph and aggregate them for target entities
        event_nodes = [n for n, attr in self.graph.graph.nodes(data=True) if attr.get('type', '').startswith('event_')]
        
        for event_id in event_nodes:
            event_data = self.graph.graph.nodes[event_id]
            timestamp = event_data['timestamp']
            
            # Find entities involved in this event
            # Edges from event to entity
            involved_entities = [v for u, v, k, d in self.graph.graph.out_edges(event_id, data=True, keys=True) 
                                 if d.get('timestamp') == timestamp and not str(v).startswith("Event_")]
            
            day_key = timestamp.strftime('%Y-%m-%d')
            week_key = timestamp.strftime('%Y-%W')
            month_key = timestamp.strftime('%Y-%m')
            
            for entity in involved_entities:
                # Level 1: Daily
                if day_key not in self.memory[entity]['daily']:
                    self.memory[entity]['daily'][day_key] = {'event_count': 0, 'event_types': set()}
                self.memory[entity]['daily'][day_key]['event_count'] += 1
                self.memory[entity]['daily'][day_key]['event_types'].add(event_data['type'])
                
                # Level 2: Weekly
                if week_key not in self.memory[entity]['weekly']:
                    self.memory[entity]['weekly'][week_key] = {'event_count': 0, 'event_types': set()}
                self.memory[entity]['weekly'][week_key]['event_count'] += 1
                self.memory[entity]['weekly'][week_key]['event_types'].add(event_data['type'])
                
                # Level 3: Monthly
                if month_key not in self.memory[entity]['monthly']:
                    self.memory[entity]['monthly'][month_key] = {'event_count': 0, 'event_types': set()}
                self.memory[entity]['monthly'][month_key]['event_count'] += 1
                self.memory[entity]['monthly'][month_key]['event_types'].add(event_data['type'])
                
                # Level 4: Lifetime
                if 'lifetime' not in self.memory[entity]:
                    self.memory[entity]['lifetime'] = {'total_events': 0, 'first_seen': timestamp, 'last_seen': timestamp}
                self.memory[entity]['lifetime']['total_events'] += 1
                if timestamp < self.memory[entity]['lifetime']['first_seen']:
                    self.memory[entity]['lifetime']['first_seen'] = timestamp
                if timestamp > self.memory[entity]['lifetime']['last_seen']:
                    self.memory[entity]['lifetime']['last_seen'] = timestamp

        print(f"[*] Hierarchical Memory built in {time.time() - start_time:.2f} seconds.")

    def get_summary(self, entity_id: str, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Retrieve hierarchical summary valid strictly before the given timestamp."""
        summary = {
            'daily_recent': [],
            'weekly_recent': [],
            'monthly_recent': [],
            'lifetime': {}
        }
        
        if entity_id not in self.memory:
            return summary
            
        entity_mem = self.memory[entity_id]
        
        # Filter past memories
        day_key = timestamp.strftime('%Y-%m-%d')
        week_key = timestamp.strftime('%Y-%W')
        month_key = timestamp.strftime('%Y-%m')
        
        summary['daily_recent'] = {k: v for k, v in entity_mem.get('daily', {}).items() if k < day_key}
        summary['weekly_recent'] = {k: v for k, v in entity_mem.get('weekly', {}).items() if k < week_key}
        summary['monthly_recent'] = {k: v for k, v in entity_mem.get('monthly', {}).items() if k < month_key}
        
        if 'lifetime' in entity_mem:
            lt = entity_mem['lifetime']
            if lt['first_seen'] < timestamp:
                summary['lifetime'] = {
                    'total_events_approx': len(summary['daily_recent']), # approximate to avoid future leak
                    'first_seen': lt['first_seen']
                }
        
        return summary

# ==============================================================================
# STAGE 4: DYNAMIC STATE MEMORY
# ==============================================================================
class DynamicStateMemory:
    """
    Maintains a continuous state vector for entities, updated exponentially over time.
    NO DEEP LEARNING. Just architectural demonstration.
    State features: activity_frequency, recency_score
    """
    def __init__(self, loader: DatasetLoader, graph: TemporalWorldGraph, alpha: float = 0.1):
        self.loader = loader
        self.graph = graph
        self.alpha = alpha
        
        # Store state history: entity_id -> list of (timestamp, state_dict)
        self.state_history = collections.defaultdict(list)
        
    def build_states(self):
        print("[*] Building Dynamic State Memory...")
        start_time = time.time()
        
        # Collect all events chronologically
        events = []
        for n, attr in self.graph.graph.nodes(data=True):
            if attr.get('type', '').startswith('event_'):
                events.append((attr['timestamp'], n, attr))
                
        events.sort(key=lambda x: x[0])
        
        # Current active state
        current_state = collections.defaultdict(lambda: {'activity_score': 0.0, 'last_activity': None})
        
        for timestamp, event_id, attr in events:
            # Find involved entities
            involved = [v for u, v, k, d in self.graph.graph.out_edges(event_id, data=True, keys=True) 
                        if not str(v).startswith("Event_")]
            
            for entity in involved:
                state = current_state[entity]
                
                # Update activity score (EMA)
                # If a long time passed, we could decay it. For simplicity, just EMA on event occurrence.
                state['activity_score'] = (1 - self.alpha) * state['activity_score'] + self.alpha * 1.0
                state['last_activity'] = timestamp
                
                # Save snapshot
                # We copy to avoid mutation reference issues
                self.state_history[entity].append((timestamp, dict(state)))
                
        print(f"[*] State Memory built in {time.time() - start_time:.2f} seconds.")
        
    def get_state(self, entity_id: str, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Returns the state vector strictly before timestamp."""
        if entity_id not in self.state_history:
            return {}
            
        history = self.state_history[entity_id]
        
        # Binary search for the latest state before timestamp
        # For simplicity in prototype, linear scan backwards
        for i in range(len(history)-1, -1, -1):
            if history[i][0] < timestamp:
                return history[i][1]
        return {}

# ==============================================================================
# STAGE 5: TEMPORAL RETRIEVAL ENGINE
# ==============================================================================
class TemporalRetrievalEngine:
    """
    Given an entity, timestamp, and query window, retrieve ONLY info before timestamp.
    Outputs LLM-ready context.
    """
    def __init__(self, graph: TemporalWorldGraph, hierarchy: HierarchicalWorldMemory, states: DynamicStateMemory):
        self.graph = graph
        self.hierarchy = hierarchy
        self.states = states
        
    def retrieve(self, entity_id: str, timestamp: pd.Timestamp, window_days: int = 30) -> RetrievalContext:
        
        # 1. Entity Summary (Raw attributes)
        if not self.graph.graph.has_node(entity_id):
            attrs = {}
        else:
            attrs = {k: v for k, v in self.graph.graph.nodes[entity_id].items() if k != 'type'}
            
        entity_summary = attrs
        
        # 2. Continuous State
        continuous_state = self.states.get_state(entity_id, timestamp)
        
        # 3. Temporal Graph Traversal (Recent Events & Neighbors)
        recent_events = []
        neighbor_entities = []
        
        cutoff_time = timestamp - pd.Timedelta(days=window_days)
        
        if self.graph.graph.has_node(entity_id):
            # Edges from entity to events
            for u, v, k, d in self.graph.graph.out_edges(entity_id, data=True, keys=True):
                edge_time = d.get('timestamp')
                # Filter strictly before timestamp, and within window
                if edge_time is not None and cutoff_time <= edge_time < timestamp:
                    if str(v).startswith("Event_"):
                        event_data = self.graph.graph.nodes[v].copy()
                        # Convert timestamp to string for serialization
                        if 'timestamp' in event_data:
                            event_data['timestamp'] = event_data['timestamp'].isoformat()
                        recent_events.append(event_data)
                        
                        # Find other entities involved in this event (neighbors)
                        for ev_u, ev_v, ev_k, ev_d in self.graph.graph.out_edges(v, data=True, keys=True):
                            if ev_v != entity_id and not str(ev_v).startswith("Event_"):
                                neighbor_entities.append({'entity_id': ev_v, 'relation': ev_k})
                                
        # 4. Hierarchical Context (Trends)
        historical_trends = self.hierarchy.get_summary(entity_id, timestamp)
        
        return RetrievalContext(
            entity_summary=entity_summary,
            continuous_state=continuous_state,
            recent_events=recent_events, # could be trimmed if too large
            neighbor_entities=neighbor_entities,
            historical_trends=historical_trends
        )

# ==============================================================================
# STAGE 6: BASELINE RAG
# ==============================================================================
class BaselineRAG:
    """
    Simulates standard document retrieval.
    Flattens all historical textual information before timestamp.
    """
    def __init__(self, loader: DatasetLoader):
        self.loader = loader
        self.documents = []
        
    def build(self):
        print("[*] Building Baseline RAG Documents...")
        start_time = time.time()
        
        # Flatten events into text documents
        for table_name, table in self.loader.event_tables.items():
            df = table.df.head(CONFIG["max_rows_per_table"])
            time_col = table.time_col
            
            for _, row in df.iterrows():
                timestamp = pd.to_datetime(row[time_col])
                
                # Create a pseudo-document
                text_repr = f"At {timestamp.isoformat()}, Event {table_name} occurred. Details: "
                text_repr += ", ".join([f"{k}={v}" for k, v in row.items() if k != time_col])
                
                self.documents.append({
                    'timestamp': timestamp,
                    'text': text_repr,
                    'raw_row': row.to_dict()
                })
                
        # Sort documents by time for faster filtering
        self.documents.sort(key=lambda x: x['timestamp'])
        print(f"[*] Baseline RAG built with {len(self.documents)} documents in {time.time() - start_time:.2f} seconds.")
        
    def retrieve(self, entity_id: str, timestamp: pd.Timestamp, window_days: int = 30) -> str:
        cutoff_time = timestamp - pd.Timedelta(days=window_days)
        
        retrieved_docs = []
        # Simulate retrieval filtering
        for doc in self.documents:
            if doc['timestamp'] >= timestamp:
                break # Since sorted, we can break early
            
            if doc['timestamp'] >= cutoff_time:
                # In RAG, we would do vector similarity. 
                # Here we just keyword match the entity_id to simulate perfect retriever.
                raw_id = entity_id.split("::")[-1] if "::" in entity_id else entity_id
                if str(raw_id) in doc['text']:
                    retrieved_docs.append(doc['text'])
                    
        return "\n".join(retrieved_docs)

# ==============================================================================
# STAGE 7: BENCHMARK AND EVALUATION
# ==============================================================================
class Benchmark:
    def __init__(self):
        self.metrics = {}
        
    def evaluate(self, 
                 graph: TemporalWorldGraph, 
                 engine: TemporalRetrievalEngine, 
                 rag: BaselineRAG,
                 test_queries: List[Tuple[str, pd.Timestamp]]):
                     
        print("\n" + "="*50)
        print("RUNNING BENCHMARK & EVALUATION")
        print("="*50)
        
        # 1. Storage Comparison (Approximate bytes in memory)
        # Extremely rough approximation for demonstration
        import sys
        graph_storage = sys.getsizeof(graph.graph) + len(graph.graph.nodes) * 100 + len(graph.graph.edges) * 100
        rag_storage = sys.getsizeof(rag.documents) + sum(len(d['text']) for d in rag.documents)
        
        self.metrics['graph_storage_kb'] = graph_storage / 1024
        self.metrics['rag_storage_kb'] = rag_storage / 1024
        
        # 2. Retrieval Latency & Context Size
        tm_latencies = []
        rag_latencies = []
        
        tm_context_sizes = []
        rag_context_sizes = []
        
        temporal_violations = 0
        
        for entity_id, timestamp in test_queries:
            # Temporal Memory
            t0 = time.time()
            context_tm = engine.retrieve(entity_id, timestamp)
            t1 = time.time()
            tm_latencies.append(t1 - t0)
            
            context_str = json.dumps(context_tm.to_dict(), default=str)
            tm_context_sizes.append(len(context_str))
            
            # Temporal Correctness Verification (TM)
            if self._has_future_leak(context_tm, timestamp):
                temporal_violations += 1
                
            # Baseline RAG
            t0 = time.time()
            context_rag = rag.retrieve(entity_id, timestamp)
            t1 = time.time()
            rag_latencies.append(t1 - t0)
            
            rag_context_sizes.append(len(context_rag))
            
        self.metrics['avg_tm_latency_ms'] = np.mean(tm_latencies) * 1000
        self.metrics['avg_rag_latency_ms'] = np.mean(rag_latencies) * 1000
        
        self.metrics['avg_tm_context_chars'] = np.mean(tm_context_sizes)
        self.metrics['avg_rag_context_chars'] = np.mean(rag_context_sizes)
        
        # Token estimation (~4 chars per token)
        self.metrics['avg_tm_tokens'] = self.metrics['avg_tm_context_chars'] / 4
        self.metrics['avg_rag_tokens'] = self.metrics['avg_rag_context_chars'] / 4
        
        self.metrics['temporal_violations'] = temporal_violations
        
        self.print_results()
        
    def _has_future_leak(self, context: RetrievalContext, query_time: pd.Timestamp) -> bool:
        """Verify no events in context are >= query_time"""
        for event in context.recent_events:
            ev_time = pd.to_datetime(event.get('timestamp'))
            if ev_time >= query_time:
                return True
        # State and trends are checked by their respective modules, 
        # but robust verification would check their inner timestamps too.
        if 'last_activity' in context.continuous_state:
            last_act = context.continuous_state['last_activity']
            if last_act and pd.to_datetime(last_act) >= query_time:
                return True
        return False
        
    def print_results(self):
        print("\n=== Benchmark Table ===")
        print(f"{'Metric':<30} | {'Temporal World Memory':<25} | {'Baseline RAG':<25}")
        print("-" * 85)
        print(f"{'Storage Usage (KB)':<30} | {self.metrics['graph_storage_kb']:<25.2f} | {self.metrics['rag_storage_kb']:<25.2f}")
        print(f"{'Avg Retrieval Latency (ms)':<30} | {self.metrics['avg_tm_latency_ms']:<25.2f} | {self.metrics['avg_rag_latency_ms']:<25.2f}")
        print(f"{'Avg Context Size (chars)':<30} | {self.metrics['avg_tm_context_chars']:<25.2f} | {self.metrics['avg_rag_context_chars']:<25.2f}")
        print(f"{'Est. Token Count':<30} | {self.metrics['avg_tm_tokens']:<25.2f} | {self.metrics['avg_rag_tokens']:<25.2f}")
        
        print("\n=== Temporal Correctness ===")
        if self.metrics['temporal_violations'] == 0:
            print("Status: PASSED (No future information leaked into past retrieval)")
        else:
            print(f"Status: FAILED ({self.metrics['temporal_violations']} violations detected!)")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("Starting Temporal World Memory Prototype...")
    
    # 1. Loader
    loader = DatasetLoader(CONFIG["dataset_name"])
    loader.load()
    
    # 2. Graph
    graph = TemporalWorldGraph(loader)
    graph.build()
    
    # 3. Hierarchy
    hierarchy = HierarchicalWorldMemory(graph)
    hierarchy.build_hierarchy()
    
    # 4. State
    states = DynamicStateMemory(loader, graph, alpha=CONFIG["alpha_ema"])
    states.build_states()
    
    # 5. Engine
    engine = TemporalRetrievalEngine(graph, hierarchy, states)
    
    # 6. Baseline RAG
    rag = BaselineRAG(loader)
    rag.build()
    
    # Select sample queries for evaluation
    # We pick some events and query the state of involved entities right before the event
    print("\n[*] Generating test queries...")
    test_queries = []
    
    # Collect some valid entities and timestamps
    # Using random sampling for benchmark
    event_nodes = [n for n, attr in graph.graph.nodes(data=True) if attr.get('type', '').startswith('event_')]
    np.random.seed(42)
    sample_size = min(50, len(event_nodes))
    sampled_events = np.random.choice(event_nodes, size=sample_size, replace=False)
    
    for ev in sampled_events:
        ev_data = graph.graph.nodes[ev]
        timestamp = ev_data['timestamp']
        # Find an entity involved
        involved = [v for u, v in graph.graph.out_edges(ev) if not str(v).startswith("Event_")]
        if involved:
            test_queries.append((involved[0], timestamp))
            
    # 7. Benchmark
    bench = Benchmark()
    bench.evaluate(graph, engine, rag, test_queries)
    
    # 8. Output Sample LLM Context
    if test_queries:
        sample_query = test_queries[0]
        print("\n" + "="*50)
        print(f"SAMPLE LLM CONTEXT (Entity: {sample_query[0]}, Time: {sample_query[1]})")
        print("="*50)
        context = engine.retrieve(sample_query[0], sample_query[1])
        print(json.dumps(context.to_dict(), indent=2, default=str))

if __name__ == "__main__":
    main()
