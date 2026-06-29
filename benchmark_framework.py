"""
Research Benchmark Expansion Framework
========================================
Scientifically rigorous benchmarking of Temporal World Memory (TWM).
"""

import time
import json
import collections
import sys
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from relbench.datasets import get_dataset

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================
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
# CORE ARCHITECTURE (Fixed logic, adapted for incremental updates)
# ==============================================================================
class DatasetLoader:
    def __init__(self, dataset_name: str = "rel-salt"):
        self.dataset_name = dataset_name
        self.dataset = get_dataset(self.dataset_name)
        self.db = self.dataset.get_db()
        self.entity_tables: Dict[str, Any] = {}
        self.event_tables: Dict[str, Any] = {}
        
        for table_name, table in self.db.table_dict.items():
            if table.time_col is None:
                self.entity_tables[table_name] = table
            else:
                self.event_tables[table_name] = table

class TemporalWorldGraph:
    def __init__(self, loader: DatasetLoader):
        self.loader = loader
        self.graph = nx.MultiDiGraph()
        self.is_initialized = False

    def init_entities(self):
        if self.is_initialized: return
        for table_name, table in self.loader.entity_tables.items():
            df = table.df.head(30000)
            pkey = table.pkey_col
            fkeys = table.fkey_col_to_pkey_table
            
            records = df.to_dict('records')
            for row in records:
                node_id = f"{table_name}::{row[pkey]}"
                self.graph.add_node(node_id, type=table_name, **row)
                for fk_col, target_table in fkeys.items():
                    fk_val = row.get(fk_col)
                    if pd.notna(fk_val):
                        target_id = f"{target_table}::{fk_val}"
                        if not self.graph.has_node(target_id):
                            self.graph.add_node(target_id, type=target_table)
                        self.graph.add_edge(node_id, target_id, type='structural_fk', fk_col=fk_col, timestamp=pd.Timestamp.min)
        self.is_initialized = True

    def add_events(self, events: List[Dict[str, Any]]):
        self.init_entities()
        for ev in events:
            event_node_id = f"Event_{ev['table_name']}::{ev['pkey_val']}"
            timestamp = ev['timestamp']
            
            self.graph.add_node(event_node_id, type=f"event_{ev['table_name']}", timestamp=timestamp, **ev['row'])
            
            for fk_col, target_table in ev['fkeys'].items():
                fk_val = ev['row'].get(fk_col)
                if pd.notna(fk_val):
                    target_id = f"{target_table}::{fk_val}"
                    if not self.graph.has_node(target_id):
                        self.graph.add_node(target_id, type=target_table)
                    self.graph.add_edge(event_node_id, target_id, type=fk_col, timestamp=timestamp)
                    self.graph.add_edge(target_id, event_node_id, type=f"involved_in_{fk_col}", timestamp=timestamp)

class HierarchicalWorldMemory:
    def __init__(self, graph: TemporalWorldGraph):
        self.graph = graph
        self.memory = collections.defaultdict(lambda: collections.defaultdict(dict))
        
    def add_events(self, events: List[Dict[str, Any]]):
        for ev in events:
            timestamp = ev['timestamp']
            event_type = f"event_{ev['table_name']}"
            
            # Find entities involved
            involved_entities = []
            for fk_col, target_table in ev['fkeys'].items():
                fk_val = ev['row'].get(fk_col)
                if pd.notna(fk_val):
                    involved_entities.append(f"{target_table}::{fk_val}")
            
            day_key = timestamp.strftime('%Y-%m-%d')
            week_key = timestamp.strftime('%Y-%W')
            month_key = timestamp.strftime('%Y-%m')
            
            for entity in involved_entities:
                # Level 1: Daily
                if day_key not in self.memory[entity]['daily']:
                    self.memory[entity]['daily'][day_key] = {'event_count': 0, 'event_types': set()}
                self.memory[entity]['daily'][day_key]['event_count'] += 1
                self.memory[entity]['daily'][day_key]['event_types'].add(event_type)
                
                # Level 2: Weekly
                if week_key not in self.memory[entity]['weekly']:
                    self.memory[entity]['weekly'][week_key] = {'event_count': 0, 'event_types': set()}
                self.memory[entity]['weekly'][week_key]['event_count'] += 1
                self.memory[entity]['weekly'][week_key]['event_types'].add(event_type)
                
                # Level 3: Monthly
                if month_key not in self.memory[entity]['monthly']:
                    self.memory[entity]['monthly'][month_key] = {'event_count': 0, 'event_types': set()}
                self.memory[entity]['monthly'][month_key]['event_count'] += 1
                self.memory[entity]['monthly'][month_key]['event_types'].add(event_type)
                
                # Level 4: Lifetime
                if 'lifetime' not in self.memory[entity]:
                    self.memory[entity]['lifetime'] = {'total_events': 0, 'first_seen': timestamp, 'last_seen': timestamp}
                self.memory[entity]['lifetime']['total_events'] += 1
                if timestamp < self.memory[entity]['lifetime']['first_seen']:
                    self.memory[entity]['lifetime']['first_seen'] = timestamp
                if timestamp > self.memory[entity]['lifetime']['last_seen']:
                    self.memory[entity]['lifetime']['last_seen'] = timestamp

    def get_summary(self, entity_id: str, timestamp: pd.Timestamp) -> Dict[str, Any]:
        summary = {'daily_recent': {}, 'weekly_recent': {}, 'monthly_recent': {}, 'lifetime': {}}
        if entity_id not in self.memory: return summary
            
        entity_mem = self.memory[entity_id]
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
                    'total_events_approx': len(summary['daily_recent']),
                    'first_seen': lt['first_seen']
                }
        return summary

class DynamicStateMemory:
    def __init__(self, loader: DatasetLoader, graph: TemporalWorldGraph, alpha: float = 0.1):
        self.loader = loader
        self.graph = graph
        self.alpha = alpha
        self.state_history = collections.defaultdict(list)
        self.current_state = collections.defaultdict(lambda: {'activity_score': 0.0, 'last_activity': None})
        
    def add_events(self, events: List[Dict[str, Any]]):
        for ev in events:
            timestamp = ev['timestamp']
            involved = []
            for fk_col, target_table in ev['fkeys'].items():
                fk_val = ev['row'].get(fk_col)
                if pd.notna(fk_val):
                    involved.append(f"{target_table}::{fk_val}")
                    
            for entity in involved:
                state = self.current_state[entity]
                state['activity_score'] = (1 - self.alpha) * state['activity_score'] + self.alpha * 1.0
                state['last_activity'] = timestamp
                self.state_history[entity].append((timestamp, dict(state)))
                
    def get_state(self, entity_id: str, timestamp: pd.Timestamp) -> Dict[str, float]:
        if entity_id not in self.state_history: return {}
        history = self.state_history[entity_id]
        for i in range(len(history)-1, -1, -1):
            if history[i][0] < timestamp:
                return history[i][1]
        return {}

# ==============================================================================
# BASELINES
# ==============================================================================

class BaselineA_FlatRAG:
    """Retrieve every historical document before timestamp."""
    def __init__(self):
        self.documents = []
        
    def add_events(self, events: List[Dict[str, Any]]):
        for ev in events:
            timestamp = ev['timestamp']
            text_repr = f"At {timestamp.isoformat()}, Event {ev['table_name']} occurred. Details: "
            text_repr += ", ".join([f"{k}={v}" for k, v in ev['row'].items()])
            self.documents.append({
                'timestamp': timestamp,
                'text': text_repr,
                'raw_row': ev['row']
            })
        self.documents.sort(key=lambda x: x['timestamp'])
        
    def retrieve(self, entity_id: str, timestamp: pd.Timestamp) -> str:
        retrieved_docs = []
        raw_id = str(entity_id.split("::")[-1])
        for doc in self.documents:
            if doc['timestamp'] >= timestamp:
                break
            # keyword match
            if raw_id in doc['text']:
                retrieved_docs.append(doc['text'])
        return "\n".join(retrieved_docs)

class BaselineB_TopK_RAG:
    """Retrieve top K most recent documents."""
    def __init__(self, k=20):
        self.documents = []
        self.k = k
        
    def add_events(self, events: List[Dict[str, Any]]):
        for ev in events:
            timestamp = ev['timestamp']
            text_repr = f"At {timestamp.isoformat()}, Event {ev['table_name']} occurred. Details: "
            text_repr += ", ".join([f"{k}={v}" for k, v in ev['row'].items()])
            self.documents.append({
                'timestamp': timestamp,
                'text': text_repr
            })
        self.documents.sort(key=lambda x: x['timestamp'])
        
    def retrieve(self, entity_id: str, timestamp: pd.Timestamp) -> str:
        retrieved_docs = []
        raw_id = str(entity_id.split("::")[-1])
        # Find all valid
        valid = [doc for doc in self.documents if doc['timestamp'] < timestamp and raw_id in doc['text']]
        # Take Top K recent
        valid = sorted(valid, key=lambda x: x['timestamp'], reverse=True)[:self.k]
        return "\n".join([doc['text'] for doc in valid])

class BaselineC_MetadataRAG:
    """Exact metadata filtering."""
    def __init__(self):
        self.documents = []
        
    def add_events(self, events: List[Dict[str, Any]]):
        for ev in events:
            timestamp = ev['timestamp']
            involved = set()
            for fk_col, target_table in ev['fkeys'].items():
                fk_val = ev['row'].get(fk_col)
                if pd.notna(fk_val):
                    involved.add(f"{target_table}::{fk_val}")
            
            text_repr = f"At {timestamp.isoformat()}, Event {ev['table_name']} occurred. Details: "
            text_repr += ", ".join([f"{k}={v}" for k, v in ev['row'].items()])
            
            self.documents.append({
                'timestamp': timestamp,
                'text': text_repr,
                'involved': involved
            })
        self.documents.sort(key=lambda x: x['timestamp'])
        
    def retrieve(self, entity_id: str, timestamp: pd.Timestamp, window_days: int = 30) -> str:
        cutoff = timestamp - pd.Timedelta(days=window_days)
        retrieved_docs = []
        for doc in self.documents:
            if doc['timestamp'] >= timestamp: break
            if doc['timestamp'] >= cutoff and entity_id in doc['involved']:
                retrieved_docs.append(doc['text'])
        return "\n".join(retrieved_docs)

class BaselineSystem:
    """Wrapper to run ablation components dynamically."""
    def __init__(self, use_graph: bool, use_hierarchy: bool, use_state: bool):
        self.use_graph = use_graph
        self.use_hierarchy = use_hierarchy
        self.use_state = use_state
        self.graph = None
        self.hierarchy = None
        self.states = None
        
    def bind(self, graph, hierarchy, states):
        self.graph = graph
        self.hierarchy = hierarchy
        self.states = states
        
    def retrieve(self, entity_id: str, timestamp: pd.Timestamp, window_days: int = 30) -> RetrievalContext:
        entity_summary = {}
        if self.graph.graph.has_node(entity_id):
            entity_summary = {k: v for k, v in self.graph.graph.nodes[entity_id].items() if k != 'type'}
            
        continuous_state = self.states.get_state(entity_id, timestamp) if self.use_state else {}
        historical_trends = self.hierarchy.get_summary(entity_id, timestamp) if self.use_hierarchy else {}
        
        recent_events = []
        neighbor_entities = []
        if self.use_graph:
            cutoff_time = timestamp - pd.Timedelta(days=window_days)
            if self.graph.graph.has_node(entity_id):
                for u, v, k, d in self.graph.graph.out_edges(entity_id, data=True, keys=True):
                    edge_time = d.get('timestamp')
                    if edge_time is not None and cutoff_time <= edge_time < timestamp:
                        if str(v).startswith("Event_"):
                            event_data = self.graph.graph.nodes[v].copy()
                            if 'timestamp' in event_data:
                                event_data['timestamp'] = event_data['timestamp'].isoformat()
                            recent_events.append(event_data)
                            for ev_u, ev_v, ev_k, ev_d in self.graph.graph.out_edges(v, data=True, keys=True):
                                if ev_v != entity_id and not str(ev_v).startswith("Event_"):
                                    neighbor_entities.append({'entity_id': ev_v, 'relation': ev_k})
                                    
        return RetrievalContext(entity_summary, continuous_state, recent_events, neighbor_entities, historical_trends)

# ==============================================================================
# EVALUATION & BENCHMARK SUITE
# ==============================================================================
class BenchmarkSuite:
    def __init__(self, dataset_name="rel-salt"):
        self.loader = DatasetLoader(dataset_name)
        # We will extract events and sort them globally
        self.all_events = []
        self._prepare_events()
        
    def _prepare_events(self):
        print("[*] Preparing global event stream...")
        for table_name, table in self.loader.event_tables.items():
            df = table.df.sort_values(table.time_col).head(30000)
            time_col = table.time_col
            pkey = table.pkey_col
            fkeys = table.fkey_col_to_pkey_table
            
            # Use faster extraction
            records = df.to_dict('records')
            for row in records:
                self.all_events.append({
                    'timestamp': pd.to_datetime(row[time_col]),
                    'table_name': table_name,
                    'pkey_val': row[pkey],
                    'row': row,
                    'fkeys': fkeys
                })
        self.all_events.sort(key=lambda x: x['timestamp'])
        print(f"[*] Total events prepared: {len(self.all_events)}")
        
    def estimate_llm_cost(self, chars: int) -> Dict[str, float]:
        tokens = chars / 4
        # Approximations per 1M input tokens
        gpt4o = (tokens / 1_000_000) * 5.00
        claude35 = (tokens / 1_000_000) * 3.00
        gemini15 = (tokens / 1_000_000) * 3.50
        return {"GPT-4o": gpt4o, "Claude-3.5": claude35, "Gemini-1.5": gemini15}

    def _verify_temporal_correctness(self, context_str: str, query_time: pd.Timestamp) -> bool:
        # A simple check: if any timestamp substring in context is >= query_time
        # In a strict check, we iterate all datetime-like strings
        import re
        dates = re.findall(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', context_str)
        for d in dates:
            if pd.to_datetime(d) >= query_time:
                return False
        return True

    def run_ablation_study(self, limit=5000):
        print("\n" + "="*50)
        print("RUNNING ABLATION STUDY")
        print("="*50)
        
        events = self.all_events[:limit]
        
        # Build shared core
        graph = TemporalWorldGraph(self.loader)
        graph.add_events(events)
        hierarchy = HierarchicalWorldMemory(graph)
        hierarchy.add_events(events)
        states = DynamicStateMemory(self.loader, graph)
        states.add_events(events)
        
        # Baselines
        baselines = {
            "D (Graph)": BaselineSystem(True, False, False),
            "E (Graph+Hierarchy)": BaselineSystem(True, True, False),
            "F (Graph+State)": BaselineSystem(True, False, True),
            "G (Full TWM)": BaselineSystem(True, True, True),
        }
        for b in baselines.values(): b.bind(graph, hierarchy, states)
        
        # RAG Baselines
        rag_flat = BaselineA_FlatRAG()
        rag_flat.add_events(events)
        rag_topk = BaselineB_TopK_RAG(k=20)
        rag_topk.add_events(events)
        rag_meta = BaselineC_MetadataRAG()
        rag_meta.add_events(events)
        
        # Sample queries
        queries = []
        np.random.seed(42)
        for i in range(100):
            ev = np.random.choice(events)
            target = next((f"{t}::{v}" for k,t in ev['fkeys'].items() if pd.notna(v:=ev['row'].get(k))), None)
            if target: queries.append((target, ev['timestamp']))
            
        results = []
        all_systems = {
            "A (Flat RAG)": rag_flat,
            "B (Top-20 RAG)": rag_topk,
            "C (Metadata RAG)": rag_meta,
            **baselines
        }
        
        for name, system in all_systems.items():
            lats, sizes, tokens = [], [], []
            correct = True
            
            for entity_id, ts in queries:
                t0 = time.time()
                if hasattr(system, 'retrieve') and not isinstance(system, BaselineSystem):
                    ctx = system.retrieve(entity_id, ts)
                    ctx_str = ctx
                else:
                    ctx = system.retrieve(entity_id, ts)
                    ctx_str = json.dumps(ctx.to_dict(), default=str)
                lats.append((time.time()-t0)*1000)
                sizes.append(len(ctx_str))
                tokens.append(len(ctx_str)/4)
                
                if not self._verify_temporal_correctness(ctx_str, ts):
                    correct = False
            
            cost = self.estimate_llm_cost(np.mean(sizes))
            results.append({
                "Architecture": name,
                "Avg Latency (ms)": np.mean(lats),
                "P95 Latency (ms)": np.percentile(lats, 95),
                "Avg Context Size": np.mean(sizes),
                "Tokens": np.mean(tokens),
                "Cost/10k_Queries ($)": cost["GPT-4o"] * 10000,
                "Temporal Correctness": "PASS" if correct else "FAIL"
            })
            
        df_res = pd.DataFrame(results)
        print(df_res.to_markdown(index=False))
        return df_res

    def run_scaling_benchmark(self):
        print("\n" + "="*50)
        print("RUNNING SCALING BENCHMARK")
        print("="*50)
        
        scales = [100, 500, 1000, 5000, 10000, 25000]
        scaling_results = []
        
        for s in scales:
            if s > len(self.all_events): break
            events = self.all_events[:s]
            
            t0 = time.time()
            graph = TemporalWorldGraph(self.loader)
            graph.add_events(events)
            hierarchy = HierarchicalWorldMemory(graph)
            hierarchy.add_events(events)
            states = DynamicStateMemory(self.loader, graph)
            states.add_events(events)
            twm_build_time = time.time() - t0
            
            t0 = time.time()
            rag = BaselineA_FlatRAG()
            rag.add_events(events)
            rag_build_time = time.time() - t0
            
            # memory size
            twm_mem = sys.getsizeof(graph.graph) + sys.getsizeof(hierarchy.memory) + sys.getsizeof(states.state_history)
            rag_mem = sys.getsizeof(rag.documents) + sum(len(d['text']) for d in rag.documents)
            
            scaling_results.append({
                "Events": s,
                "TWM Build (s)": twm_build_time,
                "RAG Build (s)": rag_build_time,
                "TWM Mem (KB)": twm_mem / 1024,
                "RAG Mem (KB)": rag_mem / 1024,
            })
            
        df_scale = pd.DataFrame(scaling_results)
        print(df_scale.to_markdown(index=False))
        
        # Plotting
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(df_scale['Events'], df_scale['TWM Mem (KB)'], label='TWM', marker='o')
        plt.plot(df_scale['Events'], df_scale['RAG Mem (KB)'], label='Flat RAG', marker='s')
        plt.title('Memory Growth')
        plt.xlabel('Events processed')
        plt.ylabel('Memory (KB)')
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(df_scale['Events'], df_scale['TWM Build (s)'], label='TWM', marker='o')
        plt.plot(df_scale['Events'], df_scale['RAG Build (s)'], label='Flat RAG', marker='s')
        plt.title('Ingestion Time Scaling')
        plt.xlabel('Events processed')
        plt.ylabel('Time (s)')
        plt.legend()
        
        plt.tight_layout()
        if not os.path.exists("scratch"): os.makedirs("scratch")
        plt.savefig("scratch/scaling_benchmark.png")
        print("[*] Saved scaling_benchmark.png")

    def run_incremental_update_benchmark(self, chunk_size=2000, max_chunks=5):
        print("\n" + "="*50)
        print("RUNNING INCREMENTAL UPDATE BENCHMARK")
        print("="*50)
        
        graph = TemporalWorldGraph(self.loader)
        hierarchy = HierarchicalWorldMemory(graph)
        states = DynamicStateMemory(self.loader, graph)
        rag = BaselineA_FlatRAG()
        
        results = []
        for i in range(max_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            if start_idx >= len(self.all_events): break
            
            chunk = self.all_events[start_idx:end_idx]
            
            # TWM update (O(K))
            t0 = time.time()
            graph.add_events(chunk)
            hierarchy.add_events(chunk)
            states.add_events(chunk)
            twm_lat = (time.time() - t0) * 1000
            
            # Naive RAG Update (often requires complete re-indexing in production vector DBs, 
            # here we just append to list but we'll measure the append cost)
            t0 = time.time()
            rag.add_events(chunk)
            rag_lat = (time.time() - t0) * 1000
            
            results.append({
                "Chunk": i+1,
                "Total Events": end_idx,
                "TWM Update Latency (ms)": twm_lat,
                "RAG Update Latency (ms)": rag_lat,
                "Graph Nodes added": graph.graph.number_of_nodes(),
                "Graph Edges added": graph.graph.number_of_edges()
            })
            
        df = pd.DataFrame(results)
        print(df.to_markdown(index=False))
        
        plt.figure(figsize=(6,4))
        plt.plot(df['Total Events'], df['TWM Update Latency (ms)'], label='TWM Update', marker='o')
        plt.title('Incremental Update Latency')
        plt.xlabel('Total Events')
        plt.ylabel('Update Latency (ms) for chunk')
        plt.legend()
        plt.savefig("scratch/incremental_update.png")
        print("[*] Saved incremental_update.png")

if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run_ablation_study(limit=5000)
    suite.run_scaling_benchmark()
    suite.run_incremental_update_benchmark()
    print("\n[*] Benchmarks complete.")
