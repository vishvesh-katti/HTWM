"""
HTWM Final Experiment: End-to-End Evaluation
==============================================
Ultimate Capstone Validation testing HTWM against Flat RAG, Top-K, BM25,
FAISS, and Graph Retrieval. Computes Pareto frontiers, F1 vs Token tradeoff,
and formal statistical significance (Wilcoxon).
"""

import time
import json
import random
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import wilcoxon

# Import HTWM Phase I core and Phase II baselines
from htwm_prototype import EventExtractor, HTWM, FlatRAG, TopKRAG
from htwm_phase2_eval import BM25RAG, FaissRAG

os.makedirs("scratch", exist_ok=True)
np.random.seed(42)
random.seed(42)

# ==============================================================================
# NEW BASELINE: GRAPH RETRIEVAL
# ==============================================================================
class GraphRAG:
    """Standard Ego-Graph Traversal RAG (K-hop retrieval)"""
    def __init__(self, max_hops=1, max_nodes=20):
        import networkx as nx
        self.graph = nx.MultiDiGraph()
        self.max_hops = max_hops
        self.max_nodes = max_nodes
        
    def ingest(self, events):
        for ev in events:
            for i, e1 in enumerate(ev['involved']):
                self.graph.add_node(e1, label=e1.split('::')[0])
                for e2 in ev['involved'][i+1:]:
                    self.graph.add_edge(e1, e2, timestamp=ev['timestamp'], type=ev['type'])
                    self.graph.add_edge(e2, e1, timestamp=ev['timestamp'], type=ev['type'])
                    
    def retrieve(self, entity, target_ts):
        import networkx as nx
        if entity not in self.graph: return ""
        # Get ego graph (1 hop)
        ego = nx.ego_graph(self.graph, entity, radius=self.max_hops)
        # Filter edges by time and construct context
        context_lines = []
        for u, v, data in ego.edges(data=True):
            if data['timestamp'] <= target_ts:
                context_lines.append(f"{u} {data['type']} {v} at {data['timestamp']}")
            if len(context_lines) >= self.max_nodes: break
        return "\n".join(context_lines)

# ==============================================================================
# BUSINESS QA GENERATOR & LLM SIMULATOR
# ==============================================================================
class FinalQAGenerator:
    def __init__(self, events):
        self.events = events
        
    def generate(self, count=100):
        qa_pairs = []
        valid_events = [ev for ev in self.events if len(ev['involved']) > 0]
        
        entity_hist = defaultdict(list)
        for ev in valid_events:
            for e in ev['involved']:
                entity_hist[e].append(ev)
                
        valid_entities = [e for e, hist in entity_hist.items() if len(hist) > 5]
        
        for i in range(count):
            entity = random.choice(valid_entities)
            hist = entity_hist[entity]
            mid_idx = max(1, len(hist) // 2)
            target_ts = hist[mid_idx]['timestamp']
            
            # Ground truth construction (what actually happened before T)
            prior_events = [ev for ev in hist if ev['timestamp'] <= target_ts]
            facts = set()
            for p_ev in prior_events:
                facts.add(p_ev['type'])
                for inv in p_ev['involved']: facts.add(inv)
                
            qa_pairs.append({
                'entity': entity,
                'query': f"What happened to {entity} before {target_ts}?",
                'target_ts': target_ts,
                'ground_truth': facts
            })
        return qa_pairs

class StrictLLMSimulator:
    @staticmethod
    def evaluate(context_str, ground_truth_facts):
        ctx_lower = context_str.lower()
        if len(ctx_lower.strip()) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0, len(ground_truth_facts), 0, 0.0
            
        hits = sum(1 for f in ground_truth_facts if str(f).lower() in ctx_lower)
        total_facts = len(ground_truth_facts)
        
        tokens = len(ctx_lower) / 4.0 # generic token estimator
        api_cost = (tokens / 1000.0) * 0.001
        
        # Hallucinations logic: count extra 'noise' keywords that aren't ground truth
        # Very rough heuristic: every 100 characters that aren't hits count as a hallucinated/irrelevant fact
        est_retrieved = max(hits, len(ctx_lower) / 100)
        hallucinations = max(0, est_retrieved - hits)
        
        recall = hits / total_facts if total_facts > 0 else 1.0
        precision = hits / est_retrieved if est_retrieved > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        exact_match = 1.0 if recall == 1.0 else 0.0
        missing = total_facts - hits
        
        return precision, recall, f1, exact_match, hits, missing, hallucinations, api_cost

# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================
class EndToEndEvaluation:
    def __init__(self):
        print("[*] Loading 10,000 events for Final Benchmark...")
        self.extractor = EventExtractor(limit=10000)
        self.events, self.entities = self.extractor.extract()
        
    def _format_table(self, title, df):
        print(f"\n=== {title} ===")
        print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
        
    def run(self):
        qa_pairs = FinalQAGenerator(self.events).generate(100)
        
        systems = {
            "Flat RAG": FlatRAG(),
            "Top-K RAG": TopKRAG(),
            "BM25": BM25RAG(),
            "FAISS": FaissRAG(),
            "Graph Retrieval": GraphRAG(),
            "HTWM": HTWM()
        }
        
        print("[*] Ingesting data into all architectures...")
        for name, sys in systems.items():
            if name == "HTWM": sys.ingest(self.events, self.entities)
            else: sys.ingest(self.events)
            
        print("[*] Running 100 queries against architectures...")
        metrics = {name: defaultdict(list) for name in systems.keys()}
        
        for qa in qa_pairs:
            for name, sys in systems.items():
                t0 = time.time()
                if name == "HTWM": ctx = sys.retrieve_context(qa['entity'], qa['target_ts'])
                elif name == "Graph Retrieval": ctx = sys.retrieve(qa['entity'], qa['target_ts'])
                else: ctx = sys.retrieve(qa['query'], qa['target_ts'])
                lat = (time.time() - t0)*1000
                
                p, r, f1, em, hits, missing, hallu, cost = StrictLLMSimulator.evaluate(ctx, qa['ground_truth'])
                tokens = len(ctx) / 4.0
                
                metrics[name]['lat'].append(lat)
                metrics[name]['f1'].append(f1)
                metrics[name]['p'].append(p)
                metrics[name]['r'].append(r)
                metrics[name]['hallu'].append(hallu)
                metrics[name]['tok'].append(tokens)
                metrics[name]['cost'].append(cost)

        # Aggregate
        results = []
        for name in systems.keys():
            m = metrics[name]
            results.append({
                "Architecture": name,
                "F1": np.mean(m['f1']),
                "Precision": np.mean(m['p']),
                "Recall": np.mean(m['r']),
                "Hallucinations": np.mean(m['hallu']),
                "Context Tokens": np.mean(m['tok']),
                "Retrieval Latency (ms)": np.mean(m['lat']),
                "API Cost ($)": np.mean(m['cost'])
            })
            
        df = pd.DataFrame(results)
        self._format_table("Final Aggregate Results", df)
        df.to_csv("scratch/final_results.csv", index=False)
        
        # Statistical Significance (HTWM vs Best Baseline by F1)
        # Find best baseline
        baselines = [n for n in systems.keys() if n != "HTWM"]
        best_baseline = max(baselines, key=lambda n: np.mean(metrics[n]['f1']))
        
        htwm_f1s = metrics['HTWM']['f1']
        base_f1s = metrics[best_baseline]['f1']
        
        # We need variance for wilcoxon, if all diffs are 0 it fails, add tiny noise
        diffs = np.array(htwm_f1s) - np.array(base_f1s)
        if np.all(diffs == 0):
            p_val = 1.0
        else:
            _, p_val = wilcoxon(htwm_f1s, base_f1s, zero_method='zsplit')
            
        print(f"\n[*] Statistical Significance: HTWM vs {best_baseline}")
        print(f"Wilcoxon signed-rank p-value: {p_val:.6f}")
        print(f"Significant (p < 0.05)? {'YES' if p_val < 0.05 else 'NO'}")
        
        # ==============================================================================
        # PLOTTING
        # ==============================================================================
        # Plot 1: Pareto Frontier (Tokens vs F1)
        plt.figure(figsize=(8,6))
        for name in systems.keys():
            x = np.mean(metrics[name]['tok'])
            y = np.mean(metrics[name]['f1'])
            
            color = 'red' if name == 'HTWM' else 'blue'
            marker = '*' if name == 'HTWM' else 'o'
            s = 200 if name == 'HTWM' else 100
            
            plt.scatter(x, y, label=name, color=color, marker=marker, s=s)
            plt.text(x*1.05, y*1.02, name, fontsize=10, weight='bold' if name=='HTWM' else 'normal')
            
        plt.xlabel("Average Context Tokens (Lower is Better)")
        plt.ylabel("Average F1 Score (Higher is Better)")
        plt.title("Pareto Frontier: Retrieval Quality vs Context Cost")
        plt.grid(True)
        plt.savefig("scratch/pareto_frontier.png")
        
        # Plot 2: Latency vs F1
        plt.figure(figsize=(8,6))
        for name in systems.keys():
            x = np.mean(metrics[name]['lat'])
            y = np.mean(metrics[name]['f1'])
            
            color = 'red' if name == 'HTWM' else 'green'
            marker = '*' if name == 'HTWM' else 's'
            s = 200 if name == 'HTWM' else 100
            
            plt.scatter(x, y, label=name, color=color, marker=marker, s=s)
            plt.text(x*1.05, y*1.02, name, fontsize=10, weight='bold' if name=='HTWM' else 'normal')
            
        plt.xlabel("Retrieval Latency (ms) (Lower is Better)")
        plt.ylabel("Average F1 Score (Higher is Better)")
        plt.title("Efficiency: Retrieval Latency vs Reasoning Quality")
        plt.xscale('log') # Log scale for latency
        plt.grid(True)
        plt.savefig("scratch/latency_frontier.png")
        
if __name__ == "__main__":
    EndToEndEvaluation().run()
    print("\n[*] FINAL VALIDATION COMPLETE.")
