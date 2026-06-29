"""
HTWM Phase II: Scientific Validation
====================================
Rigorous benchmarking suite evaluating HTWM against FAISS, BM25, and Flat RAG.
Outputs deterministic QA scores, scaling metrics, and memory boundaries.
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
import faiss
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# Import frozen HTWM architecture from Phase I
from htwm_prototype import EventExtractor, HTWM, FlatRAG, TopKRAG

os.makedirs("scratch", exist_ok=True)
MAX_EVENTS = 500000

# ==============================================================================
# NEW BASELINES: BM25 & FAISS
# ==============================================================================
class BM25RAG:
    def __init__(self):
        self.docs = []
        self.corpus = []
        self.bm25 = None
        
    def ingest(self, events):
        for ev in events:
            text = json.dumps(ev, default=str)
            self.docs.append((ev['timestamp'], text))
            self.corpus.append(text.split(" "))
        self.docs.sort(key=lambda x: x[0])
        self.bm25 = BM25Okapi(self.corpus)
        
    def retrieve(self, query_str, target_ts, k=20):
        # Time filtering (simulating post-retrieval or pre-retrieval filter)
        tokenized_query = query_str.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        # Pair with docs and filter by time
        valid = [(s, self.docs[i][1]) for i, s in enumerate(scores) if self.docs[i][0] <= target_ts]
        valid.sort(key=lambda x: x[0], reverse=True)
        return "\n".join([v[1] for v in valid[:k]])

class FaissRAG:
    """Uses TF-IDF + random projection to simulate dense embeddings on CPU for latency testing."""
    def __init__(self, dim=64):
        self.docs = []
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.vectorizer = TfidfVectorizer(max_features=dim)
        self.texts = []
        
    def ingest(self, events):
        texts = [json.dumps(ev, default=str) for ev in events]
        self.texts.extend(texts)
        for ev in events:
            self.docs.append((ev['timestamp'], json.dumps(ev, default=str)))
        
        # Fit transform and pad to exact dim
        vectors = self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
        if vectors.shape[1] < self.dim:
            pad = np.zeros((vectors.shape[0], self.dim - vectors.shape[1]), dtype=np.float32)
            vectors = np.hstack((vectors, pad))
        self.index.add(vectors)
        self.docs.sort(key=lambda x: x[0])
        
    def retrieve(self, query_str, target_ts, k=20):
        vec = self.vectorizer.transform([query_str]).toarray().astype(np.float32)
        if vec.shape[1] < self.dim:
            pad = np.zeros((vec.shape[0], self.dim - vec.shape[1]), dtype=np.float32)
            vec = np.hstack((vec, pad))
        
        # We retrieve K*10 to account for time filtering
        D, I = self.index.search(vec, min(len(self.docs), k*10))
        res = []
        for idx in I[0]:
            if idx != -1 and self.docs[idx][0] <= target_ts:
                res.append(self.docs[idx][1])
                if len(res) >= k: break
        return "\n".join(res)

# ==============================================================================
# DETERMINISTIC EVALUATOR & QA GENERATOR
# ==============================================================================
class SyntheticQAGenerator:
    def __init__(self, events):
        self.events = events
        
    def generate(self, count=100):
        qa_pairs = []
        # Sample entities that have multiple events
        entity_hist = defaultdict(list)
        for ev in self.events:
            for e in ev['involved']:
                entity_hist[e].append(ev)
                
        valid_entities = [e for e, hist in entity_hist.items() if len(hist) > 5]
        if len(valid_entities) == 0:
            valid_entities = list(entity_hist.keys())
            
        for _ in range(count):
            entity = random.choice(valid_entities)
            hist = entity_hist[entity]
            
            # Pick a target timestamp in the middle of history
            mid_idx = max(1, len(hist) // 2)
            target_ts = hist[mid_idx]['timestamp']
            
            # Ground truth: what actually happened before T
            prior_events = [ev for ev in hist if ev['timestamp'] <= target_ts]
            # Extract ground truth facts (keywords, attributes)
            facts = set()
            for p_ev in prior_events:
                facts.add(p_ev['type'])
                for inv in p_ev['involved']: facts.add(inv)
                
            query = f"What happened to {entity} before {target_ts}?"
            qa_pairs.append({
                'entity': entity,
                'query': query,
                'target_ts': target_ts,
                'ground_truth': facts
            })
        return qa_pairs

class DeterministicJudge:
    @staticmethod
    def evaluate(context_str, ground_truth_facts):
        ctx_lower = context_str.lower()
        if len(ctx_lower.strip()) == 0:
            return 0.0, 0.0, 0.0, 0.0
            
        hits = sum(1 for f in ground_truth_facts if str(f).lower() in ctx_lower)
        total_facts = len(ground_truth_facts)
        
        # Heuristic calculation
        recall = hits / total_facts if total_facts > 0 else 1.0
        # Simulated precision based on length bloat (penalize massive contexts)
        # Assuming every 100 chars contains ~1 fact roughly
        estimated_retrieved_facts = max(hits, len(ctx_lower) / 100)
        precision = hits / estimated_retrieved_facts if estimated_retrieved_facts > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        exact_match = 1.0 if recall == 1.0 else 0.0
        
        return precision, recall, f1, exact_match

# ==============================================================================
# EXPERIMENT SUITE
# ==============================================================================
class PhaseIIExperiments:
    def __init__(self):
        print("[*] Loading datasets up to 100k events...")
        self.extractor = EventExtractor(limit=100000)
        self.events, self.entities = self.extractor.extract()
        
    def _get_chunk(self, size):
        return self.events[:size]

    def _format_table(self, title, df):
        print(f"\n=== {title} ===")
        print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))

    def run_experiment_1_and_3(self):
        # Combines Context Efficiency & Reasoning Quality
        chunk = self._get_chunk(10000)
        qa_gen = SyntheticQAGenerator(chunk)
        qa_pairs = qa_gen.generate(50)
        
        systems = {
            "Flat RAG": FlatRAG(),
            "Top-20 RAG": TopKRAG(),
            "BM25": BM25RAG(),
            "FAISS": FaissRAG(),
            "HTWM": HTWM()
        }
        
        # Ingest
        for name, sys in systems.items():
            if name == "HTWM":
                sys.ingest(chunk, self.entities)
            else:
                sys.ingest(chunk)
                
        results = []
        for name, sys in systems.items():
            latencies = []
            ctx_sizes = []
            f1s = []
            
            for qa in qa_pairs:
                t0 = time.time()
                if name == "HTWM":
                    ctx = sys.retrieve_context(qa['entity'], qa['target_ts'])
                else:
                    ctx = sys.retrieve(qa['query'], qa['target_ts'])
                lat = (time.time() - t0)*1000
                latencies.append(lat)
                ctx_sizes.append(len(ctx))
                
                # Eval
                p, r, f1, em = DeterministicJudge.evaluate(ctx, qa['ground_truth'])
                f1s.append(f1)
                
            results.append({
                "Architecture": name,
                "Latency (ms)": np.mean(latencies),
                "Context Size": np.mean(ctx_sizes),
                "Reasoning F1": np.mean(f1s)
            })
            
        df = pd.DataFrame(results)
        self._format_table("Exp 1 & 3: Efficiency & Reasoning Quality", df)
        
        # Plot Reasoning vs Context
        plt.figure(figsize=(7,5))
        for _, row in df.iterrows():
            plt.scatter(row['Context Size'], row['Reasoning F1'], label=row['Architecture'], s=100)
            plt.text(row['Context Size']*1.05, row['Reasoning F1'], row['Architecture'])
        plt.xlabel('Context Size (Chars)')
        plt.ylabel('Reasoning Quality (F1 Score)')
        plt.title('Reasoning Quality vs Context Cost')
        plt.grid(True)
        plt.savefig("scratch/reasoning_vs_context.png")

    def run_experiment_4(self):
        # Memory Compression Trade-off
        chunk = self._get_chunk(5000)
        htwm = HTWM()
        htwm.ingest(chunk, self.entities)
        
        qa_pairs = SyntheticQAGenerator(chunk).generate(20)
        thresholds = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        results = []
        
        target_ts = chunk[-1]['timestamp']
        
        for t in thresholds:
            htwm.compression.compress(target_ts, threshold=t)
            archived = len(htwm.compression.archived_nodes)
            
            f1s = []
            for qa in qa_pairs:
                ctx = htwm.retrieve_context(qa['entity'], qa['target_ts'])
                p, r, f1, em = DeterministicJudge.evaluate(ctx, qa['ground_truth'])
                f1s.append(f1)
                
            results.append({
                "Threshold": t,
                "Compression Rate (%)": (archived / max(1, len(htwm.graph.graph.nodes))) * 100,
                "Reasoning F1": np.mean(f1s)
            })
            
        df = pd.DataFrame(results)
        self._format_table("Exp 4: Memory Compression Trade-off", df)
        
        plt.figure()
        plt.plot(df['Compression Rate (%)'], df['Reasoning F1'], marker='o')
        plt.xlabel('Graph Compression Rate (%)')
        plt.ylabel('Reasoning Quality (F1)')
        plt.title('HTWM Adaptive Forgetting Trade-off')
        plt.grid(True)
        plt.savefig("scratch/compression_tradeoff.png")

    def run_experiment_5(self):
        # Incremental Streaming Update Cost
        htwm = HTWM()
        faiss_rag = FaissRAG()
        bm25 = BM25RAG()
        
        chunk_size = 2000
        results = []
        
        for i in range(5):
            start = i * chunk_size
            end = start + chunk_size
            chunk = self.events[start:end]
            
            def bench(sys_func):
                t0 = time.time()
                sys_func()
                return (time.time() - t0)*1000
                
            htwm_t = bench(lambda: htwm.ingest(chunk, self.entities if i==0 else None))
            # Vector DB incremental update (simulated via faiss.add)
            faiss_t = bench(lambda: faiss_rag.ingest(chunk))
            # BM25 incremental (requires full rebuild in python typically)
            bm25_t = bench(lambda: bm25.ingest(chunk))
            
            results.append({
                "Total Events": end,
                "HTWM Update (ms)": htwm_t,
                "FAISS Update (ms)": faiss_t,
                "BM25 Update (ms)": bm25_t
            })
            
        df = pd.DataFrame(results)
        self._format_table("Exp 5: Incremental Learning Update Latency", df)

    def run_experiment_7(self):
        # Memory Saturation up to limit
        scales = [1000, 10000, 50000, 100000]
        results = []
        
        for s in scales:
            if s > len(self.events): break
            chunk = self.events[:s]
            htwm = HTWM()
            t0 = time.time()
            htwm.ingest(chunk, self.entities)
            build_time = time.time() - t0
            
            # Test context bounded-ness
            entity = next(ev for ev in reversed(chunk) if len(ev['involved'])>0)['involved'][0]
            ctx = htwm.retrieve_context(entity, chunk[-1]['timestamp'])
            
            results.append({
                "Events": s,
                "Build (s)": build_time,
                "Nodes": htwm.graph.graph.number_of_nodes(),
                "Edges": htwm.graph.graph.number_of_edges(),
                "Context Size (chars)": len(ctx)
            })
            
        df = pd.DataFrame(results)
        self._format_table("Exp 7 & 10: Saturation & Complexity", df)

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    suite = PhaseIIExperiments()
    suite.run_experiment_1_and_3()
    suite.run_experiment_4()
    suite.run_experiment_5()
    suite.run_experiment_7()
    print("\n[*] Phase II Validation Complete.")
