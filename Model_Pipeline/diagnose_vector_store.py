from src.retrieval.advanced_retriever import AdvancedRetriever
import re

def highlight(text):
    return text.replace("research", "\033[93mresearch\033[0m")  # highlight research in yellow

print("\n" + "="*80)
print("VECTOR STORE DIAGNOSTIC SUITE — FOCUSED ON 'How does GitLab conduct research?'")
print("="*80)

print("\nInitializing retriever...")
retriever = AdvancedRetriever()
print("✅ Retriever loaded\n")

# -------------------------
# Helper: Pretty print results
# -------------------------
def print_results(results, title=""):
    print("\n" + "-"*80)
    print(title)
    print("-"*80)
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        doc = r.get("document", "")[:300].strip().replace("\n", " ")
        doc = highlight(doc)

        print(f"\n{i}. Title: {meta.get('title', 'Unknown')}")
        print(f"   Rank Score: {r.get('rerank_score', 0):.4f}")
        print(f"   Dense Similarity: {r.get('dense_score', 0):.4f}")
        print(f"   Metadata: {meta}")
        print(f"   Preview: {doc if doc else '[EMPTY DOCUMENT]'}")
        print("-")

# -------------------------
# TEST 1 — Simple meeting query
# -------------------------
print("\n" + "="*80)
print("TEST 1: Searching for 'CI/CD UX meeting'")
print("="*80)

t1 = retriever.retrieve("CI/CD UX meeting", k=10)
print_results(t1, "Top 10 for 'CI/CD UX meeting'")

# -------------------------
# TEST 2 — Basic research keyword
# -------------------------
print("\n" + "="*80)
print("TEST 2: Searching for 'user research'")
print("="*80)

t2 = retriever.retrieve("user research", k=10)
print_results(t2, "Top 10 for 'user research'")

# -------------------------
# TEST 3 — The broken query
# -------------------------
print("\n" + "="*80)
print("TEST 3: Searching for 'how does gitlab conduct research'")
print("="*80)

t3 = retriever.retrieve("how does gitlab conduct research", k=10)
print_results(t3, "Top 10 for 'how does gitlab conduct research'")

# -------------------------
# Extra Debug — List *all* research-related chunks in the vector store
# -------------------------
print("\n" + "="*80)
print("TEST 4: Manually scan for 'research' in ALL vector store chunks")
print("="*80)

query_vector = retriever.encoder.encode("research", normalize_embeddings=True)
all_hits = retriever.vector_store.query(query_vector, n_results=50)

print("\nScanning top 50 chunks for explicit occurrences of 'research'...")
print("-"*80)

for i in range(len(all_hits["documents"][0])):
    doc = all_hits["documents"][0][i]
    meta = all_hits["metadatas"][0][i]
    if re.search(r"research", doc, re.IGNORECASE):
        print(f"✔ FOUND RESEARCH in chunk {i+1}")
        print(f"  Title: {meta.get('title', 'Unknown')}")
        print(f"  Preview: {highlight(doc[:200]).replace('\n',' ')}")
        print("-"*40)

print("\n" + "="*80)
print("DIAGNOSTICS COMPLETE")
print("="*80 + "\n")
