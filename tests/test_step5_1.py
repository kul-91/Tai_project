import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from security.policy import Policy
from rag.knowledge_base import KnowledgeBase


def main():
    policy = Policy()
    assistant = KnowledgeBase(policy=policy)

    print("Discovering and indexing documents in authorized locations...")
    indexed_chunks = assistant.build_index()
    print(f"Indexed {indexed_chunks} chunks from files: {assistant.indexed_files}")

    if indexed_chunks == 0:
        print("No document chunks were indexed.")
        return

    question = "what is the name of professor of Trustworhty AI course in ABV-IIITM??"
    print(f"\nAsking question: '{question}'")

    response = assistant.ask(question, k=3, threshold=0.35)

    print("\n--- SentinelAI Answer ---")
    print(response["answer"])

    print("\n--- Sources ---")
    for src in response["sources"]:
        print(src)

    print("\nEvidence count:", response["evidence_count"])


if __name__ == "__main__":
    main()
