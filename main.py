"""Main entry point for SentinelAI Assistant."""

import sys
from pathlib import Path

from agent.orchestrator import AgentOrchestrator
from rag.knowledge_base import KnowledgeBase
from security.policy import Policy


def main():
    print("=" * 64)
    print("  SentinelAI - Secure Offline Local AI Assistant")
    print("=" * 64)

    policy = Policy()
    allowed_dirs = policy.get_allowed_locations()
    print(f"\n[+] Authorized storage locations: {allowed_dirs}")

    print("\n[+] Discovering and indexing authorized documents...")
    kb = KnowledgeBase(policy=policy)
    num_chunks = kb.build_index()

    print(f"[+] Successfully indexed {num_chunks} chunks across {len(kb.indexed_files)} files:")
    for f in kb.indexed_files:
        print(f"    * {f}")

    orchestrator = AgentOrchestrator(
        policy=policy,
        knowledge_base=kb,
    )

    print("\n" + "-" * 64)
    print("Ready for questions or commands. (Type 'exit' or 'quit' to quit)")
    print("-" * 64)

    while True:
        try:
            question = input("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting SentinelAI.")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        print("\nThinking...")
        response = orchestrator.run(question)

        if response.get("success"):
            print("\nAnswer:")
            print(response.get("answer"))

            sources = response.get("sources", [])
            if sources:
                print("\nSources:")
                for s in sources:
                    src = s.get("source", "")
                    page_str = f", Page {s['page']}" if s.get("page") is not None else ""
                    score_str = f" [Score: {s['score']:.4f}]" if "score" in s else ""
                    print(f"  - {src}{page_str}{score_str}")
        else:
            print("\nNotice:")
            print(response.get("error") or "I don't have enough information in the documents to answer this question.")

        # If not attached to an interactive terminal, process single input and exit
        if not sys.stdin.isatty():
            break


if __name__ == "__main__":
    main()