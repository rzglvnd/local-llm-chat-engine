import argparse
import json
from pathlib import Path
from .store import DocumentStore


def main():
    parser = argparse.ArgumentParser(description="Local LLM Chat Engine CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Ingest a JSONL or folder of text files")
    p_ingest.add_argument("source", help="file (jsonl) or directory of .txt files")

    p_query = sub.add_parser("query", help="Query the local store")
    p_query.add_argument("query")
    p_query.add_argument("--k", type=int, default=3)

    args = parser.parse_args()
    store = DocumentStore()

    if args.cmd == "ingest":
        path = Path(args.source)
        docs = []
        if path.is_dir():
            for f in path.glob("**/*.txt"):
                docs.append({"text": f.read_text(), "id": str(f.relative_to(path))})
        else:
            # assume jsonl
            with open(path, "r", encoding="utf8") as fh:
                for line in fh:
                    docs.append(json.loads(line))
        res = store.ingest(docs)
        print(json.dumps(res, indent=2))
    elif args.cmd == "query":
        res = store.search(args.query, k=args.k)
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
