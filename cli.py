import argparse
import json
from pathlib import Path
from store import DocumentStore


def _load_store_if_exists(store: DocumentStore, path: Path) -> None:
    if path.exists():
        store.load(str(path))


def main():
    parser = argparse.ArgumentParser(description="Local LLM Chat Engine CLI")
    sub = parser.add_subparsers(dest="cmd")
    parser.add_argument(
        "--snapshot",
        default="store.pkl",
        help="Path to persisted store snapshot (default: store.pkl)",
    )

    p_ingest = sub.add_parser("ingest", help="Ingest a JSONL or folder of text files")
    p_ingest.add_argument("source", help="file (jsonl) or directory of .txt files")
    p_ingest.add_argument("--save", action="store_true", help="Persist store after ingest")

    p_query = sub.add_parser("query", help="Query the local store")
    p_query.add_argument("query")
    p_query.add_argument("--k", type=int, default=3)

    sub.add_parser("stats", help="Show store stats")

    p_save = sub.add_parser("save", help="Save snapshot to disk")
    p_save.add_argument("--path", default=None, help="Snapshot path override")

    p_load = sub.add_parser("load", help="Load snapshot from disk")
    p_load.add_argument("--path", default=None, help="Snapshot path override")

    args = parser.parse_args()
    store = DocumentStore()
    snapshot_path = Path(args.snapshot)

    if args.cmd in {"query", "stats"}:
        _load_store_if_exists(store, snapshot_path)

    if args.cmd == "ingest":
        path = Path(args.source)
        docs = []
        if path.is_dir():
            for f in path.glob("**/*.txt"):
                docs.append({"text": f.read_text(encoding="utf8"), "id": str(f.relative_to(path))})
        else:
            # assume jsonl
            with open(path, "r", encoding="utf8") as fh:
                for line in fh:
                    docs.append(json.loads(line))
        res = store.ingest(docs)
        if args.save:
            store.save(str(snapshot_path))
            res["snapshot"] = str(snapshot_path)
        print(json.dumps(res, indent=2))
    elif args.cmd == "query":
        res = store.search(args.query, k=args.k)
        print(json.dumps(res, indent=2))
    elif args.cmd == "stats":
        print(json.dumps(store.stats(), indent=2))
    elif args.cmd == "save":
        target = Path(args.path) if args.path else snapshot_path
        store.save(str(target))
        print(json.dumps({"saved": str(target)}, indent=2))
    elif args.cmd == "load":
        target = Path(args.path) if args.path else snapshot_path
        if not target.exists():
            raise SystemExit(f"Snapshot file not found: {target}")
        store.load(str(target))
        print(json.dumps({"loaded": str(target), "stats": store.stats()}, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

