import argparse
import os
import sys

import config
from ingest import ingest
from query import ask

os.environ["CHROMA_TELEMETRY"] = "false"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video RAG - ask questions about a YouTube video"
    )
    parser.add_argument(
        "--url",
        required=True,
        help="YouTube video URL to analyse",
    )
    args = parser.parse_args()

    print("\n🎬 Video RAG - AI That Sees YouTube\n")
    print("Step 1/2 - Ingesting video (this runs once per video)\n")
    client, collection = ingest(args.url)

    print("\nStep 2/2 - Ready to answer questions\n")
    print("-" * 50)
    print("Type your question and press Enter.")
    print("Type 'quit' or press Ctrl+C to exit.")
    print("-" * 50)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nBye!")
            sys.exit(0)

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            sys.exit(0)

        print("\nAI: ", end="", flush=True)
        answer = ask(client, collection, question)
        print(answer)


if __name__ == "__main__":
    main()
