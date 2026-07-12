"""Knowledge-base ingestion CLI.

    python -m scripts.ingest_kb --tickers NVDA MSFT --limit 3
    python -m scripts.ingest_kb                       # every individual stock in the book
    python -m scripts.ingest_kb --tickers NVDA --news # also archive current headlines

Requires SEC_USER_AGENT in .env (EDGAR rejects anonymous requests).
"""

import argparse
import json

from app.knowledge.ingestion import archive_news, ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest SEC filings into the knowledge base")
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="tickers to ingest (default: every individual stock in the book)")
    parser.add_argument("--limit", type=int, default=3, help="filings per ticker (default 3)")
    parser.add_argument("--forms", nargs="*", default=["10-K", "10-Q"],
                        help="form types (default: 10-K 10-Q)")
    parser.add_argument("--news", action="store_true", help="also archive current headlines")
    args = parser.parse_args()

    summary = ingest(tickers=args.tickers, limit=args.limit, forms=tuple(args.forms))
    print(json.dumps(summary, indent=2))
    if args.news and args.tickers:
        added = archive_news(args.tickers)
        print(f"news_archive: +{added} headlines")


if __name__ == "__main__":
    main()
