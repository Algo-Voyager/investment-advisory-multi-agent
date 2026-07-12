"""SEC EDGAR adapter — official filings (10-K / 10-Q / 8-K). No key, BUT:

EDGAR **rejects any request without a descriptive User-Agent containing contact
info** — that's what `settings.SEC_USER_AGENT` ("Name email@example.com") is for.
EDGAR also enforces ~10 requests/second; we self-limit well under that.

This adapter doesn't fit the MarketDataAdapter interface (filings, not prices),
so it stands alone. Phase 5's RAG ingestion is its main consumer.
"""

import requests

from app.config import settings
from app.errors.exceptions import RateLimitError, ToolError
from app.logging import get_logger
from app.tools.decorators import cached, rate_limited, retry

log = get_logger(__name__)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"


class SECEdgarAdapter:
    name = "sec_edgar"

    def available(self) -> bool:
        return bool(settings.SEC_USER_AGENT)

    def _headers(self) -> dict:
        if not settings.SEC_USER_AGENT:
            raise ToolError(f"[{self.name}] SEC_USER_AGENT is not set — EDGAR requires "
                            '"Your Name your-email@example.com" in .env')
        return {"User-Agent": settings.SEC_USER_AGENT,
                "Accept-Encoding": "gzip, deflate"}

    @rate_limited(calls_per_minute=300)  # EDGAR allows ~10/s; we stay at 5/s
    def _get(self, url: str) -> requests.Response:
        resp = requests.get(url, headers=self._headers(), timeout=20)
        if resp.status_code == 429:
            raise RateLimitError(f"[{self.name}] rate limited (429)")
        if resp.status_code == 403:
            raise ToolError(f"[{self.name}] 403 — is SEC_USER_AGENT a real 'Name email' string?")
        resp.raise_for_status()
        return resp

    @cached(ttl_seconds=86400)  # the ticker→CIK map changes rarely
    def _cik_for(self, ticker: str) -> int:
        data = self._get(TICKER_MAP_URL).json()
        for entry in data.values():
            if entry["ticker"].upper() == ticker.upper():
                return int(entry["cik_str"])
        raise ToolError(f"[{self.name}] no CIK found for '{ticker}' — "
                        f"ETFs/funds don't file 10-K/10-Q")

    @retry(max_attempts=3)
    def get_recent_filings(self, ticker: str, forms: tuple = ("10-K", "10-Q", "8-K"),
                           limit: int = 5) -> dict:
        """List a company's most recent filings of the given form types."""
        cik = self._cik_for(ticker)
        recent = self._get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        filings = []
        for i, form in enumerate(recent["form"]):
            if form not in forms:
                continue
            accession = recent["accessionNumber"][i].replace("-", "")
            filings.append({
                "ticker": ticker,
                "form": form,
                "filing_date": recent["filingDate"][i],
                "accession": recent["accessionNumber"][i],
                "primary_doc": recent["primaryDocument"][i],
                "url": ARCHIVES_URL.format(cik=cik, accession=accession,
                                           doc=recent["primaryDocument"][i]),
            })
            if len(filings) >= limit:
                break
        return {"ticker": ticker, "cik": cik, "filings": filings, "source": self.name}

    @retry(max_attempts=3)
    def get_filing_document(self, url: str) -> str:
        """Fetch a filing's primary document (HTML — Phase 5 strips and chunks it)."""
        return self._get(url).text
