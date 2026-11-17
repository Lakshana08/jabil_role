"""Builds a downloadable report (PDF/Excel/CSV) and holds it in a
short-lived, token-keyed store - A2A responses carry a download_url, not
the file itself; /reports/download/{token} streams it back via get().

Reports are kept on local disk, not in process memory: every uvicorn worker
in the container shares the same directory, so a download link works no
matter which worker serves it, and finished files don't sit in RAM. (More
than one CF instance would need a shared store such as Object Store/Redis
behind this same build()/get() interface.)"""

import csv
import io
import json
import os
import re
import secrets
import tempfile
import time

from openpyxl import Workbook
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.config import get_settings

SUPPORTED_FORMATS = {"csv", "xlsx", "pdf"}

_CONTENT_TYPES = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pdf": "application/pdf",
}

# Below this width, ReportLab's Paragraph wrap breaks and blows up doc.build() -
# wide tables are split into column chunks that each stay above this floor.
_MIN_COL_WIDTH = 45

# A PDF of tens of thousands of rows is unreadable and very slow/memory-heavy
# for ReportLab to lay out - larger data belongs in CSV/Excel.
PDF_MAX_ROWS = 2000

_REPORT_DIR = os.path.join(tempfile.gettempdir(), "smart_material_reports")
# secrets.token_urlsafe(16) output - anything else is rejected before it can
# touch the filesystem (no path traversal via the download URL).
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")

_TABLE_STYLE = TableStyle(
    [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0a6ed1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
)


class ReportNotFoundError(Exception):
    pass


class ReportBuilder:
    def __init__(self, directory: str = _REPORT_DIR) -> None:
        self._dir = directory
        os.makedirs(self._dir, exist_ok=True)

    def build(self, rows: list[dict], report_format: str, base_filename: str) -> dict:
        if report_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported report format '{report_format}', expected one of {SUPPORTED_FORMATS}")

        if report_format == "pdf" and len(rows) > PDF_MAX_ROWS:
            file_rows = rows[:PDF_MAX_ROWS]
            content = self._to_pdf(
                file_rows,
                note=f"Showing the first {PDF_MAX_ROWS:,} of {len(rows):,} rows. "
                "Download the CSV or Excel report for the full data.",
            )
        else:
            file_rows = rows
            content = {
                "csv": self._to_csv,
                "xlsx": self._to_xlsx,
                "pdf": self._to_pdf,
            }[report_format](rows)

        token = secrets.token_urlsafe(16)
        filename = f"{base_filename}.{report_format}"
        now = time.time()
        self._evict_expired(now)
        meta = {
            "filename": filename,
            "content_type": _CONTENT_TYPES[report_format],
            "expires_at": now + get_settings().report_download_ttl_seconds,
        }
        # Content first, metadata last (each via atomic rename): a token only
        # becomes downloadable once its file is completely written.
        self._write_atomic(self._content_path(token), content)
        self._write_atomic(self._meta_path(token), json.dumps(meta).encode("utf-8"))

        download_url = f"{get_settings().app_public_url}/reports/download/{token}"
        return {"token": token, "filename": filename, "download_url": download_url, "rows_in_file": len(file_rows)}

    def get(self, token: str) -> tuple[bytes, str, str]:
        not_found = ReportNotFoundError(f"No report found for token '{token}' (expired or never existed)")
        if not _TOKEN_RE.match(token):
            raise not_found
        meta = self._read_meta(token)
        if meta is None or meta["expires_at"] < time.time():
            self._remove(token)
            raise not_found
        try:
            with open(self._content_path(token), "rb") as handle:
                return handle.read(), meta["filename"], meta["content_type"]
        except FileNotFoundError:
            raise not_found from None

    def _evict_expired(self, now: float) -> None:
        """Sweeps tokens whose TTL has passed, whether or not anyone ever
        downloaded them - get() alone only reclaims a token that's looked up
        again after expiry, which never happens for an abandoned report."""
        for name in os.listdir(self._dir):
            if not name.endswith(".json"):
                continue
            token = name[: -len(".json")]
            meta = self._read_meta(token)
            if meta is None or meta["expires_at"] < now:
                self._remove(token)

    def _content_path(self, token: str) -> str:
        return os.path.join(self._dir, f"{token}.bin")

    def _meta_path(self, token: str) -> str:
        return os.path.join(self._dir, f"{token}.json")

    def _read_meta(self, token: str) -> dict | None:
        try:
            with open(self._meta_path(token), encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _remove(self, token: str) -> None:
        # Another worker may be sweeping the same token at the same moment.
        for path in (self._meta_path(token), self._content_path(token)):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    @staticmethod
    def _write_atomic(path: str, content: bytes) -> None:
        tmp_path = f"{path}.{secrets.token_hex(4)}.tmp"
        with open(tmp_path, "wb") as handle:
            handle.write(content)
        os.replace(tmp_path, path)

    @staticmethod
    def _to_csv(rows: list[dict]) -> bytes:
        buffer = io.StringIO()
        if rows:
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return buffer.getvalue().encode("utf-8")

    @staticmethod
    def _to_xlsx(rows: list[dict]) -> bytes:
        workbook = Workbook()
        sheet = workbook.active
        if rows:
            headers = list(rows[0].keys())
            sheet.append(headers)
            for row in rows:
                sheet.append([row.get(h, "") for h in headers])
        buffer = io.BytesIO()
        workbook.save(buffer)
        return buffer.getvalue()

    @staticmethod
    def _to_pdf(rows: list[dict], note: str | None = None) -> bytes:
        # Landscape + even colWidths + wrapped cells - S/4 tables often have
        # 10+ columns, which clips/cuts off with ReportLab's default sizing.
        buffer = io.BytesIO()
        page_size = landscape(A4)
        doc = SimpleDocTemplate(buffer, pagesize=page_size, leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=18)

        cell_style = getSampleStyleSheet()["BodyText"]
        cell_style.fontSize = 7
        cell_style.leading = 9

        flowables = []
        if note:
            flowables.append(Paragraph(note, getSampleStyleSheet()["Italic"]))
            flowables.append(Spacer(1, 8))
        if rows:
            headers = list(rows[0].keys())
            available_width = page_size[0] - doc.leftMargin - doc.rightMargin
            max_cols_per_chunk = max(1, int(available_width // _MIN_COL_WIDTH))

            for chunk_start in range(0, len(headers), max_cols_per_chunk):
                chunk_headers = headers[chunk_start : chunk_start + max_cols_per_chunk]
                col_width = available_width / len(chunk_headers)
                data = [[Paragraph(str(h), cell_style) for h in chunk_headers]]
                for row in rows:
                    data.append([Paragraph(str(row.get(h, "")), cell_style) for h in chunk_headers])
                table = Table(data, colWidths=[col_width] * len(chunk_headers), repeatRows=1)
                table.setStyle(_TABLE_STYLE)
                flowables.append(table)
                if chunk_start + max_cols_per_chunk < len(headers):
                    flowables.append(Spacer(1, 12))
        else:
            table = Table([["No data"]])
            table.setStyle(_TABLE_STYLE)
            flowables.append(table)

        doc.build(flowables)
        return buffer.getvalue()


_builder = ReportBuilder()


def get_report_builder() -> ReportBuilder:
    return _builder