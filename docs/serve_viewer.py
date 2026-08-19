#!/usr/bin/env python3
"""Serve RGB-D Depth Lab with HTTP byte-range support for video seeking."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO


class RangeHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self) -> BinaryIO | None:
        path = Path(self.translate_path(self.path))
        range_header = self.headers.get("Range")
        if not range_header or not path.is_file():
            self._range = None
            return super().send_head()

        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Range header")
            return None
        size = path.stat().st_size
        start_text, end_text = match.groups()
        if not start_text:
            suffix = int(end_text)
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(start_text)
            end = min(int(end_text), size - 1) if end_text else size - 1
        if start >= size or start > end:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        handle = path.open("rb")
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        handle.seek(start)
        self._range = (start, end)
        return handle

    def copyfile(self, source: BinaryIO, outputfile: BinaryIO) -> None:
        byte_range = getattr(self, "_range", None)
        if byte_range is None:
            shutil.copyfileobj(source, outputfile)
            return
        remaining = byte_range[1] - byte_range[0] + 1
        while remaining > 0:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    server = ThreadingHTTPServer((args.host, args.port), RangeHandler)
    print(f"RGB-D Depth Lab full video: http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
