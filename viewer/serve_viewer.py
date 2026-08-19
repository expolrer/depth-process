#!/usr/bin/env python3
"""Serve the video viewer with HTTP byte-range support."""

from __future__ import annotations

import argparse
import os
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


RANGE_PATTERN = re.compile(r"bytes=(\d*)-(\d*)$")


class RangeRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    range_to_send: tuple[int, int] | None = None

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        if self.path.endswith("catalog.json") or self.path.endswith("build_report.json"):
            self.send_header("Cache-Control", "no-store")
        elif self.path.endswith(".mp4"):
            self.send_header("Cache-Control", "public, max-age=604800, immutable")
        else:
            self.send_header("Cache-Control", "public, max-age=300")
        super().end_headers()

    def send_head(self):  # noqa: ANN201
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            for index in ("index.html", "index.htm"):
                candidate = path / index
                if candidate.is_file():
                    path = candidate
                    break
            else:
                return super().send_head()
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        try:
            file_handle = path.open("rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        stat = path.stat()
        size = stat.st_size
        start, end = 0, max(0, size - 1)
        range_header = self.headers.get("Range")
        if range_header:
            match = RANGE_PATTERN.fullmatch(range_header.strip())
            if not match:
                file_handle.close()
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return None
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            elif last:
                suffix_length = min(int(last), size)
                start = size - suffix_length
                end = size - 1
            if start >= size or start > end:
                file_handle.close()
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.range_to_send = (start, end)
        else:
            self.send_response(HTTPStatus.OK)
            self.range_to_send = None

        content_type = self.guess_type(str(path))
        if content_type.startswith("text/") or content_type in {
            "application/json",
            "application/javascript",
        }:
            content_type += "; charset=utf-8"
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        self.end_headers()
        file_handle.seek(start)
        return file_handle

    def copyfile(self, source, outputfile) -> None:  # noqa: ANN001
        if self.range_to_send is None:
            return super().copyfile(source, outputfile)
        start, end = self.range_to_send
        remaining = end - start + 1
        while remaining > 0:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            try:
                outputfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break
            remaining -= len(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(args.directory.resolve())
    server = ThreadingHTTPServer((args.host, args.port), RangeRequestHandler)
    print(f"Serving {args.directory.resolve()} at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
