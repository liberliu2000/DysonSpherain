#!/usr/bin/env python3
"""Send a paper bundle over SMTP without storing credentials in files."""

from __future__ import annotations

import argparse
import email.utils
import getpass
import json
import mimetypes
import smtplib
import ssl
import sys
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path


def _attach_file(message: EmailMessage, path: Path) -> None:
    ctype, encoding = mimetypes.guess_type(path.name)
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"
    maintype, subtype = ctype.split("/", 1)
    message.add_attachment(
        path.read_bytes(),
        maintype=maintype,
        subtype=subtype,
        filename=path.name,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ssl", action="store_true")
    parser.add_argument("--username", required=True)
    parser.add_argument("--from-email", required=True)
    parser.add_argument("--from-name", required=True)
    parser.add_argument("--to", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--attach", action="append", default=[])
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    password = sys.stdin.readline().rstrip("\n")
    if not password:
        password = getpass.getpass("SMTP password: ")
    if not password:
        raise RuntimeError("SMTP password was not provided on stdin")

    message = EmailMessage()
    message["From"] = email.utils.formataddr((args.from_name, args.from_email))
    message["To"] = args.to
    message["Subject"] = args.subject
    message["Date"] = email.utils.formatdate(localtime=True)
    message["Message-ID"] = email.utils.make_msgid(domain=args.from_email.split("@")[-1])
    message.set_content(args.body)

    attachments = [Path(p).resolve() for p in args.attach]
    for attachment in attachments:
        _attach_file(message, attachment)

    if args.ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(args.host, args.port, context=context, timeout=60) as smtp:
            smtp.login(args.username, password)
            refused = smtp.send_message(message)
    else:
        with smtplib.SMTP(args.host, args.port, timeout=60) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(args.username, password)
            refused = smtp.send_message(message)

    log = {
        "status": "sent",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "port": args.port,
        "ssl": args.ssl,
        "from": args.from_email,
        "to": args.to,
        "subject": args.subject,
        "attachments": [
            {"path": str(p), "bytes": p.stat().st_size}
            for p in attachments
        ],
        "message_id": message["Message-ID"],
        "refused": refused,
    }
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(log, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
