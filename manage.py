"""Small operational helpers.

    python manage.py hash-password              # prompts, prints a hash
    python manage.py check-config               # what is configured, what is not
    python manage.py rebuild-index [--force]    # rebuild the vector index offline
    python manage.py ask "your question"        # query the pipeline from the CLI
"""

from __future__ import annotations

import argparse
import getpass
import sys

from dotenv import load_dotenv

load_dotenv()


def cmd_hash_password(_args) -> int:
    """Generate a password hash for the Supabase ``admin_users.password_hash`` column.

    The original app compared the submitted password to that column directly,
    so it held plaintext. Logging in with the old password still works once and
    transparently upgrades the row, but generating a fresh hash here and pasting
    it in is the clean path.
    """
    from rngai.security import hash_password

    password = getpass.getpass("New admin password: ")
    if len(password) < 10:
        print("Refusing: use at least 10 characters.", file=sys.stderr)
        return 1
    if password != getpass.getpass("Repeat: "):
        print("Passwords did not match.", file=sys.stderr)
        return 1

    print("\nStore this in admin_users.password_hash:\n")
    print(hash_password(password))
    return 0


def cmd_check_config(_args) -> int:
    from rngai.config import config

    def mark(ok: bool) -> str:
        return "OK     " if ok else "MISSING"

    print(f"{mark(bool(config.secret_key) and not config.ephemeral_secret)} SECRET_KEY")
    print(f"{mark(config.has_llm)} NVIDIA_API_KEY ({len(config.nvidia_api_keys)} key(s))")
    print(f"{mark(bool(config.supabase_url and config.supabase_key))} SUPABASE_URL / SUPABASE_KEY")
    print(f"{mark(bool(config.groq_api_key))} GROQ_API_KEY (voice input)")
    print()
    print(f"  chat model      : {config.chat_model}")
    print(f"  embedding model : {config.embedding_model}")
    print(f"  data directory  : {config.data_dir}")
    print(f"  cache directory : {config.cache_dir}")
    if config.warnings:
        print("\nWarnings:")
        for warning in config.warnings:
            print(f"  - {warning}")
    return 0


def _services():
    from rngai.webapp import services

    return services


def cmd_rebuild_index(args) -> int:
    services = _services()
    ok = services.knowledge.build(force=args.force)
    stats = services.knowledge.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0 if ok else 1


def cmd_ask(args) -> int:
    services = _services()
    if not services.knowledge.build():
        print("Knowledge base failed to build.", file=sys.stderr)
        return 1

    print()
    for event in services.chat.stream(args.question, session_id="cli", debug=True):
        if event.get("token"):
            sys.stdout.write(event["token"])
            sys.stdout.flush()
        if event.get("done"):
            debug = event.get("debug") or {}
            print("\n" + "-" * 60)
            print(
                f"cached={event['cached']} elapsed={event['elapsed_ms']}ms "
                f"in={event['input_tokens']} out={event['output_tokens']}"
            )
            for source in debug.get("sources", []):
                print(f"  [{source['score']:.3f}] {source['source']} :: {source['heading'][:70]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hash-password", help="generate an admin password hash")
    sub.add_parser("check-config", help="report which credentials are configured")

    rebuild = sub.add_parser("rebuild-index", help="rebuild the vector index")
    rebuild.add_argument("--force", action="store_true", help="ignore all caches")

    ask = sub.add_parser("ask", help="run one question through the pipeline")
    ask.add_argument("question")

    args = parser.parse_args()
    return {
        "hash-password": cmd_hash_password,
        "check-config": cmd_check_config,
        "rebuild-index": cmd_rebuild_index,
        "ask": cmd_ask,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
