"""Fixtures live here; shared helpers are in core_helpers.py (a unique module name, so the
service test suites can run together)."""

import asyncio
import sys

# psycopg's async API cannot run on Windows' default ProactorEventLoop (the Postgres tests would hang)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
