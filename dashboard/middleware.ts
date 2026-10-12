import { NextResponse, type NextRequest } from "next/server";

// Runs on every request except static assets and /healthz.
//  1. If DASHBOARD_PASSWORD is set, the whole dashboard sits behind HTTP Basic auth (any user name).
//  2. /api/core/* and /api/monitor/* are proxied to the services with their API keys added HERE, on the
//     server, so no service key ever reaches the browser. Targets and keys are read at runtime.
// This is a single shared password and shared service keys, not per-user authentication; use TLS in front.

export const config = { matcher: ["/((?!_next/static|_next/image|favicon.ico|healthz).*)"] };

function safeEqual(a: string, b: string): boolean {
  const x = new TextEncoder().encode(a);
  const y = new TextEncoder().encode(b);
  let diff = x.length ^ y.length;
  for (let i = 0; i < Math.max(x.length, y.length); i++) diff |= (x[i] ?? 0) ^ (y[i] ?? 0);
  return diff === 0;
}

function passwordOk(req: NextRequest, password: string): boolean {
  const header = req.headers.get("authorization") ?? "";
  const [scheme, encoded] = header.split(" ");
  if (scheme?.toLowerCase() !== "basic" || !encoded) return false;
  try {
    const decoded = atob(encoded);
    return safeEqual(decoded.slice(decoded.indexOf(":") + 1), password);
  } catch {
    return false;
  }
}

const UPSTREAMS = [
  { prefix: "/api/core", url: () => process.env.CORE_URL ?? "http://127.0.0.1:8000", key: () => process.env.CORE_API_KEY },
  {
    prefix: "/api/monitor",
    url: () => process.env.MONITOR_URL ?? "http://127.0.0.1:8090",
    key: () => process.env.MONITOR_API_KEY,
  },
];

export function middleware(req: NextRequest) {
  const password = process.env.DASHBOARD_PASSWORD;
  if (password && !passwordOk(req, password)) {
    return new NextResponse("Authentication required", {
      status: 401,
      headers: { "WWW-Authenticate": 'Basic realm="ColdChain dashboard", charset="UTF-8"' },
    });
  }

  const { pathname, search } = req.nextUrl;
  for (const u of UPSTREAMS) {
    if (pathname === u.prefix || pathname.startsWith(u.prefix + "/")) {
      const headers = new Headers(req.headers);
      headers.delete("authorization"); // the dashboard password is not for the services
      const key = u.key();
      if (key) headers.set("x-api-key", key);
      const target = new URL(u.url().replace(/\/$/, "") + pathname.slice(u.prefix.length) + search);
      return NextResponse.rewrite(target, { request: { headers } });
    }
  }
  return NextResponse.next();
}
