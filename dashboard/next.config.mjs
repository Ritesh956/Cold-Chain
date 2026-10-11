// The browser only talks to this server; it proxies to core and the monitor, so neither needs CORS.
// Rewrite targets are fixed at BUILD time (CORE_URL, MONITOR_URL).
const core = process.env.CORE_URL ?? "http://127.0.0.1:8000";
const monitor = process.env.MONITOR_URL ?? "http://127.0.0.1:8090";

/** @type {import('next').NextConfig} */
export default {
  output: "standalone",
  compress: false, // gzip buffers server-sent events
  async rewrites() {
    return [
      { source: "/api/core/:path*", destination: `${core}/:path*` },
      { source: "/api/monitor/:path*", destination: `${monitor}/:path*` },
    ];
  },
};
