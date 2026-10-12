// The browser only talks to this server. Proxying to core and the monitor (and adding their API keys)
// happens in middleware.ts, at runtime, so neither service needs CORS and no key reaches the browser.

/** @type {import('next').NextConfig} */
export default {
  output: "standalone",
  compress: false, // gzip buffers server-sent events
};
