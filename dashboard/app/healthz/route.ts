export const dynamic = "force-dynamic";

// Open to the container health check (the rest of the site can sit behind DASHBOARD_PASSWORD).
export function GET() {
  return Response.json({ status: "ok" });
}
