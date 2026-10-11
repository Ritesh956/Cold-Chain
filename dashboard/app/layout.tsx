import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ColdChain control tower",
  description: "Live freezer zones, alerts and model drift for a simulated cold store (synthetic data).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
