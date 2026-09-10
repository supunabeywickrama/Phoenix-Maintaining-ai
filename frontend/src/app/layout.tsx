import type { Metadata } from "next";
import "./globals.css";
import { ReduxProvider } from "../store/Provider";
import NavBar from "../components/NavBar";

export const metadata: Metadata = {
  title: "Phoenix Maintenance Copilot",
  description:
    "Ask questions about machine problems and get step-by-step fixes from your own manuals.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-slate-950">
        <ReduxProvider>
          <NavBar />
          <main className="flex-1">{children}</main>
        </ReduxProvider>
      </body>
    </html>
  );
}
