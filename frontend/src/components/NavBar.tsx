"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Flame, MessageSquare, BookOpen, Cog } from "lucide-react";

const NAV = [
  { name: "Ask", href: "/", icon: MessageSquare },
  { name: "Manuals", href: "/manuals", icon: BookOpen },
  { name: "Machines", href: "/machines", icon: Cog },
];

export default function NavBar() {
  const pathname = usePathname();

  return (
    <nav className="sticky top-0 z-50 flex items-center justify-between border-b border-slate-800 bg-slate-900/90 px-4 py-3 backdrop-blur md:px-8">
      <Link href="/" className="flex items-center gap-3 transition-opacity hover:opacity-80">
        <div className="rounded-lg bg-orange-600 p-2 shadow-lg shadow-orange-900/40">
          <Flame className="text-white" size={20} />
        </div>
        <div className="leading-tight">
          <div className="text-lg font-black tracking-tight text-white">PHOENIX</div>
          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-orange-400/80">
            Maintenance Copilot
          </div>
        </div>
      </Link>

      <div className="flex items-center gap-1 md:gap-2">
        {NAV.map((item) => {
          const active = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-bold transition-colors md:px-4 ${
                active
                  ? "border border-orange-500/30 bg-orange-500/10 text-orange-400"
                  : "border border-transparent text-slate-400 hover:bg-slate-800 hover:text-slate-100"
              }`}
            >
              <Icon size={17} />
              <span className="hidden sm:inline">{item.name}</span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
