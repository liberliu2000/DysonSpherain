"use client";

import { Languages, Menu, Network, X } from "lucide-react";
import { useState } from "react";
import type { Locale } from "@/lib/dashboard-data";
import { SoftButton } from "./soft-button";

type TopNavProps = {
  labels: {
    brandSubtitle: string;
    languageLabel: string;
    toggleNavigation: string;
    links: Array<{ href: string; label: string }>;
  };
  locale: Locale;
  onToggleLocale: () => void;
};

export function TopNav({ labels, locale, onToggleLocale }: TopNavProps) {
  const [open, setOpen] = useState(false);

  return (
    <header className="topbar soft-raised">
      <a href="#top" className="brand" aria-label="DysonSpherain console home">
        <span className="brand-mark soft-inset-deep" aria-hidden="true">
          <Network size={24} strokeWidth={2.3} />
        </span>
        <span>
          <strong className="font-display">DysonSpherain</strong>
          <small>{labels.brandSubtitle}</small>
        </span>
      </a>
      <nav id="primary-navigation" className="nav-list" data-open={open}>
        {labels.links.map((link) => (
          <a className="nav-link" href={link.href} key={link.href}>
            {link.label}
          </a>
        ))}
      </nav>
      <div className="topbar-actions">
        <SoftButton icon={Languages} variant="secondary" onClick={onToggleLocale} aria-label={labels.languageLabel}>
          {locale === "en" ? "中文" : "EN"}
        </SoftButton>
        <SoftButton
          className="mobile-toggle"
          icon={open ? X : Menu}
          variant="icon"
          aria-expanded={open}
          aria-controls="primary-navigation"
          onClick={() => setOpen((value) => !value)}
        >
          {labels.toggleNavigation}
        </SoftButton>
      </div>
    </header>
  );
}
