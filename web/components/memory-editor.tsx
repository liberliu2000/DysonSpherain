"use client";

import { useEffect, useMemo, useState } from "react";
import { Save, Search } from "lucide-react";
import { SoftButton } from "./soft-button";

type MemoryRecord = {
  id: string;
  title: string;
  scope: string;
  updatedAt: string;
  tags: string[];
  content: string;
};

type MemoryEditorProps = {
  labels: {
    content: string;
    empty: string;
    saved: string;
    saveMemory: string;
    searchAria: string;
    searchPlaceholder: string;
    title: string;
  };
  records: MemoryRecord[];
};

export function MemoryEditor({ labels, records: initialRecords }: MemoryEditorProps) {
  const [records, setRecords] = useState(initialRecords);
  const [activeId, setActiveId] = useState(initialRecords[0]?.id ?? "");
  const [query, setQuery] = useState("");
  const [savedNotice, setSavedNotice] = useState("");

  useEffect(() => {
    setRecords(initialRecords);
    setActiveId(initialRecords[0]?.id ?? "");
    setQuery("");
    setSavedNotice("");
  }, [initialRecords]);

  const active = records.find((record) => record.id === activeId) ?? records[0];

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) {
      return records;
    }
    return records.filter((record) =>
      [record.title, record.scope, record.content, record.tags.join(" ")]
        .join(" ")
        .toLowerCase()
        .includes(needle)
    );
  }, [query, records]);

  function updateActive(field: "title" | "content", value: string) {
    if (!active) {
      return;
    }
    setSavedNotice("");
    setRecords((current) =>
      current.map((record) =>
        record.id === active.id
          ? {
              ...record,
              [field]: value,
              updatedAt: labels.saved
            }
          : record
      )
    );
  }

  function saveActive() {
    if (!active) {
      return;
    }
    setSavedNotice(`${labels.saved}: ${active.title}`);
  }

  if (!active) {
    return <p className="panel-copy">{labels.empty}</p>;
  }

  return (
    <div className="memory-editor">
      <div className="memory-list">
        <label className="search-field">
          <Search size={18} strokeWidth={2.3} aria-hidden="true" />
          <span className="sr-only">{labels.searchAria}</span>
          <input className="soft-input compact-input" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={labels.searchPlaceholder} />
        </label>
        <div className="memory-items" role="listbox" aria-label={labels.searchAria}>
          {filtered.map((record) => (
            <button
              className={`memory-item soft-inset ${record.id === active.id ? "selected" : ""}`}
              key={record.id}
              type="button"
              role="option"
              aria-selected={record.id === active.id}
              onClick={() => setActiveId(record.id)}
            >
              <span className="memory-title">{record.title}</span>
              <span className="memory-meta">
                {record.scope} · {record.updatedAt}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="memory-detail soft-inset">
        <label>
          <span className="field-label">{labels.title}</span>
          <input className="soft-input compact-input" value={active.title} onChange={(event) => updateActive("title", event.target.value)} />
        </label>
        <label>
          <span className="field-label">{labels.content}</span>
          <textarea className="soft-input textarea editor-textarea" value={active.content} onChange={(event) => updateActive("content", event.target.value)} />
        </label>
        <div className="tag-row">
          {active.tags.map((tag) => (
            <span className="badge soft-small" key={tag}>
              {tag}
            </span>
          ))}
        </div>
        <div className="editor-actions">
          <SoftButton icon={Save} variant="primary" onClick={saveActive}>
            {labels.saveMemory}
          </SoftButton>
          {savedNotice ? <span className="badge soft-inset good">{savedNotice}</span> : null}
        </div>
      </div>
    </div>
  );
}
