import type { ComponentType } from "react";
import type { LucideProps } from "lucide-react";

type MetricCardProps = {
  label: string;
  value: string;
  trend: string;
  liveLabel: string;
  icon: ComponentType<LucideProps>;
};

export function MetricCard({ label, value, trend, liveLabel, icon: Icon }: MetricCardProps) {
  return (
    <article className="metric-card soft-raised soft-raised-hover">
      <div className="metric-label">
        <span>{label}</span>
        <span className="badge soft-inset good">
          <Icon size={15} strokeWidth={2.4} aria-hidden="true" />
          {liveLabel}
        </span>
      </div>
      <div className="metric-value font-display">{value}</div>
      <p className="metric-trend">{trend}</p>
    </article>
  );
}
