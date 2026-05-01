import type { Locale } from "@/lib/dashboard-data";

type TrendPoint = {
  label: string;
  saved: number;
};

type TokenTrendChartProps = {
  ariaLabel: string;
  data: TrendPoint[];
  locale: Locale;
  peakLabel: string;
  subtitle: string;
  title: string;
};

export function TokenTrendChart({ ariaLabel, data, locale, peakLabel, subtitle, title }: TokenTrendChartProps) {
  const compact = new Intl.NumberFormat(locale === "zh" ? "zh-CN" : "en", {
    notation: "compact",
    maximumFractionDigits: 1
  });
  const max = Math.max(...data.map((point) => point.saved));
  const points = data
    .map((point, index) => {
      const x = data.length === 1 ? 0 : (index / (data.length - 1)) * 100;
      const y = 100 - (point.saved / max) * 86 - 7;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="trend-chart soft-inset-deep">
      <div className="chart-header">
        <div>
          <strong className="font-display">{title}</strong>
          <p>{subtitle}</p>
        </div>
        <span className="badge soft-small good">
          {compact.format(max)} {peakLabel}
        </span>
      </div>
      <svg className="chart-svg" viewBox="0 0 100 100" role="img" aria-label={ariaLabel} preserveAspectRatio="none">
        <defs>
          <linearGradient id="tokenLine" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="#6C63FF" />
            <stop offset="100%" stopColor="#8B84FF" />
          </linearGradient>
        </defs>
        <polyline points={points} fill="none" stroke="url(#tokenLine)" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
        {data.map((point, index) => {
          const x = data.length === 1 ? 0 : (index / (data.length - 1)) * 100;
          const y = 100 - (point.saved / max) * 86 - 7;
          return <circle key={point.label} cx={x} cy={y} r="2.4" fill="#6C63FF" />;
        })}
      </svg>
      <div className="chart-axis" aria-hidden="true">
        {data.map((point) => (
          <span key={point.label}>{point.label}</span>
        ))}
      </div>
    </div>
  );
}
