type RouteCardProps = {
  name: string;
  status: string;
  detail: string;
  progress: number;
};

export function RouteCard({ name, status, detail, progress }: RouteCardProps) {
  return (
    <article className="route-item soft-inset">
      <div className="route-row">
        <strong>{name}</strong>
        <span className="badge soft-small">{status}</span>
      </div>
      <p>{detail}</p>
      <div className="progress-track" aria-label={`${name} confidence ${progress}%`}>
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
    </article>
  );
}
