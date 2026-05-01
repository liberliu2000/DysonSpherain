import type { ComponentType } from "react";
import type { LucideProps } from "lucide-react";

type TimelineItemProps = {
  title: string;
  detail: string;
  icon: ComponentType<LucideProps>;
};

export function TimelineItem({ title, detail, icon: Icon }: TimelineItemProps) {
  return (
    <article className="timeline-item soft-inset">
      <div className="timeline-row">
        <div>
          <strong>{title}</strong>
          <p>{detail}</p>
        </div>
        <span className="icon-well soft-small" aria-hidden="true">
          <Icon size={21} strokeWidth={2.3} />
        </span>
      </div>
    </article>
  );
}
