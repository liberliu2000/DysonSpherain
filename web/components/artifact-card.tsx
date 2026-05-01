import type { ComponentType } from "react";
import type { LucideProps } from "lucide-react";

type ArtifactCardProps = {
  title: string;
  description: string;
  badge: string;
  icon: ComponentType<LucideProps>;
};

export function ArtifactCard({ title, description, badge, icon: Icon }: ArtifactCardProps) {
  return (
    <article className="artifact-card soft-raised soft-raised-hover">
      <div className="route-row">
        <div className="icon-well soft-inset-deep" aria-hidden="true">
          <Icon size={23} strokeWidth={2.2} />
        </div>
        <span className="badge soft-inset good">{badge}</span>
      </div>
      <h3 className="panel-title font-display">{title}</h3>
      <p>{description}</p>
    </article>
  );
}
