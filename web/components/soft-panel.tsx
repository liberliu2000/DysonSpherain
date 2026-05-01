import type { ComponentType, ReactNode } from "react";
import type { LucideProps } from "lucide-react";

type SoftPanelProps = {
  title?: string;
  description?: string;
  icon?: ComponentType<LucideProps>;
  children: ReactNode;
  className?: string;
};

export function SoftPanel({ title, description, icon: Icon, children, className = "" }: SoftPanelProps) {
  return (
    <section className={`panel soft-raised soft-raised-hover ${className}`}>
      {(title || description || Icon) && (
        <div className="panel-header">
          <div>
            {title ? <h2 className="panel-title font-display">{title}</h2> : null}
            {description ? <p className="panel-copy">{description}</p> : null}
          </div>
          {Icon ? (
            <div className="icon-well soft-inset-deep" aria-hidden="true">
              <Icon size={24} strokeWidth={2.2} />
            </div>
          ) : null}
        </div>
      )}
      {children}
    </section>
  );
}
