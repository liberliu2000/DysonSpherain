import type { ButtonHTMLAttributes, ComponentType } from "react";
import type { LucideProps } from "lucide-react";

type SoftButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  icon?: ComponentType<LucideProps>;
  variant?: "primary" | "secondary" | "icon";
};

export function SoftButton({ children, className = "", icon: Icon, variant = "secondary", ...props }: SoftButtonProps) {
  return (
    <button className={`soft-button ${variant} ${className}`} type="button" {...props}>
      {Icon ? <Icon size={20} strokeWidth={2.3} aria-hidden="true" /> : null}
      {variant === "icon" ? <span className="sr-only">{children}</span> : children}
    </button>
  );
}
