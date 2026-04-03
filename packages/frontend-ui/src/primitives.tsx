import React from "react";

type ButtonVariant = "primary" | "secondary" | "ghost";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
}

export const Button: React.FC<ButtonProps> = ({
  variant = "primary",
  children,
  style,
  ...rest
}) => {
  const base: React.CSSProperties = {
    borderRadius: "var(--r-sm)",
    padding: "8px 16px",
    fontSize: 12,
    cursor: "pointer",
    border: "1px solid transparent",
    background: "transparent",
    color: "inherit",
  };

  let themed: React.CSSProperties = {};
  if (variant === "primary") {
    themed = {
      background: "var(--gold-500)",
      color: "var(--text-on-solid)",
      borderColor: "var(--gold-400)",
    };
  } else if (variant === "secondary") {
    themed = {
      background: "rgba(255,255,255,0.04)",
      color: "var(--text-80)",
      borderColor: "var(--glass-border)",
    };
  } else if (variant === "ghost") {
    themed = {
      background: "transparent",
      color: "var(--text-60)",
      borderColor: "var(--glass-border)",
    };
  }

  return (
    <button style={{ ...base, ...themed, ...style }} {...rest}>
      {children}
    </button>
  );
};

