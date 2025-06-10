import Link from "@docusaurus/Link";
import { ComponentProps } from "react";

import styles from "./FooterMenuItem.module.css";
import { cx } from "class-variance-authority";

export const FooterMenuItem = ({ className, ...props }: ComponentProps<typeof Link>) => {
  return (
    <div className="min-w-[120px]">
      <Link {...props} className={cx(className, styles.link)} />
    </div>
  );
};
