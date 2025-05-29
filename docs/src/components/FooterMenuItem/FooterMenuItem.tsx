import Link from "@docusaurus/Link";
import { ComponentProps } from "react";

import "./FooterMenuItem.module.css";

export const FooterMenuItem = (props: ComponentProps<typeof Link>) => {
  return (
    <div className="min-w-[120px]">
      <Link {...props} />
    </div>
  );
};
