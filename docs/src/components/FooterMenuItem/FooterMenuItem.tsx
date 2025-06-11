import Link from "@docusaurus/Link";
import { ComponentProps } from "react";
import { cx } from "class-variance-authority";

interface FooterMenuItemProps extends ComponentProps<typeof Link> {
  isDarkMode: boolean;
}

export const FooterMenuItem = ({ className, isDarkMode, ...props }: FooterMenuItemProps) => {
  return (
    <div className="min-w-[120px]">
      <Link 
        {...props} 
        className={cx(
          "text-[15px] font-medium no-underline hover:no-underline transition-opacity hover:opacity-80",
          isDarkMode ? "text-white visited:text-white" : "text-black visited:text-black",
          className
        )} 
      />
    </div>
  );
};