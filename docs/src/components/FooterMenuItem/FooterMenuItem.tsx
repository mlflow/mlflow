import Link from '@docusaurus/Link';
import { ComponentProps, ReactNode } from 'react';
import { cx } from 'class-variance-authority';

interface FooterMenuItemProps extends ComponentProps<typeof Link> {
  isDarkMode: boolean;
  children: ReactNode;
  className?: string;
  href: string;
}

export const FooterMenuItem = ({ className, isDarkMode, children, ...props }: FooterMenuItemProps) => {
  return (
    <div>
      <Link
        {...props}
        className={cx(
          'text-[15px] font-medium no-underline hover:no-underline transition-opacity hover:opacity-80',
          isDarkMode ? 'text-white visited:text-white' : 'text-black visited:text-black',
          className,
        )}
      >
        {children}
      </Link>
    </div>
  );
};
