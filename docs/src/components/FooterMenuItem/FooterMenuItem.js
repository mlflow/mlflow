import Link from '@docusaurus/Link';
import { cx } from 'class-variance-authority';
export const FooterMenuItem = ({ className, isDarkMode, children, ...props }) => {
    return (<div>
      <Link {...props} className={cx('text-[15px] font-medium no-underline hover:no-underline transition-opacity hover:opacity-80', isDarkMode ? 'text-white visited:text-white' : 'text-black visited:text-black', className)}>
        {children}
      </Link>
    </div>);
};
