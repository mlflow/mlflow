import { type ReactNode } from 'react';
import { OverviewLayout, AsideSections } from '@databricks/web-shared/utils';

/**
 * A wrapper for the details page layout, conditionally rendering sidebar-enabled layout based on prop.
 */
export const DetailsPageLayout = ({
  children,
  className,
  secondarySections = [],
  usingSidebarLayout,
}: {
  children: ReactNode;
  className?: string;
  secondarySections?: AsideSections;
  usingSidebarLayout?: boolean;
}) => {
  if (usingSidebarLayout) {
    return (
      <div className={className}>
        {/* prettier-ignore */}
        <OverviewLayout
          asideSections={secondarySections}
          isTabLayout
          sidebarSize="lg"
          verticalStackOrder="aside-first"
        >
          {children}
        </OverviewLayout>
      </div>
    );
  }
  return <div className={className}>{children}</div>;
};
