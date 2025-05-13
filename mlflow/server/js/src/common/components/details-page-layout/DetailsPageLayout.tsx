import { type ReactNode } from 'react';
import { OverviewLayout, SecondarySections } from '@databricks/web-shared/utils';

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
  secondarySections?: SecondarySections;
  usingSidebarLayout?: boolean;
}) => {
  if (usingSidebarLayout) {
    return (
      <div className={className}>
        {/* prettier-ignore */}
        <OverviewLayout
          secondarySections={secondarySections}
          isTabLayout
          sidebarSize="lg"
          verticalStackOrder="secondary-first"
        >
          {children}
        </OverviewLayout>
      </div>
    );
  }
  return <div className={className}>{children}</div>;
};
