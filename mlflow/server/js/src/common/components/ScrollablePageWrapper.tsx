import { PageWrapper } from '@databricks/design-system';

/**
 * Wraps the page content in the scrollable container so e.g. constrained tables behave correctly.
 */
export const ScrollablePageWrapper = ({ children, className }: { children: React.ReactNode; className?: string }) => {
  return (
    <PageWrapper
      // Subtract header height
      css={{ height: 'calc(100% - 60px)' }}
      className={className}
    >
      {children}
    </PageWrapper>
  );
};
