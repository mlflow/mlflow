import { PageWrapper } from '@databricks/design-system';

export const ScrollablePageWrapperStyles = { height: 'calc(100% - 60px)' };

/**
 * Wraps the page content in the scrollable container so e.g. constrained tables behave correctly.
 */
export const ScrollablePageWrapper = ({ children }: { children: React.ReactNode }) => {
  return (
    <PageWrapper
      // Subtract header height
      css={ScrollablePageWrapperStyles}
    >
      {children}
    </PageWrapper>
  );
};
