import { ParagraphSkeleton, useDesignSystemTheme, GenericSkeleton, TitleSkeleton } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

const SkeletonLines = ({ count }: { count: number }) => (
  <>
    {new Array(count).fill('').map((_, i) => (
      <ParagraphSkeleton
        key={i}
        seed={i.toString()}
        label={
          i === 0 ? (
            <FormattedMessage
              defaultMessage="Artifact loading"
              description="Run page > artifact view > loading skeleton label"
            />
          ) : undefined
        }
      />
    ))}
  </>
);

/**
 * Loading state for the artifact browser with sidepane and content area
 */
export const ArtifactViewBrowserSkeleton = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', flex: 1 }}>
      <div css={{ flex: 1 }}>
        <div css={{ margin: theme.spacing.sm }}>
          <SkeletonLines count={9} />
        </div>
      </div>
      <div css={{ flex: 3, borderLeft: `1px solid ${theme.colors.border}` }}>
        <div css={{ margin: theme.spacing.sm }}>
          <TitleSkeleton css={{ marginBottom: theme.spacing.md }} />
          <SkeletonLines count={3} />

          <div css={{ width: '75%', marginTop: theme.spacing.md }}>
            <SkeletonLines count={3} />
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Generic loading state for the artifact viewer
 */
export const ArtifactViewSkeleton = (divProps: React.HTMLAttributes<HTMLDivElement>) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div data-testid="mlflow-artifact-view-skeleton" css={{ margin: theme.spacing.md }} {...divProps}>
      <SkeletonLines count={9} />
    </div>
  );
};
