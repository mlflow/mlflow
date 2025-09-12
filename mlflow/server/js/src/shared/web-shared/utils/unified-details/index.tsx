import { GenericSkeleton, ParagraphSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { ReactNode } from 'react';
import { useRef } from 'react';
import { FormattedMessage } from 'react-intl';
import useResponsiveContainer from './useResponsiveContainer';

export interface AsideSectionProps {
  id: string;
  title?: ReactNode;
  content: ReactNode;
  isTitleLoading?: boolean;
}

export type MaybeAsideSection = AsideSectionProps | null;
export type AsideSections = Array<MaybeAsideSection>;

const SIDEBAR_WIDTHS = {
  sm: 316,
  lg: 480,
} as const;
const VERTICAL_MARGIN_PX = 16;
const DEFAULT_MAX_WIDTH = 450;

export const OverviewLayout = ({
  isLoading,
  asideSections,
  children,
  isTabLayout = true,
  sidebarSize = 'sm',
  verticalStackOrder,
}: {
  isLoading?: boolean;
  asideSections: AsideSections;
  children: ReactNode;
  isTabLayout?: boolean;
  sidebarSize?: 'sm' | 'lg';
  verticalStackOrder?: 'main-first' | 'aside-first';
}) => {
  const { theme } = useDesignSystemTheme();
  const containerRef = useRef<HTMLDivElement>(null);

  const stackVertically = useResponsiveContainer(containerRef, { small: theme.responsive.breakpoints.lg }) === 'small';

  // Determine vertical stack order, i.e. should the main content be on top or bottom
  const verticalDisplayPrimaryContentOnTop = verticalStackOrder === 'main-first';

  const totalSidebarWidth = SIDEBAR_WIDTHS[sidebarSize];
  const innerSidebarWidth = totalSidebarWidth - VERTICAL_MARGIN_PX;

  const secondaryStackedStyles = stackVertically
    ? verticalDisplayPrimaryContentOnTop
      ? { width: '100%' }
      : { borderBottom: `1px solid ${theme.colors.border}`, width: '100%' }
    : verticalDisplayPrimaryContentOnTop
    ? {
        width: innerSidebarWidth,
      }
    : {
        paddingBottom: theme.spacing.sm,
        width: innerSidebarWidth,
      };

  return (
    <div
      data-testid="entity-overview-container"
      ref={containerRef}
      css={{
        display: 'flex',
        flexDirection: stackVertically ? (verticalDisplayPrimaryContentOnTop ? 'column' : 'column-reverse') : 'row',
        gap: theme.spacing.lg,
      }}
    >
      <div
        css={{
          display: 'flex',
          flexGrow: 1,
          flexDirection: 'column',
          gap: theme.spacing.md,
          width: stackVertically ? '100%' : `calc(100% - ${totalSidebarWidth}px)`,
        }}
      >
        {isLoading ? <GenericSkeleton /> : children}
      </div>
      <div
        style={{
          display: 'flex',
          ...(isTabLayout && { marginTop: -theme.spacing.md }), // remove the gap between tab list and sidebar content
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.lg,
            ...secondaryStackedStyles,
          }}
        >
          {isLoading && <GenericSkeleton />}
          {!isLoading && <SidebarWrapper secondarySections={asideSections} />}
        </div>
      </div>
    </div>
  );
};

const SidebarWrapper = ({ secondarySections }: { secondarySections: AsideSections }) => {
  return (
    <div>
      {secondarySections
        .filter((section) => section !== null)
        .filter((section) => section?.content !== null)
        .map(({ title, isTitleLoading, content, id }, index) => (
          <AsideSection title={title} isTitleLoading={isTitleLoading} content={content} key={id} index={index} />
        ))}
    </div>
  );
};

export const AsideSectionTitle = ({ children }: { children: ReactNode }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Typography.Title
      level={4}
      style={{
        whiteSpace: 'nowrap',
        marginRight: theme.spacing.lg,
        marginTop: 0,
      }}
    >
      {children}
    </Typography.Title>
  );
};

const AsideSection = ({
  title,
  content,
  index,
  isTitleLoading = false,
}: Omit<AsideSectionProps, 'id'> & {
  index: number;
}) => {
  const { theme } = useDesignSystemTheme();

  const titleComponent = isTitleLoading ? (
    <ParagraphSkeleton
      label={
        <FormattedMessage
          defaultMessage="Section title loading"
          description="Loading skeleton label for overview page section title in Catalog Explorer"
        />
      }
    />
  ) : title ? (
    <AsideSectionTitle>{title}</AsideSectionTitle>
  ) : null;

  const compactStyles = { padding: `${theme.spacing.md}px 0 ${theme.spacing.md}px 0` };

  return (
    <div
      css={{
        ...compactStyles,
        ...(index === 0 ? {} : { borderTop: `1px solid ${theme.colors.border}` }),
      }}
    >
      {titleComponent}
      {content}
    </div>
  );
};

export const KeyValueProperty = ({
  keyValue,
  value,
  maxWidth,
}: {
  keyValue: string;
  value: React.ReactNode;
  maxWidth?: number | string;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      data-testid={`key-value-${keyValue}`}
      css={{
        display: 'flex',
        alignItems: 'center',
        '&:has(+ div)': {
          marginBottom: theme.spacing.xs,
        },
        maxWidth: maxWidth ?? DEFAULT_MAX_WIDTH,
        wordBreak: 'break-word',
        lineHeight: theme.typography.lineHeightLg,
      }}
    >
      <div
        css={{
          color: theme.colors.textSecondary,
          flex: 0.5,
          alignSelf: 'start',
        }}
      >
        {keyValue}
      </div>
      <div
        css={{
          flex: 1,
          alignSelf: 'start',
          overflow: 'hidden',
        }}
      >
        {value}
      </div>
    </div>
  );
};

export const NoneCell = () => {
  return (
    <Typography.Text color="secondary">
      <FormattedMessage defaultMessage="None" description="Cell value when there's no content" />
    </Typography.Text>
  );
};
