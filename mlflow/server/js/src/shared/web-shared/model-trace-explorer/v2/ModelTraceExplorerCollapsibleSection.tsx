import { useState } from 'react';

import { Button, ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';

export const ModelTraceExplorerCollapsibleSection = ({
  sectionKey,
  title,
  children,
  withBorder = false,
  isExceptionSection = false,
  className,
  headerPadding,
  contentPadding,
  initialExpanded = true,
}: {
  sectionKey: string;
  title: React.ReactNode;
  children: React.ReactNode;
  withBorder?: boolean;
  isExceptionSection?: boolean;
  className?: string;
  headerPadding?: number | string;
  contentPadding?: number | string;
  initialExpanded?: boolean;
}): React.ReactElement | null => {
  const [expanded, setExpanded] = useState(initialExpanded);
  const { theme } = useDesignSystemTheme();

  const borderColor = isExceptionSection ? theme.colors.actionDangerPrimaryBackgroundDefault : theme.colors.border;
  const headerBackground = isExceptionSection
    ? `${theme.colors.actionDangerPrimaryBackgroundDefault}15`
    : theme.colors.backgroundSecondary;
  const contentBackground = isExceptionSection ? theme.colors.backgroundPrimary : undefined;

  return (
    <div
      className={className}
      css={{
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div
        css={{
          alignItems: 'center',
          display: 'flex',
          flexDirection: 'row',
          gap: theme.spacing.xs,
          padding: withBorder ? (headerPadding ?? theme.spacing.sm) : 0,
          // Only exception sections get a header background + border (for danger
          // emphasis); normal section headers stay borderless and transparent.
          background: withBorder && isExceptionSection ? headerBackground : undefined,
          marginBottom: withBorder ? 0 : theme.spacing.sm,
          borderBlock: withBorder && isExceptionSection ? `1px solid ${borderColor}` : undefined,
          '&:has(button[aria-expanded="true"]) [data-tools-expand-hint]': {
            display: 'none',
          },
        }}
      >
        <Typography.Title withoutMargins level={4} css={{ width: '100%' }}>
          {title}
        </Typography.Title>
        <Button
          size="small"
          componentId="shared.model-trace-explorer.expand"
          data-testid={`model-trace-explorer-${sectionKey}-section-toggle`}
          aria-expanded={expanded}
          type="tertiary"
          icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          onClick={() => setExpanded(!expanded)}
        />
      </div>
      {expanded && (
        <div
          css={{
            padding: withBorder ? (contentPadding ?? theme.spacing.sm) : 0,
            background: contentBackground,
          }}
        >
          {children}
        </div>
      )}
    </div>
  );
};
