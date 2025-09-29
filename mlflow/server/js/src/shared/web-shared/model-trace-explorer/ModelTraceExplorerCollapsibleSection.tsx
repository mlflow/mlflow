import { useState } from 'react';

import { Button, ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';

export const ModelTraceExplorerCollapsibleSection = ({
  sectionKey,
  title,
  children,
  withBorder = false,
  className,
}: {
  sectionKey: string;
  title: React.ReactNode;
  children: React.ReactNode;
  withBorder?: boolean;
  className?: string;
}) => {
  const [expanded, setExpanded] = useState(true);
  const { theme } = useDesignSystemTheme();
  return (
    <div
      className={className}
      css={{
        display: 'flex',
        flexDirection: 'column',
        borderRadius: theme.borders.borderRadiusMd,
      }}
    >
      <div
        css={{
          alignItems: 'center',
          display: 'flex',
          flexDirection: 'row',
          gap: theme.spacing.xs,
          padding: withBorder ? theme.spacing.sm : 0,
          background: withBorder ? theme.colors.backgroundSecondary : undefined,
          borderTopLeftRadius: theme.borders.borderRadiusMd,
          borderTopRightRadius: theme.borders.borderRadiusMd,
          borderBottomLeftRadius: expanded ? 0 : theme.borders.borderRadiusMd,
          borderBottomRightRadius: expanded ? 0 : theme.borders.borderRadiusMd,
          border: withBorder ? `1px solid ${theme.colors.border}` : undefined,
          marginBottom: withBorder ? 0 : theme.spacing.sm,
        }}
      >
        <Button
          size="small"
          componentId={`shared.model-trace-explorer.expand-${sectionKey}`}
          type="tertiary"
          icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          onClick={() => setExpanded(!expanded)}
        />
        <Typography.Title withoutMargins level={4} css={{ width: '100%' }}>
          {title}
        </Typography.Title>
      </div>
      {expanded && (
        <div
          css={{
            border: withBorder ? `1px solid ${theme.colors.border}` : undefined,
            borderTop: 'none',
            borderBottomLeftRadius: withBorder ? theme.borders.borderRadiusMd : undefined,
            borderBottomRightRadius: withBorder ? theme.borders.borderRadiusMd : undefined,
            padding: withBorder ? theme.spacing.sm : 0,
          }}
        >
          {children}
        </div>
      )}
    </div>
  );
};
