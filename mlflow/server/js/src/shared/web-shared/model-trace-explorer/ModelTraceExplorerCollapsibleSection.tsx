import { useState } from 'react';

import { Button, ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';

export const ModelTraceExplorerCollapsibleSection = ({
  sectionKey,
  title,
  children,
  withBorder = false,
  isExceptionSection = false,
  className,
}: {
  sectionKey: string;
  title: React.ReactNode;
  children: React.ReactNode;
  withBorder?: boolean;
  isExceptionSection?: boolean;
  className?: string;
}) => {
  const [expanded, setExpanded] = useState(true);
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
          background: withBorder ? headerBackground : undefined,
          borderTopLeftRadius: theme.borders.borderRadiusMd,
          borderTopRightRadius: theme.borders.borderRadiusMd,
          borderBottomLeftRadius: expanded ? 0 : theme.borders.borderRadiusMd,
          borderBottomRightRadius: expanded ? 0 : theme.borders.borderRadiusMd,
          border: withBorder ? `1px solid ${borderColor}` : undefined,
          marginBottom: withBorder ? 0 : theme.spacing.sm,
        }}
      >
        <Button
          size="small"
          componentId="shared.model-trace-explorer.expand"
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
            border: withBorder ? `1px solid ${borderColor}` : undefined,
            borderTop: 'none',
            borderBottomLeftRadius: withBorder ? theme.borders.borderRadiusMd : undefined,
            borderBottomRightRadius: withBorder ? theme.borders.borderRadiusMd : undefined,
            padding: withBorder ? theme.spacing.sm : 0,
            background: contentBackground,
          }}
        >
          {children}
        </div>
      )}
    </div>
  );
};
