import type { ReactNode } from 'react';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';

/** A bordered section of the skill page with a shaded title bar. */
export const SkillSectionCard = ({
  title,
  description,
  actions,
  children,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <section
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        overflow: 'hidden',
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing.sm,
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          backgroundColor: theme.colors.backgroundSecondary,
          borderBottom: `1px solid ${theme.colors.border}`,
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <Typography.Text bold>{title}</Typography.Text>
          {description && <Typography.Text color="secondary">{description}</Typography.Text>}
        </div>
        {actions}
      </div>
      <div css={{ padding: theme.spacing.md }}>{children}</div>
    </section>
  );
};
