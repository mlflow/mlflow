import type { ReactNode } from 'react';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';

export interface FieldLabelProps {
  children: ReactNode;
}

/**
 * Bold label for a form field. Renders as a block with a small bottom
 * margin so the label sits cleanly above its input — works for both
 * block inputs (``Input``, ``SimpleSelect``, ``Radio.Group``,
 * ``DialogCombobox``) and inline-text content (e.g. the read-only
 * "Workspace: <name>" line in the role-permission form).
 */
export const FieldLabel = ({ children }: FieldLabelProps) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
      {children}
    </Typography.Text>
  );
};
