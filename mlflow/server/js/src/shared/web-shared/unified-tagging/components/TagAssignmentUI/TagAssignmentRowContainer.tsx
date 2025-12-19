import { useDesignSystemTheme } from '@databricks/design-system';

export function TagAssignmentRowContainer({ children }: { children: React.ReactNode }) {
  const { theme } = useDesignSystemTheme();
  return <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>{children}</div>;
}
