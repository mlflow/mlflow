import { Typography } from '@databricks/design-system';

export function TagAssignmentLabel({ children }: { children: React.ReactNode }) {
  return <Typography.Text bold>{children}</Typography.Text>;
}
