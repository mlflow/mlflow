import { useDesignSystemTheme } from '@databricks/design-system';

export interface StackedComponentsProps {
  first: React.ReactNode;
  second?: React.ReactNode;
  // Allow overriding the default styles if needed:
  gap?: string;
  borderRadius?: string;
  marginY?: string;
}

export const StackedComponents = (props: StackedComponentsProps) => {
  const { first, second, gap, borderRadius, marginY } = props;
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: gap || theme.spacing.sm,
        borderRadius: borderRadius || theme.legacyBorders.borderRadiusMd,
        marginTop: marginY || 'auto',
        marginBottom: marginY || 'auto',
      }}
    >
      {first}
      {second}
    </div>
  );
};
