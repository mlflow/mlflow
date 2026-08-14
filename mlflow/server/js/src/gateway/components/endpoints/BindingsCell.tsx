import { LinkIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { EndpointBinding } from '../../types';

interface BindingsCellProps {
  bindings: EndpointBinding[];
  onViewBindings: () => void;
}

export const BindingsCell = ({ bindings, onViewBindings }: BindingsCellProps) => {
  const { theme } = useDesignSystemTheme();

  if (!bindings || bindings.length === 0) {
    return <Typography.Text color="secondary">-</Typography.Text>;
  }

  return (
    <button
      type="button"
      onClick={onViewBindings}
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        background: 'none',
        border: 'none',
        padding: 0,
        cursor: 'pointer',
        color: theme.colors.actionPrimaryBackgroundDefault,
        '&:hover': {
          textDecoration: 'underline',
        },
      }}
    >
      <LinkIcon css={{ color: theme.colors.textSecondary, fontSize: 14 }} />
      <Typography.Text css={{ color: 'inherit' }}>
        {bindings.length} {bindings.length === 1 ? 'resource' : 'resources'}
      </Typography.Text>
    </button>
  );
};
