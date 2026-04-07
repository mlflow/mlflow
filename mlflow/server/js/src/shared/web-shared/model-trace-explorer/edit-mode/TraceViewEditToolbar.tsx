import { Button, Input, useDesignSystemTheme } from '@databricks/design-system';
import type { SpanRange } from '../hooks/useTraceViews';

interface TraceViewEditToolbarProps {
  name: string;
  onNameChange: (name: string) => void;
  ranges: SpanRange[];
  onCancel: () => void;
  onSave: () => void;
  isSaving: boolean;
}

export const TraceViewEditToolbar = ({
  name,
  onNameChange,
  ranges,
  onCancel,
  onSave,
  isSaving,
}: TraceViewEditToolbarProps) => {
  const { theme } = useDesignSystemTheme();
  const canSave = name.trim().length > 0 && ranges.length > 0 && !isSaving;

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        padding: theme.spacing.sm,
        backgroundColor: theme.colors.backgroundSecondary,
        borderBottom: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
      }}
    >
      <Input
        componentId="trace-view-edit-toolbar.name"
        value={name}
        onChange={(e) => onNameChange(e.target.value)}
        placeholder="View name"
        css={{ flex: 1 }}
      />
      <Button
        componentId="trace-view-edit-toolbar.cancel"
        type="tertiary"
        onClick={onCancel}
      >
        Cancel
      </Button>
      <Button
        componentId="trace-view-edit-toolbar.save"
        type="primary"
        onClick={onSave}
        disabled={!canSave}
        loading={isSaving}
      >
        Save
      </Button>
    </div>
  );
};
