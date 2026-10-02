import { Button, PencilIcon, useDesignSystemTheme } from '@databricks/design-system';

export const SkillPencilButton = ({
  componentId,
  label,
  onClick,
  disabled,
}: {
  componentId: string;
  label: string;
  onClick: () => void;
  disabled?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Button
      componentId={componentId}
      size="small"
      icon={<PencilIcon />}
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      css={{
        color: theme.colors.textSecondary,
        backgroundColor: 'transparent',
        borderColor: 'transparent',
        boxShadow: 'none',
      }}
    />
  );
};
