import { Button, PencilIcon, useDesignSystemTheme } from '@databricks/design-system';

export const SkillPencilButton = ({
  componentId,
  label,
  onClick,
}: {
  componentId: string;
  label: string;
  onClick: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Button
      componentId={componentId}
      size="small"
      icon={<PencilIcon />}
      aria-label={label}
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
