import { InfoIcon, Popover, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { popoverTriggerStyles, sectionHeadingRowStyles } from '../styles';

export const SubsectionHelpHeading = ({
  title,
  componentId,
  helpAriaLabel,
  helpText,
  actions,
}: {
  title: React.ReactNode;
  componentId: string;
  helpAriaLabel: string;
  helpText: React.ReactNode;
  actions?: React.ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={sectionHeadingRowStyles(theme)}>
      <Typography.Text bold>{title}</Typography.Text>
      <Popover.Root componentId={componentId}>
        <Popover.Trigger css={popoverTriggerStyles(theme)} aria-label={helpAriaLabel}>
          <InfoIcon />
        </Popover.Trigger>
        <Popover.Content align="start" css={{ maxWidth: 360 }}>
          <Typography.Paragraph withoutMargins>{helpText}</Typography.Paragraph>
          <Popover.Arrow />
        </Popover.Content>
      </Popover.Root>
      {actions}
    </div>
  );
};
