import { Tooltip, type TooltipProps } from './Tooltip';
import { useDesignSystemTheme } from '../Hooks';
import { InfoIcon } from '../Icon';

export interface InfoTooltipProps extends Omit<TooltipProps, 'children'> {
  iconTitle?: string;
}

export const InfoTooltip = ({ content, iconTitle = 'More information', ...props }: InfoTooltipProps): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  return (
    <Tooltip content={content} {...props}>
      <InfoIcon
        tabIndex={0}
        aria-hidden="false"
        aria-label={iconTitle}
        alt={iconTitle}
        css={{ color: theme.colors.textSecondary }}
      />
    </Tooltip>
  );
};
