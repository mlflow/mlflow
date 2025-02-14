import { InfoCircleOutlined } from '@ant-design/icons';

import type { LegacyTooltipProps } from './LegacyTooltip';
import { LegacyTooltip } from './LegacyTooltip';
import { useDesignSystemTheme } from '../Hooks';
import { addDebugOutlineIfEnabled } from '../utils/debug';

/**
 * `LegacyInfoTooltip` is deprecated in favor of the new `InfoTooltip` component
 * @deprecated
 */
export interface LegacyInfoTooltipProps extends Omit<React.HTMLAttributes<HTMLSpanElement>, 'title'> {
  title: React.ReactNode;
  tooltipProps?: Omit<LegacyTooltipProps, 'children' | 'title'>;
  iconTitle?: string;
  isKeyboardFocusable?: boolean;
}

/**
 * `LegacyInfoTooltip` is deprecated in favor of the new `InfoTooltip` component
 * @deprecated
 */
export const LegacyInfoTooltip = ({
  title,
  tooltipProps,
  iconTitle,
  isKeyboardFocusable = true,
  ...iconProps
}: LegacyInfoTooltipProps): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  return (
    <LegacyTooltip useAsLabel title={title} {...tooltipProps}>
      <span {...addDebugOutlineIfEnabled()} style={{ display: 'inline-flex' }}>
        {/*eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex */}
        <InfoCircleOutlined
          tabIndex={isKeyboardFocusable ? 0 : -1}
          aria-hidden="false"
          aria-label={iconTitle}
          alt={iconTitle}
          css={{ fontSize: theme.typography.fontSizeSm, color: theme.colors.textSecondary }}
          {...iconProps}
        />
      </span>
    </LegacyTooltip>
  );
};
