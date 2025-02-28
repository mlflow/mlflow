import type { Theme } from '@emotion/react';

import { useDesignSystemTheme } from '../Hooks';
import type { IconProps } from '../Icon';
import { CircleIcon, CircleOffIcon, CircleOutlineIcon } from '../Icon';

export type ResourceStatus = 'online' | 'disconnected' | 'offline';
export interface ResourceStatusIndicatorProps {
  status: ResourceStatus;
  style?: React.CSSProperties;
}

const STATUS_TO_ICON: {
  [key in ResourceStatus]: ({
    theme,
    style,
    ...props
  }: IconProps & { theme: Theme; style?: React.CSSProperties }) => JSX.Element;
} = {
  online: ({ theme, style, ...props }) => <CircleIcon color="success" css={{ ...style }} {...props} />,
  disconnected: ({ theme, style, ...props }) => (
    <CircleOutlineIcon css={{ color: theme.colors.grey500, ...style }} {...props} />
  ),
  offline: ({ theme, style, ...props }) => <CircleOffIcon css={{ color: theme.colors.grey500, ...style }} {...props} />,
};

export const ResourceStatusIndicator = (props: ResourceStatusIndicatorProps) => {
  const { status, style, ...restProps } = props;
  const { theme } = useDesignSystemTheme();
  const StatusIcon = STATUS_TO_ICON[status];

  return <StatusIcon theme={theme} style={style} {...restProps} />;
};
