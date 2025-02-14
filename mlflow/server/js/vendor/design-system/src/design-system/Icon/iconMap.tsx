import type { IconType as AntDIconType } from 'antd/lib/notification';
import { forwardRef, type ForwardRefExoticComponent } from 'react';

import type { IconProps } from './Icon';
import { CheckCircleFillIcon, DangerFillIcon, InfoFillIcon, WarningFillIcon } from './__generated/icons';

interface MinimalIconProps {
  className?: string;
}

// TODO: Replace with custom icons
// TODO: Reuse in Alert
const filledIconsMap: Record<AntDIconType, ForwardRefExoticComponent<IconProps>> = {
  error: DangerFillIcon,
  warning: WarningFillIcon,
  success: CheckCircleFillIcon,
  info: InfoFillIcon,
};

export interface SeverityIconProps extends MinimalIconProps {
  severity: AntDIconType;
}

export const SeverityIcon = forwardRef<HTMLSpanElement, SeverityIconProps>(function (props, ref): JSX.Element {
  const FilledIcon = filledIconsMap[props.severity];
  return <FilledIcon ref={ref} {...props} />;
});
