import type { IconType as AntDIconType } from 'antd/lib/notification';
import { type ForwardRefExoticComponent } from 'react';
interface MinimalIconProps {
    className?: string;
}
export interface SeverityIconProps extends MinimalIconProps {
    severity: AntDIconType;
}
export declare const SeverityIcon: ForwardRefExoticComponent<SeverityIconProps & import("react").RefAttributes<HTMLSpanElement>>;
export {};
//# sourceMappingURL=iconMap.d.ts.map