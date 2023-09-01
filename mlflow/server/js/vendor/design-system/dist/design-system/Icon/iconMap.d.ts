/// <reference types="react" />
import type { IconType as AntDIconType } from 'antd/lib/notification';
interface MinimalIconProps {
    className?: string;
}
export interface SeverityIconProps extends MinimalIconProps {
    severity: AntDIconType;
}
export declare function SeverityIcon(props: SeverityIconProps): JSX.Element;
export {};
//# sourceMappingURL=iconMap.d.ts.map