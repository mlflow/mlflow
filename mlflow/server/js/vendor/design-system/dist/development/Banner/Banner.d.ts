import React from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../../design-system/DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventProps } from '../../design-system/types';
export declare const BANNER_MIN_HEIGHT = 68;
export declare const BANNER_MAX_HEIGHT = 82;
export type BannerLevel = 'info' | 'warning' | 'error';
export interface BannerProps extends AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnView> {
    level: BannerLevel;
    message: string;
    description?: string;
    ctaText?: string;
    onAccept?: () => void;
    closable?: boolean;
    onClose?: () => void;
    'data-testid'?: string;
    closeButtonAriaLabel?: string;
}
export declare const Banner: React.FC<BannerProps>;
//# sourceMappingURL=Banner.d.ts.map