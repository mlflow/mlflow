import type { ReactNode } from 'react';
import React from 'react';
export declare const BANNER_MIN_HEIGHT = 68;
export declare const BANNER_MAX_HEIGHT = 82;
export type BannerLevel = 'info' | 'warning' | 'error';
export interface BannerProps {
    level: BannerLevel;
    message: ReactNode;
    description?: ReactNode;
    ctaText?: ReactNode;
    onAccept?: () => void;
    closable?: boolean;
    onClose?: () => void;
    'data-testid'?: string;
    closeButtonAriaLabel?: string;
}
export declare const Banner: React.FC<BannerProps>;
//# sourceMappingURL=Banner.d.ts.map