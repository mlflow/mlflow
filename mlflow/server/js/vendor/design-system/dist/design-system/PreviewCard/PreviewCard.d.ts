import type { HTMLAttributes } from 'react';
import React from 'react';
import type { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventOptionalProps, DangerousGeneralProps } from '../types';
export interface PreviewCardProps extends DangerousGeneralProps, Omit<HTMLAttributes<HTMLDivElement>, 'title'>, AnalyticsEventOptionalProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
    icon?: React.ReactNode;
    title?: React.ReactNode;
    subtitle?: React.ReactNode;
    titleActions?: React.ReactNode;
    startActions?: React.ReactNode;
    endActions?: React.ReactNode;
    image?: React.ReactNode;
    fullBleedImage?: boolean;
    size?: 'default' | 'large';
    onClick?: React.MouseEventHandler<HTMLDivElement>;
    disabled?: boolean;
    selected?: boolean;
    href?: string;
    /** If href is defined, this will be used as the target attribute */
    target?: string;
}
export declare const PreviewCard: ({ icon, title, subtitle, titleActions, children, startActions, endActions, image, fullBleedImage, onClick, size, dangerouslyAppendEmotionCSS, componentId, analyticsEvents, disabled, selected, href, target, ...props }: PreviewCardProps) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=PreviewCard.d.ts.map