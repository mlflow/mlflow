import type { HTMLAttributes } from 'react';
import React from 'react';
import type { DesignSystemEventProviderAnalyticsEventTypes } from '../../design-system/DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventOptionalProps, DangerousGeneralProps } from '../../design-system/types';
export interface PreviewCardProps extends DangerousGeneralProps, Omit<HTMLAttributes<HTMLDivElement>, 'title'>, AnalyticsEventOptionalProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
    icon?: React.ReactNode;
    title?: React.ReactNode;
    subtitle?: React.ReactNode;
    titleActions?: React.ReactNode;
    startActions?: React.ReactNode;
    endActions?: React.ReactNode;
    image?: React.ReactNode;
    size?: 'default' | 'large';
    onClick?: React.MouseEventHandler<HTMLDivElement>;
}
export declare const PreviewCard: ({ icon, title, subtitle, titleActions, children, startActions, endActions, image, onClick, size, dangerouslyAppendEmotionCSS, componentId, analyticsEvents, ...props }: PreviewCardProps) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=PreviewCard.d.ts.map