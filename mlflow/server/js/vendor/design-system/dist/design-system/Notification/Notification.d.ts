import * as Toast from '@radix-ui/react-toast';
import React from 'react';
import { type DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import type { AnalyticsEventProps } from '../types';
export interface NotificationProps extends Toast.ToastProps, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnView> {
    severity?: 'info' | 'success' | 'warning' | 'error';
    isCloseable?: boolean;
}
export declare const Root: React.ForwardRefExoticComponent<NotificationProps & React.RefAttributes<HTMLLIElement>>;
export interface NotificationTitleProps extends Toast.ToastTitleProps {
}
export declare const Title: React.ForwardRefExoticComponent<NotificationTitleProps & React.RefAttributes<HTMLDivElement>>;
export interface NotificationDescriptionProps extends Toast.ToastDescriptionProps {
}
export declare const Description: React.ForwardRefExoticComponent<NotificationDescriptionProps & React.RefAttributes<HTMLDivElement>>;
export interface NotificationCloseProps extends Toast.ToastCloseProps, AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
    closeLabel?: string;
}
export declare const Close: React.ForwardRefExoticComponent<NotificationCloseProps & React.RefAttributes<HTMLButtonElement>>;
export interface NotificationProviderProps extends Toast.ToastProviderProps {
}
export declare const Provider: ({ children, ...props }: NotificationProviderProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface NotificationViewportProps extends Toast.ToastViewportProps {
}
export declare const Viewport: (props: NotificationViewportProps) => JSX.Element;
//# sourceMappingURL=Notification.d.ts.map