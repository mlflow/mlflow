import * as Toast from '@radix-ui/react-toast';
import React from 'react';
export interface NotificationV2Props extends Toast.ToastProps {
    severity?: 'info' | 'success' | 'warning' | 'error';
    isCloseable?: boolean;
}
export declare const Root: React.ForwardRefExoticComponent<NotificationV2Props & React.RefAttributes<HTMLLIElement>>;
export interface NotificationV2TitleProps extends Toast.ToastTitleProps {
}
export declare const Title: React.ForwardRefExoticComponent<NotificationV2TitleProps & React.RefAttributes<HTMLDivElement>>;
export interface NotificationV2DescriptionProps extends Toast.ToastDescriptionProps {
}
export declare const Description: React.ForwardRefExoticComponent<NotificationV2DescriptionProps & React.RefAttributes<HTMLDivElement>>;
export interface NotificationV2CloseProps extends Toast.ToastCloseProps {
    closeLabel?: string;
}
export declare const Close: React.ForwardRefExoticComponent<NotificationV2CloseProps & React.RefAttributes<HTMLButtonElement>>;
export interface NotificationV2ProviderProps extends Toast.ToastProviderProps {
}
export declare const Provider: ({ children, ...props }: NotificationV2ProviderProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export interface NotificationV2ViewportProps extends Toast.ToastViewportProps {
}
export declare const Viewport: (props: NotificationV2ViewportProps) => JSX.Element;
//# sourceMappingURL=NotificationV2.d.ts.map