import type { SerializedStyles } from '@emotion/react';
import type { ButtonProps as AntDButtonProps } from 'antd';
import type { Theme } from '../../theme';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import type { AnalyticsEventPropsWithStartInteraction, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export declare const SMALL_BUTTON_HEIGHT = 24;
export declare const getMemoizedButtonEmotionStyles: (props: {
    theme: Theme;
    classNamePrefix: string;
    loading?: boolean | undefined;
    withIcon?: boolean | undefined;
    onlyIcon?: boolean | undefined;
    isAnchor?: boolean | undefined;
    danger?: boolean | undefined;
    enableAnimation: boolean;
    size: ButtonSize;
    type?: "link" | "primary" | "tertiary" | undefined;
    useFocusPseudoClass?: boolean | undefined;
    forceIconStyles?: boolean | undefined;
}) => any;
export declare const getButtonEmotionStyles: ({ theme, classNamePrefix, loading, withIcon, onlyIcon, isAnchor, enableAnimation, size, type, useFocusPseudoClass, forceIconStyles, danger, }: {
    theme: Theme;
    classNamePrefix: string;
    loading?: boolean | undefined;
    withIcon?: boolean | undefined;
    onlyIcon?: boolean | undefined;
    isAnchor?: boolean | undefined;
    danger?: boolean | undefined;
    enableAnimation: boolean;
    size: ButtonSize;
    type?: "link" | "primary" | "tertiary" | undefined;
    useFocusPseudoClass?: boolean | undefined;
    forceIconStyles?: boolean | undefined;
}) => SerializedStyles;
export type ButtonSize = 'middle' | 'small';
export interface ButtonProps extends Omit<AntDButtonProps, 'type' | 'ghost' | 'shape' | 'size'>, HTMLDataAttributes, DangerouslySetAntdProps<AntDButtonProps>, Omit<WithLoadingState, 'loading'>, AnalyticsEventPropsWithStartInteraction<DesignSystemEventProviderAnalyticsEventTypes.OnClick | DesignSystemEventProviderAnalyticsEventTypes.OnView> {
    type?: 'primary' | 'link' | 'tertiary';
    size?: ButtonSize;
    endIcon?: React.ReactNode;
    dangerouslySetForceIconStyles?: boolean;
    dangerouslyUseFocusPseudoClass?: boolean;
    dangerouslyAppendWrapperCss?: React.CSSProperties;
}
export declare const Button: import("react").ForwardRefExoticComponent<ButtonProps & import("react").RefAttributes<HTMLButtonElement>>;
//# sourceMappingURL=Button.d.ts.map