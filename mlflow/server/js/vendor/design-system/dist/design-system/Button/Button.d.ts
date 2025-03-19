import type { SerializedStyles } from '@emotion/react';
import type { ButtonProps as AntDButtonProps } from 'antd';
import type { ComponentTheme } from '../../theme';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import type { AnalyticsEventPropsWithStartInteraction, DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
export declare const getMemoizedButtonEmotionStyles: (props: Parameters<typeof getButtonEmotionStyles>[0]) => any;
export declare const getButtonEmotionStyles: ({ theme, classNamePrefix, loading, withIcon, onlyIcon, isAnchor, enableAnimation, size, type, useFocusPseudoClass, forceIconStyles, danger, useNewShadows, }: {
    theme: ComponentTheme;
    classNamePrefix: string;
    loading?: boolean;
    withIcon?: boolean;
    onlyIcon?: boolean;
    isAnchor?: boolean;
    danger?: boolean;
    enableAnimation: boolean;
    size: ButtonSize;
    type?: ButtonProps["type"];
    useFocusPseudoClass?: boolean;
    forceIconStyles?: boolean;
    useNewShadows: boolean;
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