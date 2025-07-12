import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import React, { useMemo } from 'react';
import { Button } from '../../design-system/Button';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, DesignSystemEventProviderComponentSubTypeMap, } from '../../design-system/DesignSystemEventProvider/DesignSystemEventProvider';
import { useDesignSystemTheme } from '../../design-system/Hooks/useDesignSystemTheme';
import { CloseIcon, DangerIcon, MegaphoneIcon, WarningIcon } from '../../design-system/Icon';
import { Typography } from '../../design-system/Typography';
import { addDebugOutlineIfEnabled } from '../../design-system/utils/debug';
import { useNotifyOnFirstView } from '../../design-system/utils/useNotifyOnFirstView';
import { primitiveColors } from '../../theme/_generated/PrimitiveColors';
const { Text, Paragraph } = Typography;
export const BANNER_MIN_HEIGHT = 68;
// Max height will allow 2 lines of description (3 lines total)
export const BANNER_MAX_HEIGHT = 82;
const useStyles = (props, theme) => {
    const bannerLevelToBannerColors = {
        info_light_purple: {
            backgroundDefaultColor: theme.isDarkMode ? '#6E2EC729' : '#ECE1FC',
            actionButtonBackgroundHoverColor: theme.colors.actionDefaultBackgroundHover,
            actionButtonBackgroundPressColor: theme.colors.actionDefaultBackgroundPress,
            textColor: theme.colors.actionDefaultTextDefault,
            textHoverColor: '#92A4B38F',
            textPressColor: theme.colors.actionDefaultTextDefault,
            borderDefaultColor: theme.isDarkMode ? '#955CE5' : '#E2D0FB',
            actionBorderColor: '#92A4B38F',
            closeIconColor: theme.isDarkMode ? '#92A4B3' : '#5F7281',
            iconColor: theme.colors.purple,
            actionButtonBorderHoverColor: theme.colors.actionDefaultBorderHover,
            actionButtonBorderPressColor: theme.colors.actionDefaultBorderPress,
            closeIconBackgroundHoverColor: theme.colors.actionTertiaryBackgroundHover,
            closeIconTextHoverColor: theme.colors.actionTertiaryTextHover,
            closeIconBackgroundPressColor: theme.colors.actionDefaultBackgroundPress,
            closeIconTextPressColor: theme.colors.actionTertiaryTextPress,
        },
        info_dark_purple: {
            backgroundDefaultColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple,
            actionButtonBackgroundHoverColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple,
            actionButtonBackgroundPressColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple,
            textColor: theme.colors.actionPrimaryTextDefault,
            textHoverColor: theme.colors.actionPrimaryTextHover,
            textPressColor: theme.colors.actionPrimaryTextPress,
            borderDefaultColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple,
        },
        // Clean up the experimental info banners
        info: {
            backgroundDefaultColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple,
            actionButtonBackgroundHoverColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple,
            actionButtonBackgroundPressColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple,
            textColor: theme.colors.actionPrimaryTextDefault,
            textHoverColor: theme.colors.actionPrimaryTextHover,
            textPressColor: theme.colors.actionPrimaryTextPress,
            borderDefaultColor: theme.isDarkMode ? '#BC92F7DB' : theme.colors.purple,
        },
        // TODO (PLAT-80558, zack.brody) Update hover and press states once we have colors for these
        warning: {
            backgroundDefaultColor: theme.colors.tagLemon,
            actionButtonBackgroundHoverColor: theme.colors.tagLemon,
            actionButtonBackgroundPressColor: theme.colors.tagLemon,
            textColor: primitiveColors.grey800,
            textHoverColor: primitiveColors.grey800,
            textPressColor: primitiveColors.grey800,
            borderDefaultColor: theme.colors.tagLemon,
        },
        error: {
            backgroundDefaultColor: theme.colors.actionDangerPrimaryBackgroundDefault,
            actionButtonBackgroundHoverColor: theme.colors.actionDangerPrimaryBackgroundHover,
            actionButtonBackgroundPressColor: theme.colors.actionDangerPrimaryBackgroundPress,
            textColor: theme.colors.actionPrimaryTextDefault,
            textHoverColor: theme.colors.actionPrimaryTextHover,
            textPressColor: theme.colors.actionPrimaryTextPress,
            borderDefaultColor: theme.colors.actionDangerPrimaryBackgroundDefault,
        },
    };
    const colorScheme = bannerLevelToBannerColors[props.level];
    return {
        banner: css `
      max-height: ${BANNER_MAX_HEIGHT}px;
      display: flex;
      align-items: center;
      width: 100%;
      padding: 8px;
      box-sizing: border-box;
      background-color: ${colorScheme.backgroundDefaultColor};
      border: 1px solid ${colorScheme.borderDefaultColor};
    `,
        iconContainer: css `
      display: flex;
      color: ${colorScheme.iconColor ? colorScheme.iconColor : colorScheme.textColor};
      align-self: ${props.description ? 'flex-start' : 'center'};
      box-sizing: border-box;
      max-width: 60px;
      padding-top: 4px;
      padding-bottom: 4px;
      padding-right: ${theme.spacing.xs}px;
    `,
        mainContent: css `
      flex-direction: column;
      align-self: ${props.description ? 'flex-start' : 'center'};
      display: flex;
      box-sizing: border-box;
      padding-right: ${theme.spacing.sm}px;
      padding-top: 2px;
      padding-bottom: 2px;
      // Add min-width so that ellipsis in child components will show.
      min-width: ${theme.spacing.lg}px;
      width: 100%;
    `,
        messageTextBlock: css `
      // handle truncation after one line
      display: -webkit-box;
      -webkit-line-clamp: 1;
      -webkit-box-orient: vertical;
      overflow: hidden;

      // Override text color to action text color
      && {
        color: ${colorScheme.textColor};
      }
    `,
        descriptionBlock: css `
      // handle truncation after two lines
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;

      // Override text color to action text color
      && {
        color: ${colorScheme.textColor};
      }
    `,
        rightContainer: css `
      margin-left: auto;
      display: flex;
      align-items: center;
    `,
        closeIconContainer: css `
      display: flex;
      margin-left: ${theme.spacing.xs}px;
      box-sizing: border-box;
      line-height: 0;
    `,
        closeButton: css `
      cursor: pointer;
      background: none;
      border: none;
      margin: 0;
      && {
        height: 24px !important;
        width: 24px !important;
        padding: ${theme.spacing.xs}px !important;
        box-shadow: unset !important;
      }
      &&:hover {
        background-color: transparent !important;
        border-color: ${colorScheme.textHoverColor}!important;
        color: ${colorScheme.closeIconTextHoverColor
            ? colorScheme.closeIconTextHoverColor
            : colorScheme.textColor}!important;
        background-color: ${colorScheme.closeIconBackgroundHoverColor
            ? colorScheme.closeIconBackgroundHoverColor
            : colorScheme.backgroundDefaultColor}!important;
      }

      &&:active {
        border-color: ${colorScheme.actionBorderColor}!important;
        color: ${colorScheme.closeIconTextPressColor
            ? colorScheme.closeIconTextPressColor
            : colorScheme.textColor}!important;
        background-color: ${colorScheme.closeIconBackgroundPressColor
            ? colorScheme.closeIconBackgroundPressColor
            : colorScheme.backgroundDefaultColor}!important;
      }
    `,
        closeIcon: css `
      color: ${colorScheme.closeIconColor ? colorScheme.closeIconColor : colorScheme.textColor}!important;
    `,
        actionButtonContainer: css `
      margin-right: ${theme.spacing.xs}px;
    `,
        // Override design system colors to show the use the action text color for text and border.
        // Also overrides text for links.
        actionButton: css `
      color: ${colorScheme.textColor}!important;
      border-color: ${colorScheme.actionBorderColor ? colorScheme.actionBorderColor : colorScheme.textColor}!important;
      box-shadow: unset !important;

      &:focus,
      &:hover {
        border-color: ${colorScheme.actionButtonBorderHoverColor
            ? colorScheme.actionButtonBorderHoverColor
            : colorScheme.textHoverColor}!important;
        color: ${colorScheme.textColor}!important;
        background-color: ${colorScheme.actionButtonBackgroundHoverColor}!important;
      }

      &:active {
        border-color: ${colorScheme.actionButtonBorderPressColor
            ? colorScheme.actionButtonBorderPressColor
            : colorScheme.actionBorderColor}!important;
        color: ${colorScheme.textPressColor}!important;
        background-color: ${colorScheme.actionButtonBackgroundPressColor}!important;
      }

      a {
        color: ${theme.colors.actionPrimaryTextDefault};
      }

      a:focus,
      a:hover {
        color: ${colorScheme.textHoverColor};
        text-decoration: none;
      }

      a:active {
        color: ${colorScheme.textPressColor}
        text-decoration: none;
      }
    `,
    };
};
const levelToIconMap = {
    info: _jsx(MegaphoneIcon, { "data-testid": "level-info-icon" }),
    info_light_purple: _jsx(MegaphoneIcon, { "data-testid": "level-info-light-purple-icon" }),
    info_dark_purple: _jsx(MegaphoneIcon, { "data-testid": "level-info-dark-purple-icon" }),
    warning: _jsx(WarningIcon, { "data-testid": "level-warning-icon" }),
    error: _jsx(DangerIcon, { "data-testid": "level-error-icon" }),
};
export const Banner = (props) => {
    const { level, message, description, ctaText, onAccept, closable, onClose, closeButtonAriaLabel, componentId, analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView], } = props;
    const [closed, setClosed] = React.useState(false);
    const { theme } = useDesignSystemTheme();
    const styles = useStyles(props, theme);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Banner,
        componentId,
        componentSubType: DesignSystemEventProviderComponentSubTypeMap[level],
        analyticsEvents: memoizedAnalyticsEvents,
    });
    const { elementRef } = useNotifyOnFirstView({ onView: eventContext.onView });
    const actionButton = onAccept && ctaText ? (_jsx("div", { css: styles.actionButtonContainer, children: _jsx(Button, { componentId: `${componentId}.accept`, onClick: onAccept, css: styles.actionButton, size: "small", children: ctaText }) })) : null;
    const close = closable !== false ? (_jsx("div", { css: styles.closeIconContainer, children: _jsx(Button, { componentId: `${componentId}.close`, css: styles.closeButton, onClick: () => {
                if (onClose) {
                    onClose();
                }
                setClosed(true);
            }, "aria-label": closeButtonAriaLabel ?? 'Close', "data-testid": "banner-dismiss", children: _jsx(CloseIcon, { css: styles.closeIcon }) }) })) : null;
    return (_jsx(_Fragment, { children: !closed && (_jsxs("div", { ref: elementRef, ...addDebugOutlineIfEnabled(), css: styles.banner, className: "banner", "data-testid": props['data-testid'], role: "alert", children: [_jsx("div", { css: styles.iconContainer, children: levelToIconMap[level] }), _jsxs("div", { css: styles.mainContent, children: [_jsx(Text, { size: "md", bold: true, css: styles.messageTextBlock, title: message, children: message }), description && (_jsx(Paragraph, { withoutMargins: true, css: styles.descriptionBlock, title: description, children: description }))] }), _jsxs("div", { css: styles.rightContainer, children: [actionButton, close] })] })) }));
};
//# sourceMappingURL=Banner.js.map