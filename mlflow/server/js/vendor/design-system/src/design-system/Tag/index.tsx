import type { Interpolation } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import type { HTMLAttributes } from 'react';
import React, { useCallback, forwardRef, useMemo } from 'react';

import type { Theme } from '../../theme';
import { lightColorList } from '../../theme/_generated/SemanticColors-Light';
import type { SecondaryColorToken, TagColorToken } from '../../theme/colors';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { CloseIcon } from '../Icon';
import type { AnalyticsEventProps, HTMLDataAttributes } from '../types';
import { useNotifyOnFirstView } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';

export interface TagProps
  extends HTMLDataAttributes,
    HTMLAttributes<HTMLSpanElement>,
    AnalyticsEventProps<
      DesignSystemEventProviderAnalyticsEventTypes.OnView | DesignSystemEventProviderAnalyticsEventTypes.OnClick
    > {
  /**
   * The color of the tag.
   */
  color?: TagColors;

  /**
   * Text to be rendered inside the tag.
   */
  children: React.ReactNode;

  /**
   * Whether or not the tag should be closable.
   */
  closable?: boolean;

  /**
   * Function called when the close button is clicked.
   */
  onClose?: () => void;

  closeButtonProps?: Omit<React.HTMLAttributes<HTMLButtonElement>, 'children' | 'onClick' | 'onMouseDown'>;
  icon?: React.ReactNode;
}

export type TagColors =
  | 'default'
  | 'brown'
  | 'coral'
  | 'charcoal'
  | 'indigo'
  | 'lemon'
  | 'lime'
  | 'pink'
  | 'purple'
  | 'teal'
  | 'turquoise';
type SemanticTagColors =
  | 'Default'
  | 'Brown'
  | 'Coral'
  | 'Charcoal'
  | 'Indigo'
  | 'Lemon'
  | 'Lime'
  | 'Pink'
  | 'Purple'
  | 'Teal'
  | 'Turquoise';

const oldTagColorsMap: Record<SecondaryColorToken | 'default' | 'charcoal', TagColorToken | 'grey600'> = {
  default: 'tagDefault',
  brown: 'tagBrown',
  coral: 'tagCoral',
  charcoal: 'grey600',
  indigo: 'tagIndigo',
  lemon: 'tagLemon',
  lime: 'tagLime',
  pink: 'tagPink',
  purple: 'tagPurple',
  teal: 'tagTeal',
  turquoise: 'tagTurquoise',
};

function getTagEmotionStyles(
  theme: Theme,
  color: SecondaryColorToken | 'default' | 'charcoal' = 'default',
  clickable = false,
  closable = false,
  useNewTagColors?: boolean,
) {
  let textColor = theme.colors.tagText;
  let backgroundColor = theme.colors[oldTagColorsMap[color]];
  let iconColor = '';
  let outlineColor = theme.colors.actionDefaultBorderFocus;

  if (useNewTagColors) {
    const capitalizedColor = (color.charAt(0).toUpperCase() + color.slice(1)) as SemanticTagColors;
    textColor = theme.DU_BOIS_INTERNAL_ONLY.colors[`tagText${capitalizedColor}`];
    backgroundColor = theme.DU_BOIS_INTERNAL_ONLY.colors[`tagBackground${capitalizedColor}`];
    iconColor = theme.DU_BOIS_INTERNAL_ONLY.colors[`tagIcon${capitalizedColor}`];

    if (color === 'charcoal') {
      outlineColor = theme.colors.white;
    }
  }

  let iconHover = theme.colors.tagIconHover;
  let iconPress = theme.colors.tagIconPress;
  let tagHover = theme.colors.tagHover;
  let tagPress = theme.colors.tagPress;

  // Because the default tag background color changes depending on system theme, so do its other variables.
  if (color === 'default' && !useNewTagColors) {
    textColor = theme.colors.textPrimary;
    iconHover = theme.colors.actionTertiaryTextHover;
    iconPress = theme.colors.actionTertiaryTextPress;
  }

  // Because lemon is a light yellow, all its variables pull from the light mode palette, regardless of system theme.
  if (color === 'lemon' && !useNewTagColors) {
    textColor = lightColorList.textPrimary;
    iconHover = lightColorList.actionTertiaryTextHover;
    iconPress = lightColorList.actionTertiaryTextPress;
    tagHover = lightColorList.actionTertiaryBackgroundHover;
    tagPress = lightColorList.actionTertiaryBackgroundPress;
  }

  return {
    wrapper: {
      backgroundColor: backgroundColor,
      display: 'inline-flex',
      alignItems: 'center',
      marginRight: theme.spacing.sm,
      borderRadius: theme.legacyBorders.borderRadiusMd,
    },
    tag: {
      border: 'none',
      color: textColor,
      padding: useNewTagColors ? '' : '2px 4px',
      backgroundColor: useNewTagColors ? 'transparent' : backgroundColor,
      borderRadius: theme.legacyBorders.borderRadiusMd,
      marginRight: theme.spacing.sm,
      display: 'inline-block',
      cursor: clickable ? 'pointer' : 'default',

      ...(useNewTagColors && {
        ...(closable && {
          borderTopRightRadius: 0,
          borderBottomRightRadius: 0,
        }),
        ...(clickable && {
          '&:hover': {
            '& > div': {
              backgroundColor: theme.colors.actionDefaultBackgroundHover,
            },
          },
          '&:active': {
            '& > div': {
              backgroundColor: theme.colors.actionDefaultBackgroundPress,
            },
          },
        }),
      }),
    },
    content: {
      display: 'flex',
      alignItems: 'center',
      minWidth: 0,
      ...(useNewTagColors && {
        height: theme.typography.lineHeightBase,
      }),
    },
    close: {
      height: useNewTagColors ? theme.typography.lineHeightBase : theme.general.iconFontSize,
      width: useNewTagColors ? theme.typography.lineHeightBase : theme.general.iconFontSize,
      lineHeight: `${theme.general.iconFontSize}px`,
      padding: 0,
      color: textColor,
      fontSize: theme.general.iconFontSize,
      borderTopRightRadius: theme.legacyBorders.borderRadiusMd,
      borderBottomRightRadius: theme.legacyBorders.borderRadiusMd,
      border: 'none',
      background: 'none',
      cursor: 'pointer',
      marginLeft: theme.spacing.xs,
      ...(useNewTagColors
        ? {
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: 0,
          }
        : {
            marginRight: -theme.spacing.xs,
            margin: '-2px -4px -2px 4px',
          }),

      '&:hover': {
        backgroundColor: useNewTagColors ? theme.colors.actionDefaultBackgroundHover : tagHover,
        color: iconHover,
      },

      '&:active': {
        backgroundColor: useNewTagColors ? theme.colors.actionDefaultBackgroundPress : tagPress,
        color: iconPress,
      },

      '&:focus-visible': {
        outlineStyle: 'solid',
        outlineWidth: 1,
        outlineOffset: useNewTagColors ? -2 : 1,
        outlineColor,
      },

      '.anticon': {
        verticalAlign: 0,
      },
    },
    text: {
      padding: 0,
      fontSize: theme.typography.fontSizeBase,
      fontWeight: theme.typography.typographyRegularFontWeight,
      lineHeight: theme.typography.lineHeightSm,
      '& .anticon': {
        verticalAlign: 'text-top',
      },
      whiteSpace: 'nowrap',
    },
    icon: {
      color: iconColor,
      paddingLeft: theme.spacing.xs,
      height: theme.typography.lineHeightBase,
      display: 'inline-flex',
      alignItems: 'center',
      borderTopLeftRadius: theme.legacyBorders.borderRadiusMd,
      borderBottomLeftRadius: theme.legacyBorders.borderRadiusMd,

      '& + div': {
        borderTopLeftRadius: 0,
        borderBottomLeftRadius: 0,

        ...(closable && {
          borderTopRightRadius: 0,
          borderBottomRightRadius: 0,
        }),
      },
    },
    childrenWrapper: {
      paddingLeft: theme.spacing.xs,
      paddingRight: theme.spacing.xs,
      height: theme.typography.lineHeightBase,
      display: 'inline-flex',
      alignItems: 'center',
      borderRadius: theme.legacyBorders.borderRadiusMd,
      minWidth: 0,
    },
  } satisfies Record<string, Interpolation<Theme>>;
}

export const Tag = forwardRef<HTMLDivElement, TagProps>((props, forwardedRef) => {
  const { theme } = useDesignSystemTheme();
  const {
    color,
    children,
    closable,
    onClose,
    role = 'status',
    closeButtonProps,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
    componentId,
    icon,
    onClick,
    ...attributes
  } = props;
  const isClickable = Boolean(props.onClick);
  const useNewTagColors = safex('databricks.fe.designsystem.useNewTagColors', false);

  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Tag,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
  });
  const { elementRef } = useNotifyOnFirstView<HTMLDivElement>({ onView: eventContext.onView });
  const mergedRef = useMergeRefs([elementRef, forwardedRef]);

  const closeButtonComponentId = componentId ? `${componentId}.close` : undefined;

  const closeButtonEventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Button,
    componentId: closeButtonComponentId,
    analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
  });

  const handleClick = useCallback(
    (e) => {
      if (onClick) {
        eventContext.onClick(e);
        onClick(e);
      }
    },
    [eventContext, onClick],
  );

  const handleCloseClick = useCallback(
    (e) => {
      closeButtonEventContext.onClick(e);
      e.stopPropagation();
      if (onClose) {
        onClose();
      }
    },
    [closeButtonEventContext, onClose],
  );

  const styles = getTagEmotionStyles(theme, color, isClickable, closable, useNewTagColors);

  return useNewTagColors ? (
    <div
      ref={mergedRef}
      role={role}
      onClick={handleClick}
      css={[styles.wrapper]}
      {...attributes}
      {...addDebugOutlineIfEnabled()}
      // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
      tabIndex={isClickable ? 0 : -1}
    >
      <div css={[styles.tag, styles.content, styles.text, { marginRight: 0 }]} {...eventContext.dataComponentProps}>
        {icon && <div css={[styles.icon]}>{icon}</div>}
        <div css={[styles.childrenWrapper]}>{children}</div>
      </div>
      {closable && (
        <button
          css={styles.close}
          tabIndex={0}
          onClick={handleCloseClick}
          onMouseDown={(e) => {
            // Keeps dropdowns of any underlying select from opening.
            e.stopPropagation();
          }}
          {...closeButtonProps}
          {...closeButtonEventContext.dataComponentProps}
        >
          <CloseIcon css={{ fontSize: theme.general.iconFontSize - 4 }} />
        </button>
      )}
    </div>
  ) : (
    <div
      ref={mergedRef}
      role={role}
      {...attributes}
      onClick={handleClick}
      css={styles.tag}
      {...addDebugOutlineIfEnabled()}
      {...eventContext.dataComponentProps}
    >
      <div css={[styles.content, styles.text]}>
        {children}

        {closable && (
          <button
            css={styles.close}
            tabIndex={0}
            onClick={handleCloseClick}
            onMouseDown={(e) => {
              // Keeps dropdowns of any underlying select from opening.
              e.stopPropagation();
            }}
            {...closeButtonProps}
            {...closeButtonEventContext.dataComponentProps}
          >
            <CloseIcon css={{ fontSize: theme.general.iconFontSize - 4 }} />
          </button>
        )}
      </div>
    </div>
  );
});
