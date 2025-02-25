import type { SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import * as Popover from '@radix-ui/react-popover';
import type { ReactNode } from 'react';
import React, { forwardRef, useEffect } from 'react';

import { DialogComboboxCountBadge } from './DialogComboboxCountBadge';
import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import type { Theme } from '../../theme';
import { Button } from '../Button';
import { DesignSystemEventProviderComponentTypes, useComponentFinderContext } from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks';
import { ChevronDownIcon, CloseIcon } from '../Icon';
import { LegacyTooltip } from '../LegacyTooltip';
import { useSelectContext } from '../Select/hooks/useSelectContext';
import { ClearSelectionButton } from '../_shared_/Combobox/ClearSelectionButton';
import type { FormElementValidationState, HTMLDataAttributes, ValidationState } from '../types';
import { useDesignSystemSafexFlags } from '../utils';
import { getValidationStateColor, importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface DialogComboboxTriggerProps
  extends Popover.PopoverTriggerProps,
    FormElementValidationState,
    HTMLDataAttributes {
  minWidth?: number | string;
  maxWidth?: number | string;
  width?: number | string;
  removable?: boolean;
  onRemove?: () => void;
  allowClear?: boolean;
  onClear?: () => void;
  showTagAfterValueCount?: number;
  controlled?: boolean;
  wrapperProps?: { css?: SerializedStyles } & React.HTMLAttributes<HTMLDivElement>;
  withChevronIcon?: boolean;
  withInlineLabel?: boolean;
  isBare?: boolean;
  // This does not affect the selected values only the format they are shown in inside the trigger
  renderDisplayedValue?: (value: string) => string | ReactNode;
}

const getTriggerWrapperStyles = (
  theme: Theme,
  clsPrefix: string,
  removable: boolean,
  width?: number | string,
  useNewFormUISpacing?: boolean,
): SerializedStyles =>
  css(
    importantify({
      display: 'inline-flex',
      alignItems: 'center',

      ...(useNewFormUISpacing && {
        [`& + .${clsPrefix}-form-message`]: {
          marginTop: theme.spacing.sm,
        },
      }),

      ...(width && {
        width: width,
      }),

      ...(removable && {
        '& > button:last-of-type': importantify({
          borderBottomLeftRadius: 0,
          borderTopLeftRadius: 0,
          marginLeft: -1,
        }),
      }),
    }),
  );

const getTriggerStyles = (
  theme: Theme,
  disabled: boolean = false,
  maxWidth: number | string,
  minWidth: number | string,
  removable: boolean,
  width?: number | string,
  validationState?: ValidationState,
  isBare?: boolean,
  isSelect?: boolean,
  useNewShadows?: boolean,
): SerializedStyles => {
  const removeButtonInteractionStyles = {
    ...(removable && {
      zIndex: theme.options.zIndexBase + 2,
      '&& + button': {
        marginLeft: -1,
        zIndex: theme.options.zIndexBase + 1,
      },
    }),
  };

  const validationColor = getValidationStateColor(theme, validationState);

  return css(
    importantify({
      position: 'relative',
      display: 'inline-flex',
      alignItems: 'center',
      maxWidth,
      minWidth,
      justifyContent: 'flex-start',
      background: 'transparent',
      padding: isBare ? 0 : '6px 8px 6px 12px',
      boxSizing: 'border-box',
      height: isBare ? theme.typography.lineHeightBase : theme.general.heightSm,
      border: isBare ? 'none' : `1px solid ${theme.colors.actionDefaultBorderDefault}`,
      ...(useNewShadows && {
        boxShadow: theme.shadows.xs,
      }),
      borderRadius: 4,
      color: theme.colors.textPrimary,
      lineHeight: theme.typography.lineHeightBase,
      fontSize: theme.typography.fontSizeBase,
      cursor: 'pointer',
      ...(width && {
        width: width,
        // Only set flex: 1 to items with width, otherwise in flex containers the trigger will take up all the space and break current usages that depend on content for width
        flex: 1,
      }),

      ...(removable && {
        borderBottomRightRadius: 0,
        borderTopRightRadius: 0,
        borderRightColor: 'transparent',
      }),

      '&:hover': {
        background: isBare ? 'transparent' : theme.colors.actionDefaultBackgroundHover,
        borderColor: theme.colors.actionDefaultBorderHover,
        ...removeButtonInteractionStyles,
      },
      '&:focus': {
        borderColor: theme.colors.actionDefaultBorderFocus,
        ...removeButtonInteractionStyles,
      },

      ...(validationState && {
        borderColor: validationColor,

        '&:hover': {
          borderColor: validationColor,
        },

        '&:focus': {
          outlineColor: validationColor,
          outlineOffset: -2,
        },
      }),

      ...(isSelect &&
        !disabled && {
          '&&, &&:hover, &&:focus': {
            background: 'transparent',
          },

          '&&:hover': {
            borderColor: theme.colors.actionDefaultBorderHover,
          },

          '&&:focus, &[data-state="open"]': {
            outlineColor: theme.colors.actionDefaultBorderFocus,
            outlineWidth: 2,
            outlineOffset: -2,
            outlineStyle: 'solid',
            borderColor: 'transparent',
            ...(!useNewShadows && {
              boxShadow: 'none',
            }),
          },
        }),

      [`&[disabled]`]: {
        background: theme.colors.actionDisabledBackground,
        color: theme.colors.actionDisabledText,
        pointerEvents: 'none',
        userSelect: 'none',
        borderColor: theme.colors.actionDisabledBorder,
      },
    }),
  );
};

export const DialogComboboxTrigger = forwardRef<HTMLButtonElement, DialogComboboxTriggerProps>(
  (
    {
      removable = false,
      onRemove,
      children,
      minWidth = 0,
      maxWidth = 9999,
      showTagAfterValueCount = 3,
      allowClear = true,
      controlled,
      onClear,
      wrapperProps,
      width,
      withChevronIcon = true,
      validationState,
      withInlineLabel = true,
      placeholder,
      id: legacyId,
      isBare = false,
      renderDisplayedValue: formatDisplayedValue = (value: string) => value,
      ...restProps
    },
    forwardedRef,
  ) => {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const { label, id: topLevelId, value, isInsideDialogCombobox, multiSelect, setValue } = useDialogComboboxContext();
    const { isSelect, placeholder: selectPlaceholder } = useSelectContext();
    const { useNewShadows, useNewFormUISpacing } = useDesignSystemSafexFlags();

    const id = topLevelId ?? legacyId;

    if (!isInsideDialogCombobox) {
      throw new Error('`DialogComboboxTrigger` must be used within `DialogCombobox`');
    }

    const handleRemove = () => {
      if (!onRemove) {
        // eslint-disable-next-line no-console -- TODO(FEINF-3587)
        console.warn('DialogCombobox.Trigger: Attempted remove without providing onRemove handler');
      } else {
        onRemove();
      }
    };

    const handleClear = (e: any) => {
      e.stopPropagation();

      if (controlled) {
        setValue([]);
        onClear?.();
      } else if (!onClear) {
        // eslint-disable-next-line no-console -- TODO(FEINF-3587)
        console.warn('DialogCombobox.Trigger: Attempted clear without providing onClear handler');
      } else {
        onClear();
      }
    };

    const [showTooltip, setShowTooltip] = React.useState<boolean>();

    const triggerContentRef = React.useRef<HTMLSpanElement>(null);

    useEffect(() => {
      if (value?.length > showTagAfterValueCount) {
        setShowTooltip(true);
      } else if (triggerContentRef.current) {
        const { clientWidth, scrollWidth } = triggerContentRef.current;
        setShowTooltip(clientWidth < scrollWidth);
      }
    }, [showTagAfterValueCount, value]);

    const renderFormattedValue = (v: string, index: number) => {
      const formattedValue = formatDisplayedValue(v);
      return (
        <React.Fragment key={index}>
          {index > 0 && ', '}
          {typeof formattedValue === 'string' ? formattedValue : <span>{formattedValue}</span>}
        </React.Fragment>
      );
    };

    const getStringValue = (v: string) => {
      const formattedValue = formatDisplayedValue(v);
      return typeof formattedValue === 'string' ? formattedValue : v;
    };

    const numValues = Array.isArray(value) ? value.length : 1;
    const concatenatedValues = Array.isArray(value) ? (
      <>
        {value.slice(0, numValues > 10 ? 10 : undefined).map(renderFormattedValue)}
        {numValues > 10 && ` + ${numValues - 10}`}
      </>
    ) : (
      renderFormattedValue(value, 0)
    );

    const displayedValues = <span>{concatenatedValues}</span>;

    const valuesBeforeBadge = Array.isArray(value) ? (
      <>{value.slice(0, showTagAfterValueCount).map(renderFormattedValue)}</>
    ) : (
      renderFormattedValue(value, 0)
    );

    let ariaLabel = '';

    if (!isSelect && !id && label) {
      ariaLabel = React.isValidElement(label) ? 'Dialog Combobox' : `${label}`;

      if (value?.length) {
        const stringValues = Array.isArray(value) ? value.map(getStringValue).join(', ') : getStringValue(value);
        ariaLabel += multiSelect
          ? `, multiselectable, ${value.length} options selected: ${stringValues}`
          : `, selected option: ${stringValues}`;
      } else {
        ariaLabel += multiSelect ? ', multiselectable, 0 options selected' : ', no option selected';
      }
    } else if (isSelect) {
      ariaLabel = ((typeof label === 'string' ? label : '') || restProps['aria-label']) ?? '';
    }

    const customSelectContent = isSelect && children ? children : null;
    const dialogComboboxClassname = !isSelect ? `${classNamePrefix}-dialogcombobox` : '';
    const selectV2Classname = isSelect ? `${classNamePrefix}-selectv2` : '';

    const triggerContent = isSelect ? (
      <Popover.Trigger
        {...(ariaLabel && { 'aria-label': ariaLabel })}
        ref={forwardedRef}
        role="combobox"
        aria-haspopup="listbox"
        aria-invalid={validationState === 'error'}
        id={id}
        {...restProps}
        css={getTriggerStyles(
          theme,
          restProps.disabled,
          maxWidth,
          minWidth,
          removable,
          width,
          validationState,
          isBare,
          isSelect,
          useNewShadows,
        )}
      >
        {/* Using inline styles to not override styles used with custom content */}
        <span
          css={{
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            height: theme.typography.lineHeightBase,
            marginRight: 'auto',
          }}
          ref={triggerContentRef}
        >
          {value?.length ? (
            customSelectContent ?? displayedValues
          ) : (
            <span css={{ color: theme.colors.textPlaceholder }}>{selectPlaceholder}</span>
          )}
        </span>
        {allowClear && value?.length ? <ClearSelectionButton onClick={handleClear} /> : null}
        <ChevronDownIcon css={{ color: theme.colors.textSecondary, marginLeft: theme.spacing.xs }} />
      </Popover.Trigger>
    ) : (
      <Popover.Trigger
        id={id}
        {...(ariaLabel && { 'aria-label': ariaLabel })}
        ref={forwardedRef}
        role="combobox"
        aria-haspopup="listbox"
        aria-invalid={validationState === 'error'}
        {...restProps}
        css={getTriggerStyles(
          theme,
          restProps.disabled,
          maxWidth,
          minWidth,
          removable,
          width,
          validationState,
          isBare,
          isSelect,
          useNewShadows,
        )}
      >
        {/* Using inline styles to not override styles used with custom content */}
        <span
          css={{
            display: 'flex',
            alignItems: 'center',
            height: theme.typography.lineHeightBase,
            marginRight: 'auto',

            '&, & > *': {
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            },
          }}
          ref={triggerContentRef}
        >
          {withInlineLabel ? (
            <span
              css={{
                height: theme.typography.lineHeightBase,
                marginRight: theme.spacing.xs,
                whiteSpace: 'unset',
                overflow: 'unset',
                textOverflow: 'unset',
              }}
            >
              {label}
              {value?.length ? ':' : null}
            </span>
          ) : (
            !value?.length && <span css={{ color: theme.colors.textPlaceholder }}>{placeholder}</span>
          )}
          {value?.length > showTagAfterValueCount ? (
            <>
              <span style={{ marginRight: theme.spacing.xs }}>{valuesBeforeBadge}</span>
              <DialogComboboxCountBadge
                countStartAt={showTagAfterValueCount}
                role="status"
                aria-label="Selected options count"
              />
            </>
          ) : (
            displayedValues
          )}
        </span>
        {allowClear && value?.length ? <ClearSelectionButton onClick={handleClear} /> : null}
        {withChevronIcon ? (
          <ChevronDownIcon
            css={{
              color: theme.colors.textSecondary,
              justifySelf: 'flex-end',
              marginLeft: theme.spacing.xs,
            }}
          />
        ) : null}
      </Popover.Trigger>
    );

    const dataComponentProps = useComponentFinderContext(DesignSystemEventProviderComponentTypes.DialogCombobox);

    return (
      <div
        {...wrapperProps}
        className={`${restProps?.className ?? ''} ${dialogComboboxClassname} ${selectV2Classname}`.trim()}
        css={[
          getTriggerWrapperStyles(theme, classNamePrefix, removable, width, useNewFormUISpacing),
          wrapperProps?.css,
        ]}
        {...addDebugOutlineIfEnabled()}
        {...dataComponentProps}
      >
        {showTooltip && value?.length ? (
          <LegacyTooltip title={customSelectContent ?? displayedValues}>{triggerContent}</LegacyTooltip>
        ) : (
          triggerContent
        )}
        {removable && (
          <Button
            componentId="codegen_design-system_src_design-system_dialogcombobox_dialogcomboboxtrigger.tsx_355"
            aria-label={`Remove ${label}`}
            onClick={handleRemove}
            dangerouslySetForceIconStyles
          >
            <CloseIcon aria-label={`Remove ${label}`} aria-hidden="false" />
          </Button>
        )}
      </div>
    );
  },
);

interface DialogComboboxIconButtonTriggerProps extends Popover.PopoverTriggerProps {}

/**
 * A custom button trigger that can be wrapped around any button.
 */
export const DialogComboboxCustomButtonTriggerWrapper = ({ children }: DialogComboboxIconButtonTriggerProps) => {
  return <Popover.Trigger asChild>{children}</Popover.Trigger>;
};
