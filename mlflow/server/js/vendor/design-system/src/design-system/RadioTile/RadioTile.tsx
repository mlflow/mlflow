import { css, type SerializedStyles } from '@emotion/react';

import type { Theme } from '../../theme';
import { useDesignSystemTheme } from '../Hooks';
import { Radio, useRadioGroupContext } from '../Radio/Radio';
import type { RadioChangeEvent, RadioProps } from '../Radio/Radio';

export interface RadioTileProps extends RadioProps {
  icon?: React.ReactNode;
  description?: string;
  maxWidth?: string | number;
}

const getRadioTileStyles = (theme: Theme, classNamePrefix: string, maxWidth?: string | number): SerializedStyles => {
  const radioWrapper = `.${classNamePrefix}-radio-wrapper`;

  return css({
    '&&': {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
      padding: theme.spacing.md,
      gap: theme.spacing.xs,
      borderRadius: theme.borders.borderRadiusSm,
      background: 'transparent',
      cursor: 'pointer',
      ...(maxWidth && {
        maxWidth,
      }),

      // Label, radio and icon container
      '& > div:first-of-type': {
        width: '100%',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: theme.spacing.sm,
      },

      // Description container
      '& > div:nth-of-type(2)': {
        alignSelf: 'flex-start',
        textAlign: 'left',
        color: theme.colors.textSecondary,
        fontSize: theme.typography.fontSizeSm,
      },

      '&:hover': {
        backgroundColor: theme.colors.actionDefaultBackgroundHover,
        borderColor: theme.colors.actionDefaultBorderHover,
      },

      '&:disabled': {
        borderColor: theme.colors.actionDisabledBorder,
        backgroundColor: 'transparent',
        cursor: 'not-allowed',

        '& > div:nth-of-type(2)': {
          color: theme.colors.actionDisabledText,
        },
      },
    },

    [radioWrapper]: {
      display: 'flex',
      flexDirection: 'row-reverse',
      justifyContent: 'space-between',
      flex: 1,
      margin: 0,

      '& > span': {
        padding: 0,
      },

      '::after': {
        display: 'none',
      },
    },
  });
};

export const RadioTile = (props: RadioTileProps) => {
  const { description, icon, maxWidth, checked, defaultChecked, onChange, ...rest } = props;
  const { theme, classNamePrefix } = useDesignSystemTheme();
  const { value: groupValue, onChange: groupOnChange } = useRadioGroupContext();

  return (
    <button
      role="radio"
      aria-checked={groupValue === props.value}
      onClick={() => {
        if (props.disabled) {
          return;
        }
        onChange?.(props.value);
        groupOnChange?.({ target: { value: props.value } } as RadioChangeEvent);
      }}
      tabIndex={0}
      className={`${classNamePrefix}-radio-tile`}
      css={getRadioTileStyles(theme, classNamePrefix, maxWidth)}
      disabled={props.disabled}
    >
      <div>
        {icon ? (
          <span css={{ color: props.disabled ? theme.colors.actionDisabledText : theme.colors.textSecondary }}>
            {icon}
          </span>
        ) : null}
        <Radio __INTERNAL_DISABLE_RADIO_ROLE {...rest} tabIndex={-1} />
      </div>
      {description ? <div>{description}</div> : null}
    </button>
  );
};
