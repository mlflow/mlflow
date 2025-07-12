// This media query applies to screens with a pixel density of 2 or higher

import { debounce } from 'lodash';
import { useMemo, useState } from 'react';
import { COLORS_PALETTE_DATALIST_ID } from '../../../../common/components/ColorsPaletteDatalist';
import { visuallyHidden } from '@databricks/design-system';

// and higher resolution values (e.g. Retina displays). 192 dpi is double the "default" historical 96 dpi.
const HIGH_RESOLUTION_MEDIA_QUERY = '@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi)';

const stripedHiddenBackgroundStyle = `repeating-linear-gradient(
  135deg,
  #959595 0,
  #e7e7e7 1px,
  #e7e7e7 2px,
  #959595 3px,
  #e7e7e7 4px,
  #e7e7e7 5px,
  #959595 6px,
  #e7e7e7 7px,
  #e7e7e7 8px,
  #959595 9px,
  #e7e7e7 10px,
  #e7e7e7 11px,
  #959595 12px,
  #e7e7e7 13px,
  #e7e7e7 14px
)`;

/**
 * Renders a colored rounded pill for a run.
 */
export const RunColorPill = ({
  color,
  hidden,
  onChangeColor,
  ...props
}: {
  color?: string;
  hidden?: boolean;
  onChangeColor?: (colorValue: string) => void;
}) => {
  const [colorValue, setColorValue] = useState<string | undefined>(undefined);

  const onChangeColorDebounced = useMemo(() => {
    // Implementations of <input type="color"> vary from browser to browser, some browser
    // fire an event on every color change so we debounce the event to avoid multiple
    // calls to the onChangeColor handler.
    if (onChangeColor) {
      return debounce(onChangeColor, 300);
    }
    return () => {};
  }, [onChangeColor]);

  return (
    <label
      css={{
        boxSizing: 'border-box',
        width: 12,
        height: 12,
        borderRadius: 6,
        flexShrink: 0,
        // Add a border to make the pill visible when using very light color
        border: `1px solid ${hidden ? 'transparent' : 'rgba(0,0,0,0.1)'}`,
        // Straighten it up on high-res screens
        [HIGH_RESOLUTION_MEDIA_QUERY]: {
          marginBottom: 1,
        },
        background: hidden ? stripedHiddenBackgroundStyle : undefined,
        cursor: onChangeColor ? 'pointer' : 'default',
        position: 'relative',
        '&:hover': {
          opacity: onChangeColor ? 0.8 : 1,
        },
      }}
      style={{ backgroundColor: colorValue ?? color ?? 'transparent' }}
      {...props}
    >
      <span
        css={[
          visuallyHidden,
          {
            userSelect: 'none',
          },
        ]}
      >
        {color}
      </span>
      {onChangeColor && (
        <input
          disabled={hidden}
          type="color"
          value={colorValue ?? color}
          onChange={({ target }) => {
            setColorValue(target.value);
            onChangeColorDebounced(target.value);
          }}
          list={COLORS_PALETTE_DATALIST_ID}
          css={{
            appearance: 'none',
            width: 0,
            height: 0,
            border: 0,
            padding: 0,
            position: 'absolute',
            bottom: 0,
            visibility: 'hidden',
          }}
        />
      )}
    </label>
  );
};
