// This media query applies to screens with a pixel density of 2 or higher
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
export const RunColorPill = ({ color, hidden, ...props }: { color?: string; hidden?: boolean }) => (
  <div
    css={{
      width: 12,
      height: 12,
      borderRadius: 6,
      flexShrink: 0,
      // Straighten it up on high-res screens
      [HIGH_RESOLUTION_MEDIA_QUERY]: {
        marginBottom: 1,
      },
      background: hidden ? stripedHiddenBackgroundStyle : undefined,
    }}
    style={{ backgroundColor: color ?? 'transparent' }}
    {...props}
  />
);
