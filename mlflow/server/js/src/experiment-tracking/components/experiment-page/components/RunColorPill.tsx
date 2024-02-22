// This media query applies to screens with a pixel density of 2 or higher
// and higher resolution values (e.g. Retina displays). 192 dpi is double the "default" historical 96 dpi.
const HIGH_RESOLUTION_MEDIA_QUERY = '@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi)';

/**
 * Renders a colored rounded pill for a run.
 */
export const RunColorPill = ({ color, ...props }: { color?: string }) => (
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
    }}
    style={{ backgroundColor: color ?? 'transparent' }}
    {...props}
  />
);
