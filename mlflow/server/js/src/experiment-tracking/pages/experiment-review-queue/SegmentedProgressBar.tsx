import { useDesignSystemTheme } from '@databricks/design-system';

export type ProgressBarItem = {
  color: string;
};

/**
 * A horizontal row of equal-width segments, one per item, each colored to show
 * its state. Ported from the universe review app's segmented progress bar
 * (genai/shared/eval/components/SegmentedProgressBar) — a coarser, more visible
 * progress indicator than a single filled bar. The height/width come from the
 * caller via `className` (emotion `css`).
 */
export const SegmentedProgressBar = ({ items, className }: { items: ProgressBarItem[]; className?: string }) => {
  const { theme } = useDesignSystemTheme();
  const radius = theme.borders.borderRadiusMd;

  if (items.length === 0) {
    return null;
  }

  return (
    // `gap` handles the hairline spacing between segments (only between, never
    // before the first or after the last), and `flex: 1` lets the segments
    // share the row equally. This keeps the total width at exactly 100% — unlike
    // width% + right margin, which sums past 100% and can overflow.
    <div
      css={{ display: 'flex', alignItems: 'center', height: '100%', gap: theme.spacing.xs / 2 }}
      className={className}
    >
      {items.map((item, index) => (
        <div
          key={index}
          css={{
            flex: 1,
            height: '100%',
            backgroundColor: item.color,
            transition: 'background-color 0.4s ease',
            borderTopLeftRadius: index === 0 ? radius : 0,
            borderBottomLeftRadius: index === 0 ? radius : 0,
            borderTopRightRadius: index === items.length - 1 ? radius : 0,
            borderBottomRightRadius: index === items.length - 1 ? radius : 0,
          }}
        />
      ))}
    </div>
  );
};
