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
    <div css={{ display: 'flex', alignItems: 'center', height: '100%' }} className={className}>
      {items.map((item, index) => (
        <div
          key={index}
          css={{
            width: `${100 / items.length}%`,
            height: '100%',
            backgroundColor: item.color,
            borderTopLeftRadius: index === 0 ? radius : 0,
            borderBottomLeftRadius: index === 0 ? radius : 0,
            borderTopRightRadius: index === items.length - 1 ? radius : 0,
            borderBottomRightRadius: index === items.length - 1 ? radius : 0,
            // Hairline gap between segments (no gap after the last one).
            marginRight: index === items.length - 1 ? 0 : theme.spacing.xs / 2,
          }}
        />
      ))}
    </div>
  );
};
