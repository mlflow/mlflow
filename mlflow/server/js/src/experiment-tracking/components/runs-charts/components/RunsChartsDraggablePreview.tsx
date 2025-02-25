import { useDesignSystemTheme } from '@databricks/design-system';
import { RUNS_CHARTS_UI_Z_INDEX } from '../utils/runsCharts.const';

export const RunsChartsDraggablePreview = ({
  x,
  y,
  width,
  height,
}: {
  x?: number;
  y?: number;
  width?: number | string;
  height?: number | string;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <>
      {/* Cover pointer events */}
      <div
        css={{
          position: 'absolute',
          inset: 0,
        }}
      />
      <div
        css={{
          position: 'absolute',
          backgroundColor: theme.colors.actionDefaultBackgroundHover,
          borderStyle: 'dashed',
          borderColor: theme.colors.actionDefaultBorderDefault,
          pointerEvents: 'none',
          borderRadius: theme.general.borderRadiusBase,
          borderWidth: 2,
          inset: 0,
          // Make sure the preview is above other cards
          zIndex: RUNS_CHARTS_UI_Z_INDEX.CARD_PREVIEW,
        }}
        style={{
          transform: `translate3d(${x}px, ${y}px, 0)`,
          width: width,
          height: height,
        }}
      />
    </>
  );
};
