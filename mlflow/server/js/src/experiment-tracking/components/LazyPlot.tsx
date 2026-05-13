/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { useMemo } from 'react';
import { LegacySkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { SectionErrorBoundary } from '../../common/components/error-boundaries/SectionErrorBoundary';

const Plot = React.lazy(() => import('react-plotly.js'));

export const LazyPlot = ({ fallback, layout, ...props }: any) => {
  const { theme } = useDesignSystemTheme();

  const themedLayout = useMemo(() => {
    const { xaxis, yaxis, font, ...rest } = layout || {};
    return {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: {
        color: theme.colors.textPrimary,
        ...font,
      },
      xaxis: {
        gridcolor: theme.colors.border,
        zerolinecolor: theme.colors.border,
        ...xaxis,
      },
      yaxis: {
        gridcolor: theme.colors.border,
        zerolinecolor: theme.colors.border,
        ...yaxis,
      },
      ...rest,
    };
  }, [layout, theme]);

  return (
    <SectionErrorBoundary>
      <React.Suspense fallback={fallback ?? <LegacySkeleton active />}>
        <Plot layout={themedLayout} {...props} />
      </React.Suspense>
    </SectionErrorBoundary>
  );
};
