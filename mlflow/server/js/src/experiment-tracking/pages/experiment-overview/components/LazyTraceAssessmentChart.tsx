import React from 'react';
import { LegacySkeleton } from '@databricks/design-system';
import type { TraceAssessmentChartProps } from './TraceAssessmentChart';

const TraceAssessmentChart = React.lazy(() =>
  import('./TraceAssessmentChart').then((module) => ({ default: module.TraceAssessmentChart })),
);

export const LazyTraceAssessmentChart: React.FC<TraceAssessmentChartProps> = (props) => (
  <React.Suspense fallback={<LegacySkeleton active />}>
    <TraceAssessmentChart {...props} />
  </React.Suspense>
);
