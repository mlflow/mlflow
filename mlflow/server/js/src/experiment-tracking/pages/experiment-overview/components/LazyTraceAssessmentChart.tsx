import React from 'react';
import { ContentSkeleton } from '@databricks/design-system';
import type { TraceAssessmentChartProps } from './TraceAssessmentChart';

const TraceAssessmentChart = React.lazy(() =>
  import('./TraceAssessmentChart').then((module) => ({ default: module.TraceAssessmentChart })),
);

export const LazyTraceAssessmentChart: React.FC<TraceAssessmentChartProps> = (props) => (
  <React.Suspense fallback={<ContentSkeleton />}>
    <TraceAssessmentChart {...props} />
  </React.Suspense>
);
