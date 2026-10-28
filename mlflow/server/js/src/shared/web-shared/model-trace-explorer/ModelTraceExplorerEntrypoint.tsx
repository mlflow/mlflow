import { ModelTraceExplorer as LegacyModelTraceExplorer } from './ModelTraceExplorer';
import { ModelTraceExplorer as RedesignedModelTraceExplorer } from './v2/ModelTraceExplorer';
import { shouldEnableRedesignedTraceExplorer } from './shouldEnableRedesignedTraceExplorer';

export type ModelTraceExplorerProps = React.ComponentProps<typeof LegacyModelTraceExplorer> &
  Pick<
    React.ComponentProps<typeof RedesignedModelTraceExplorer>,
    'comparisonModelTrace' | 'comparisonRole' | 'showSpanTree' | 'onShowSpanTreeChange' | 'hideAssessments'
  >;

export const ModelTraceExplorer = (props: ModelTraceExplorerProps): JSX.Element => {
  if (!shouldEnableRedesignedTraceExplorer()) {
    return <LegacyModelTraceExplorer {...props} />;
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        height: '100%',
        minHeight: 0,
        overflow: 'hidden',
      }}
    >
      <RedesignedModelTraceExplorer {...props} />
    </div>
  );
};
