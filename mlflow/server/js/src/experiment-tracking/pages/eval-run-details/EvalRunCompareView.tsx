import { useDesignSystemTheme } from '@databricks/design-system';
import { EvalRunCompareSelectionBar } from './EvalRunCompareSelectionBar';
import { EvalRunComparisonOverviewTab } from './EvalRunComparisonOverviewTab';
import { useEvalRunComparisonBaseline } from './useEvalRunComparisonBaseline';

export interface EvalRunCompareViewProps {
  experimentId: string;
  /** The run the comparison is anchored on (the detail page's path runUuid). */
  baselineRunUuid: string;
  /** Other runs being compared against the baseline; empty until the user adds some. */
  compareRunUuids: string[];
}

/**
 * Compare surface rendered on the eval-run detail page when compare mode is on. The run
 * selection bar sits above the results so every view of the comparison shares one selection.
 */
export const EvalRunCompareView = ({ experimentId, baselineRunUuid, compareRunUuids }: EvalRunCompareViewProps) => {
  const { theme } = useDesignSystemTheme();
  const onSelectBaseline = useEvalRunComparisonBaseline(experimentId, baselineRunUuid, compareRunUuids);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        gap: theme.spacing.md,
      }}
    >
      <EvalRunCompareSelectionBar
        experimentId={experimentId}
        baselineRunUuid={baselineRunUuid}
        compareRunUuids={compareRunUuids}
        onSelectBaseline={onSelectBaseline}
      />
      <EvalRunComparisonOverviewTab
        experimentId={experimentId}
        baselineRunUuid={baselineRunUuid}
        compareRunUuids={compareRunUuids}
      />
    </div>
  );
};
