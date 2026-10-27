import {
  Button,
  ChevronDownIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxCustomButtonTriggerWrapper,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  PlusIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useGenAiExperimentRunsForComparison } from '@databricks/web-shared/genai-traces-table';
import { useEffect, useMemo, useState } from 'react';
import { useIntl } from 'react-intl';
import { useEvalRunsCompareMode } from '../../components/evaluations/hooks/useEvalRunsCompareMode';

interface ComparisonRunOption {
  runUuid: string;
  runName: string;
  startTime?: number;
}

const COMPARISON_RUN_PICKER_WIDTH = 320;

export interface EvalRunComparisonsSectionProps {
  experimentId: string;
  runUuid: string;
  runName: string;
}

export const EvalRunComparisonsSection = ({ experimentId, runUuid, runName }: EvalRunComparisonsSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { compareRuns, startComparison } = useEvalRunsCompareMode();
  const { runInfos, isLoading, requestError } = useGenAiExperimentRunsForComparison(experimentId);

  const selectedFromUrl = useMemo(
    () => Array.from(new Set(compareRuns.filter((uuid) => uuid !== runUuid))),
    [compareRuns, runUuid],
  );
  const [selectedRunUuids, setSelectedRunUuids] = useState(selectedFromUrl);

  useEffect(() => {
    setSelectedRunUuids(selectedFromUrl);
  }, [selectedFromUrl]);

  const runOptions = useMemo(() => {
    const byUuid = new Map<string, ComparisonRunOption>();
    (runInfos ?? []).forEach((info) => {
      if (info.runUuid && info.runUuid !== runUuid) {
        byUuid.set(info.runUuid, {
          runUuid: info.runUuid,
          runName: info.runName || info.runUuid,
          startTime: info.startTime,
        });
      }
    });

    // Search results can omit URL-selected runs. Keep those visible so they can be cleared or retained.
    selectedRunUuids.forEach((uuid) => {
      if (uuid !== runUuid && !byUuid.has(uuid)) {
        byUuid.set(uuid, { runUuid: uuid, runName: uuid });
      }
    });

    return Array.from(byUuid.values());
  }, [runInfos, runUuid, selectedRunUuids]);

  const currentRunInfo = runInfos?.find((info) => info.runUuid === runUuid);
  const currentRunName = runName || currentRunInfo?.runName || runUuid;

  const formatCreatedDate = (startTime?: number) =>
    startTime !== undefined && Number.isFinite(startTime)
      ? intl.formatDate(startTime, { dateStyle: 'medium', timeStyle: 'short' })
      : intl.formatMessage({
          defaultMessage: 'Date unavailable',
          description: 'Fallback for an evaluation run whose creation date is unavailable',
        });

  const formatCreatedLabel = (startTime?: number) =>
    intl.formatMessage(
      {
        defaultMessage: 'Created {date}',
        description: 'Creation date shown below an evaluation run name in the comparison picker',
      },
      { date: formatCreatedDate(startTime) },
    );

  const toggleRunSelection = (runUuidToToggle: string) => {
    setSelectedRunUuids((current) =>
      current.includes(runUuidToToggle)
        ? current.filter((uuid) => uuid !== runUuidToToggle)
        : [...current, runUuidToToggle],
    );
  };

  const handleCompare = () => {
    const comparisonRunUuids = Array.from(new Set(selectedRunUuids.filter((uuid) => uuid !== runUuid)));
    startComparison(comparisonRunUuids);
  };

  const selectRunsLabel =
    selectedRunUuids.length === 0
      ? intl.formatMessage({
          defaultMessage: 'Select runs to compare',
          description: 'Trigger label for selecting evaluation runs to compare',
        })
      : intl.formatMessage(
          {
            defaultMessage: '{count, plural, one {# run selected} other {# runs selected}}',
            description: 'Trigger label showing how many evaluation runs are selected for comparison',
          },
          { count: selectedRunUuids.length },
        );

  return (
    <section
      aria-label={intl.formatMessage({
        defaultMessage: 'Comparisons',
        description: 'Evaluation run sidebar section for selecting runs to compare',
      })}
      css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
          <div css={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <Typography.Text bold ellipsis>
              {currentRunName}
            </Typography.Text>
            <Typography.Hint size="sm">{formatCreatedLabel(currentRunInfo?.startTime)}</Typography.Hint>
          </div>
        </div>
      </div>

      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
          <div css={{ flex: '1 1 auto', minWidth: 0, maxWidth: COMPARISON_RUN_PICKER_WIDTH }}>
            <DialogCombobox
              componentId="mlflow.eval_run_details.comparisons.run_selector"
              id="eval-run-comparisons-run-selector"
              value={selectedRunUuids}
              multiSelect
            >
              <DialogComboboxCustomButtonTriggerWrapper>
                <Button
                  componentId="mlflow.eval_run_details.comparisons.run_selector_trigger"
                  size="small"
                  icon={<PlusIcon />}
                  endIcon={<ChevronDownIcon />}
                  disabled={isLoading || runOptions.length === 0}
                  css={{ width: '100%', justifyContent: 'space-between' }}
                >
                  <span css={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {selectRunsLabel}
                  </span>
                </Button>
              </DialogComboboxCustomButtonTriggerWrapper>
              <DialogComboboxContent width={COMPARISON_RUN_PICKER_WIDTH}>
                <DialogComboboxOptionList>
                  {runOptions.map((option) => (
                    <DialogComboboxOptionListCheckboxItem
                      key={option.runUuid}
                      value={option.runUuid}
                      checked={selectedRunUuids.includes(option.runUuid)}
                      onChange={() => toggleRunSelection(option.runUuid)}
                    >
                      <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
                        <span css={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
                          <Typography.Text ellipsis title={option.runName}>
                            {option.runName}
                          </Typography.Text>
                          <Typography.Hint size="sm" css={{ whiteSpace: 'nowrap' }}>
                            {formatCreatedLabel(option.startTime)}
                          </Typography.Hint>
                        </span>
                      </span>
                    </DialogComboboxOptionListCheckboxItem>
                  ))}
                </DialogComboboxOptionList>
              </DialogComboboxContent>
            </DialogCombobox>
          </div>

          <Button
            componentId="mlflow.eval_run_details.comparisons.compare"
            type="primary"
            size="small"
            css={{ flexShrink: 0 }}
            disabled={selectedRunUuids.length === 0}
            onClick={handleCompare}
          >
            {intl.formatMessage({
              defaultMessage: 'Compare',
              description: 'Action that opens the evaluation run comparison view',
            })}
          </Button>
        </div>
        {requestError && (
          <div role="status">
            <Typography.Text color="secondary">
              {intl.formatMessage({
                defaultMessage: 'Could not load runs for comparison.',
                description: 'Message shown when the eval-run comparison picker cannot load runs',
              })}
            </Typography.Text>
          </div>
        )}
      </div>
    </section>
  );
};
