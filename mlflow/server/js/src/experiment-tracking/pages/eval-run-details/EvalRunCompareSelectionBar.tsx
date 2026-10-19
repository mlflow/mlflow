import {
  Button,
  ChevronDownIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxCustomButtonTriggerWrapper,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSelectItem,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';
import { useGenAiExperimentRunsForComparison } from '@databricks/web-shared/genai-traces-table';
import { Fragment, useMemo } from 'react';
import { RunColorPill } from '../../components/experiment-page/components/RunColorPill';
import { useGetExperimentRunColor } from '../../components/experiment-page/hooks/useExperimentRunColor';
import { useEvalRunsCompareMode } from '../../components/evaluations/hooks/useEvalRunsCompareMode';

interface RunOption {
  key: string;
  value: string;
  startTime?: number;
}

const COMPARISON_RUN_PICKER_WIDTH = 360;
const MAX_VISIBLE_COMPARISON_NAMES = 5;

export interface EvalRunCompareSelectionBarProps {
  experimentId: string;
  /** The run the comparison is anchored on (the detail page's path runUuid). */
  baselineRunUuid: string;
  /** Other runs being compared against the baseline; may be empty. */
  compareRunUuids: string[];
  onSelectBaseline: (runUuid: string) => void;
}

/**
 * Interactive run selector shared by the eval-run compare view's Overview and Details tabs.
 * It sits above the tabs (so both see the same selection) and drives which runs are compared:
 * the baseline selector re-anchors the comparison and the comparison selector toggles runs. Only
 * mounted in compare mode, so it never shows in the single-run view.
 */
export const EvalRunCompareSelectionBar = ({
  experimentId,
  baselineRunUuid,
  compareRunUuids,
  onSelectBaseline,
}: EvalRunCompareSelectionBarProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const getRunColor = useGetExperimentRunColor();
  const { setCompareRuns } = useEvalRunsCompareMode();

  const { runInfos } = useGenAiExperimentRunsForComparison(experimentId);

  const allRunOptions = useMemo<RunOption[]>(() => {
    const optionsByUuid = new Map<string, RunOption>();
    (runInfos ?? []).forEach((info) => {
      if (info.runUuid) {
        optionsByUuid.set(info.runUuid, {
          key: info.runUuid,
          value: info.runName ?? info.runUuid,
          startTime: info.startTime,
        });
      }
    });

    // Search results can omit URL-selected runs because of pagination or run visibility.
    [baselineRunUuid, ...compareRunUuids].forEach((runUuid) => {
      if (!optionsByUuid.has(runUuid)) {
        optionsByUuid.set(runUuid, { key: runUuid, value: runUuid });
      }
    });

    return Array.from(optionsByUuid.values());
  }, [baselineRunUuid, compareRunUuids, runInfos]);

  const runNameByUuid = useMemo(
    () => new Map(allRunOptions.map((option) => [option.key, option.value])),
    [allRunOptions],
  );

  const labelFor = (runUuid: string) => runNameByUuid.get(runUuid) ?? runUuid;

  // Re-anchoring only changes which selected run is the baseline; it does not add another run.
  const baselineOptions = useMemo(
    () => allRunOptions.filter((option) => option.key === baselineRunUuid || compareRunUuids.includes(option.key)),
    [allRunOptions, baselineRunUuid, compareRunUuids],
  );

  // A comparison can be any run other than the baseline. Selected runs that are missing from the
  // search response are already included in allRunOptions above so they remain removable.
  const comparisonOptions = useMemo(
    () => allRunOptions.filter((option) => option.key !== baselineRunUuid),
    [allRunOptions, baselineRunUuid],
  );

  const runTriggerStyle = {
    display: 'flex',
    gap: theme.spacing.sm,
    alignItems: 'center',
    flex: 1,
    minWidth: 0,
  } as const;

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

  const renderRunOption = (option: RunOption) => (
    <span
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        minWidth: 0,
      }}
    >
      <span
        aria-hidden="true"
        css={{
          display: 'flex',
          alignSelf: 'flex-start',
          marginTop: theme.spacing.xs,
          flexShrink: 0,
        }}
      >
        <RunColorPill color={getRunColor(option.key)} />
      </span>
      <span css={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
        <Typography.Text ellipsis title={option.value}>
          {option.value}
        </Typography.Text>
        <Typography.Hint size="sm" css={{ whiteSpace: 'nowrap' }}>
          {formatCreatedLabel(option.startTime)}
        </Typography.Hint>
      </span>
    </span>
  );

  const handleToggleComparison = (runUuid: string) => {
    setCompareRuns(
      compareRunUuids.includes(runUuid)
        ? compareRunUuids.filter((uuid) => uuid !== runUuid)
        : [...compareRunUuids, runUuid],
    );
  };

  const comparisonTriggerLabel = intl.formatMessage({
    defaultMessage: 'Select runs to compare',
    description: 'Trigger label for selecting evaluation runs to compare',
  });
  const visibleComparisonRunUuids = compareRunUuids.slice(0, MAX_VISIBLE_COMPARISON_NAMES);
  const hiddenComparisonCount = compareRunUuids.length - visibleComparisonRunUuids.length;

  return (
    <div
      css={{
        display: 'flex',
        flexWrap: 'nowrap',
        alignItems: 'center',
        gap: theme.spacing.sm,
        width: '100%',
        backgroundColor: theme.colors.backgroundSecondary,
        borderRadius: theme.borders.borderRadiusMd,
        padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
      }}
    >
      <Typography.Text bold>
        {intl.formatMessage({
          defaultMessage: 'Compare',
          description: 'Leading label of the eval-run comparison selection bar',
        })}
      </Typography.Text>

      {/* Switching the baseline re-anchors the comparison on the chosen run. */}
      <DialogCombobox
        componentId="mlflow.eval_run_details.compare.selection_bar.baseline"
        id="eval-run-compare-baseline"
        label={intl.formatMessage({
          defaultMessage: 'Baseline run',
          description: 'Accessible name for the baseline run selector',
        })}
        value={[baselineRunUuid]}
      >
        <DialogComboboxCustomButtonTriggerWrapper>
          <Button
            endIcon={<ChevronDownIcon />}
            componentId="mlflow.eval_run_details.compare.selection_bar.baseline_trigger"
            css={{ maxWidth: 320, minWidth: 0, '& .db-btn-content': { minWidth: 0 } }}
          >
            <span css={runTriggerStyle}>
              <Tag
                componentId="mlflow.eval_run_details.compare.selection_bar.baseline_label"
                color="default"
                css={{ margin: 0, flexShrink: 0 }}
              >
                {intl.formatMessage({
                  defaultMessage: 'Base',
                  description: 'Small label shown inside the baseline run selector in eval-run comparison mode',
                })}
              </Tag>
              <span aria-hidden="true" css={{ display: 'inline-flex', flexShrink: 0 }}>
                <RunColorPill color={getRunColor(baselineRunUuid)} />
              </span>
              <Typography.Text style={{ flex: 1, minWidth: 0 }} ellipsis title={labelFor(baselineRunUuid)}>
                {labelFor(baselineRunUuid)}
              </Typography.Text>
            </span>
          </Button>
        </DialogComboboxCustomButtonTriggerWrapper>
        <DialogComboboxContent width={COMPARISON_RUN_PICKER_WIDTH} maxHeight={400}>
          <DialogComboboxOptionList>
            {baselineOptions.map((item) => (
              <DialogComboboxOptionListSelectItem
                key={item.key}
                value={item.key}
                checked={item.key === baselineRunUuid}
                onChange={() => onSelectBaseline(item.key)}
              >
                {renderRunOption(item)}
              </DialogComboboxOptionListSelectItem>
            ))}
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>

      <span css={{ color: theme.colors.textSecondary }}>
        {intl.formatMessage({
          defaultMessage: 'to',
          description: 'Separator between the baseline run and the runs it is compared against',
        })}
      </span>

      <span css={{ flex: '0 1 auto', minWidth: 0, maxWidth: '100%' }}>
        <DialogCombobox
          componentId="mlflow.eval_run_details.compare.selection_bar.comparison"
          id="eval-run-compare-comparison"
          label={intl.formatMessage({
            defaultMessage: 'Comparison runs',
            description: 'Accessible name for the comparison run selector',
          })}
          value={compareRunUuids}
          multiSelect
        >
          <DialogComboboxCustomButtonTriggerWrapper>
            <Button
              endIcon={<ChevronDownIcon />}
              componentId="mlflow.eval_run_details.compare.selection_bar.comparison_trigger"
              disabled={comparisonOptions.length === 0}
              title={compareRunUuids.map(labelFor).join(', ')}
              css={{ maxWidth: '100%', minWidth: 0, '& .db-btn-content': { minWidth: 0 } }}
            >
              <span
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  flex: '1 1 auto',
                  minWidth: 0,
                  overflow: 'hidden',
                  whiteSpace: 'nowrap',
                  textOverflow: 'ellipsis',
                }}
              >
                {compareRunUuids.length === 0 ? (
                  comparisonTriggerLabel
                ) : (
                  <>
                    {visibleComparisonRunUuids.map((runUuid, index) => {
                      const name = labelFor(runUuid);
                      return (
                        <Fragment key={runUuid}>
                          {index > 0 && <span css={{ flexShrink: 0, marginRight: theme.spacing.xs }}>,</span>}
                          <span
                            css={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: theme.spacing.xs,
                              minWidth: 0,
                              flex: index === visibleComparisonRunUuids.length - 1 ? '1 1 auto' : '0 1 auto',
                              overflow: 'hidden',
                            }}
                          >
                            <span aria-hidden="true" css={{ display: 'inline-flex', flexShrink: 0 }}>
                              <RunColorPill color={getRunColor(runUuid)} />
                            </span>
                            <Typography.Text style={{ minWidth: 0, flex: '1 1 auto' }} ellipsis title={name}>
                              {name}
                            </Typography.Text>
                          </span>
                        </Fragment>
                      );
                    })}
                  </>
                )}
              </span>
              {hiddenComparisonCount > 0 && (
                <span css={{ flexShrink: 0, marginLeft: theme.spacing.xs }}>
                  {intl.formatMessage(
                    {
                      defaultMessage: '+{count} more',
                      description: 'Number of additional evaluation runs selected for comparison',
                    },
                    { count: hiddenComparisonCount },
                  )}
                </span>
              )}
            </Button>
          </DialogComboboxCustomButtonTriggerWrapper>
          <DialogComboboxContent width={COMPARISON_RUN_PICKER_WIDTH} maxHeight={400}>
            <DialogComboboxOptionList>
              {comparisonOptions.map((item) => (
                <DialogComboboxOptionListCheckboxItem
                  key={item.key}
                  value={item.key}
                  checked={compareRunUuids.includes(item.key)}
                  onChange={() => handleToggleComparison(item.key)}
                >
                  {renderRunOption(item)}
                </DialogComboboxOptionListCheckboxItem>
              ))}
            </DialogComboboxOptionList>
          </DialogComboboxContent>
        </DialogCombobox>
      </span>
    </div>
  );
};
