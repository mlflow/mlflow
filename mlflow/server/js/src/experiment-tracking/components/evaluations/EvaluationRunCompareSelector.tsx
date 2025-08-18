import {
  Typography,
  useDesignSystemTheme,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxCustomButtonTriggerWrapper,
  Button,
  ChevronDownIcon,
  XCircleFillIcon,
} from '@databricks/design-system';
import { useCallback, useMemo } from 'react';
import { useGetExperimentRunColor } from '../experiment-page/hooks/useExperimentRunColor';
import { useIntl } from '@databricks/i18n';
import Routes from '../../routes';
import {
  useGenAiExperimentRunsForComparison,
  COMPARE_TO_RUN_DROPDOWN_COMPONENT_ID,
} from '@databricks/web-shared/genai-traces-table';
import { RunColorPill } from '../experiment-page/components/RunColorPill';

export const EvaluationRunCompareSelector = ({
  experimentId,
  currentRunUuid,
  setCompareToRunUuid,
  compareToRunUuid,
  setCurrentRunUuid: setCurrentRunUuidProp,
}: {
  experimentId: string;
  currentRunUuid?: string;
  setCompareToRunUuid?: (runUuid: string | undefined) => void;
  compareToRunUuid?: string;
  // used in evaluation runs tab
  setCurrentRunUuid?: (runUuid: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const getRunColor = useGetExperimentRunColor();

  const { runInfos: experimentRunInfos } = useGenAiExperimentRunsForComparison(experimentId);

  const currentRunOptions = useMemo(() => {
    if (!experimentRunInfos) return undefined;
    return experimentRunInfos
      .map((runInfo) => ({
        key: runInfo.runUuid,
        value: runInfo.runName ?? runInfo.runUuid,
      }))
      .filter((result) => result.key) as {
      key: string;
      value: string;
    }[];
  }, [experimentRunInfos]);

  const compareToRunOptions = useMemo(() => {
    if (!experimentRunInfos) return undefined;
    return experimentRunInfos
      .filter((runInfo) => runInfo.runUuid !== currentRunUuid)
      .map((runInfo) => ({
        key: runInfo.runUuid,
        value: runInfo.runName ?? runInfo.runUuid,
      }))
      .filter((result) => Boolean(result.key)) as {
      key: string;
      value: string;
    }[];
  }, [experimentRunInfos, currentRunUuid]);

  const currentRunInfo = experimentRunInfos?.find((runInfo) => runInfo.runUuid === currentRunUuid);
  const compareToRunInfo = experimentRunInfos?.find((runInfo) => runInfo.runUuid === compareToRunUuid);

  const defaultSetCurrentRunUuid = useCallback(
    (runUuid: string) => {
      const link = Routes.getRunPageRoute(experimentId, runUuid) + '/evaluations';
      // Propagate all the current URL query params.
      const currentParams = new URLSearchParams(window.location.search);

      const newUrl = new URL(link, window.location.origin);
      currentParams.forEach((value, key) => {
        newUrl.searchParams.set(key, value);
      });

      window.location.href = newUrl.toString();
    },
    [experimentId],
  );

  const setCurrentRunUuid = setCurrentRunUuidProp ?? defaultSetCurrentRunUuid;

  if (!currentRunUuid) {
    return <></>;
  }

  return (
    <div
      css={{
        display: 'flex',
        gap: theme.spacing.sm,
        alignItems: 'center',
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'flex-start',
          gap: theme.spacing.sm,
        }}
      >
        <DialogCombobox
          componentId={COMPARE_TO_RUN_DROPDOWN_COMPONENT_ID}
          id="compare-to-run-combobox"
          value={currentRunUuid ? [currentRunUuid] : undefined}
        >
          <DialogComboboxCustomButtonTriggerWrapper>
            <Button
              endIcon={<ChevronDownIcon />}
              componentId="mlflow.evaluations_review.table_ui.compare_to_run_button"
              css={{
                justifyContent: 'flex-start !important',
              }}
            >
              <div
                css={{
                  display: 'flex',
                  gap: theme.spacing.sm,
                  alignItems: 'center',
                  fontSize: `${theme.typography.fontSizeSm}px !important`,
                }}
              >
                <RunColorPill color={getRunColor(currentRunUuid)} />
                {currentRunInfo?.runName ? (
                  <Typography.Hint>{currentRunInfo?.runName}</Typography.Hint>
                ) : (
                  // eslint-disable-next-line formatjs/enforce-description
                  intl.formatMessage({ defaultMessage: 'Select baseline run' })
                )}
              </div>
            </Button>
          </DialogComboboxCustomButtonTriggerWrapper>
          <DialogComboboxContent>
            <DialogComboboxOptionList>
              {(currentRunOptions || []).map((item, i) => (
                <DialogComboboxOptionListSelectItem
                  key={i}
                  value={item.value}
                  onChange={(e) => setCurrentRunUuid(item.key)}
                  checked={item.key === currentRunUuid}
                >
                  <div
                    css={{
                      display: 'flex',
                      gap: theme.spacing.sm,
                      alignItems: 'center',
                    }}
                  >
                    <RunColorPill color={getRunColor(item.key)} />
                    {item.value}
                  </div>
                </DialogComboboxOptionListSelectItem>
              ))}
            </DialogComboboxOptionList>
          </DialogComboboxContent>
        </DialogCombobox>
      </div>
      <span css={{}}>
        {intl.formatMessage({
          defaultMessage: 'compare to',
          description: 'Label for "to" in compare to, which is split between two dropdowns of two runs to compare',
        })}
      </span>
      {setCompareToRunUuid && (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div
            css={{
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'center',
            }}
          >
            <DialogCombobox
              componentId={COMPARE_TO_RUN_DROPDOWN_COMPONENT_ID}
              id="compare-to-run-combobox"
              value={compareToRunUuid ? [compareToRunUuid] : undefined}
            >
              <DialogComboboxCustomButtonTriggerWrapper>
                <Button
                  endIcon={<ChevronDownIcon />}
                  componentId="mlflow.evaluations_review.table_ui.compare_to_run_button"
                  css={{
                    justifyContent: 'flex-start !important',
                  }}
                >
                  <div
                    css={{
                      display: 'flex',
                      gap: theme.spacing.sm,
                      alignItems: 'center',
                      fontSize: `${theme.typography.fontSizeSm}px !important`,
                    }}
                  >
                    {compareToRunInfo?.runName ? (
                      <>
                        <RunColorPill color={getRunColor(compareToRunUuid)} />
                        <Typography.Hint>{compareToRunInfo?.runName}</Typography.Hint>
                      </>
                    ) : (
                      <span
                        css={{
                          color: theme.colors.textPlaceholder,
                        }}
                      >
                        {/* eslint-disable-next-line formatjs/enforce-description */}
                        {intl.formatMessage({ defaultMessage: 'baseline run' })}
                      </span>
                    )}
                  </div>
                </Button>
              </DialogComboboxCustomButtonTriggerWrapper>
              <DialogComboboxContent>
                <DialogComboboxOptionList>
                  {(compareToRunOptions || []).map((item, i) => (
                    <DialogComboboxOptionListSelectItem
                      key={i}
                      value={item.value}
                      onChange={(e) => setCompareToRunUuid(item.key)}
                      checked={item.key === compareToRunUuid}
                    >
                      <div
                        css={{
                          display: 'flex',
                          gap: theme.spacing.sm,
                          alignItems: 'center',
                        }}
                      >
                        <RunColorPill color={getRunColor(item.key)} />
                        {item.value}
                      </div>
                    </DialogComboboxOptionListSelectItem>
                  ))}
                </DialogComboboxOptionList>
              </DialogComboboxContent>
            </DialogCombobox>
            {compareToRunInfo?.runName && (
              <XCircleFillIcon
                aria-hidden="false"
                css={{
                  color: theme.colors.textPlaceholder,
                  fontSize: theme.typography.fontSizeSm,
                  marginLeft: theme.spacing.sm,

                  ':hover': {
                    color: theme.colors.actionTertiaryTextHover,
                  },
                }}
                role="button"
                onClick={() => {
                  setCompareToRunUuid(undefined);
                }}
                onPointerDownCapture={(e) => {
                  // Prevents the dropdown from opening when clearing
                  e.stopPropagation();
                }}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};
