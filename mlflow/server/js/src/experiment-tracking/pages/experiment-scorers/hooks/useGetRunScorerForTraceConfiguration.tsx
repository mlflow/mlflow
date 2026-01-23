import { useCallback, useMemo, useState } from 'react';
import { ModelTraceExplorerRunJudgeConfig } from '../../../../shared/web-shared/model-trace-explorer';
import { LLMScorer } from '../types';
import { useGetScheduledScorers } from './useGetScheduledScorers';
import { useExperimentIds } from '../../../components/experiment-page/hooks/useExperimentIds';
import { FormattedMessage } from 'react-intl';
import {
  Button,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxCustomButtonTriggerWrapper,
  DialogComboboxFooter,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  getShadowScrollStyles,
  PlusIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { PillControl } from '@databricks/design-system/development';
import ScorerModalRenderer from '../ScorerModalRenderer';
import { SCORER_FORM_MODE } from '../constants';
import { useRunSerializedScorer } from './useRunSerializedScorer';

export const useRunScorerInTracesViewConfiguration = (): ModelTraceExplorerRunJudgeConfig => {
  const [experimentId] = useExperimentIds();

  const { evaluateTraces } = useRunSerializedScorer({ experimentId });

  const renderRunJudgeButton = useCallback<NonNullable<ModelTraceExplorerRunJudgeConfig['renderRunJudgeButton']>>(
    ({ traceId, trigger }) => {
      return (
        <SelectJudgeDropdown traceId={traceId} evaluateTraces={evaluateTraces}>
          {trigger}
        </SelectJudgeDropdown>
      );
    },
    [evaluateTraces],
  );
  return {
    renderRunJudgeButton,
  };
};

/**
 * Dropdown for selecting a judge to run against a trace.
 */
const SelectJudgeDropdown = ({
  traceId,
  onScorerStarted,
  evaluateTraces,
  children,
}: {
  traceId: string;
  onScorerStarted?: (scorerName: string) => void;
  evaluateTraces: (scorer: LLMScorer, traceIds: string[]) => void;
  children: React.ReactNode;
}) => {
  const [experimentId] = useExperimentIds();
  const { data } = useGetScheduledScorers(experimentId);

  const { theme } = useDesignSystemTheme();
  const [searchValue, setSearchValue] = useState<string>('');
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [isCreateScorerModalVisible, setIsCreateScorerModalVisible] = useState(false);

  const displayedLLMScorers = useMemo(() => {
    return data?.scheduledScorers.filter(
      (scorer) => scorer.type === 'llm' && scorer.name.toLowerCase().includes(searchValue.toLowerCase()),
    ) as LLMScorer[];
  }, [data?.scheduledScorers, searchValue]);

  const handleLLMScorerClick = (scorer: LLMScorer) => {
    evaluateTraces(scorer, [traceId]);
    setDropdownOpen(false);
    onScorerStarted?.(scorer.name);
  };

  return (
    <>
      <DialogCombobox
        id="traces-view-judge-select-dropdown"
        componentId="mlflow.experiment-scorers.traces-view-judge-select-dropdown"
        open={dropdownOpen}
        onOpenChange={setDropdownOpen}
      >
        <DialogComboboxCustomButtonTriggerWrapper asChild>{children}</DialogComboboxCustomButtonTriggerWrapper>
        <DialogComboboxContent align="end" css={{ minWidth: 400 }}>
          <DialogComboboxOptionList
            css={{
              borderTop: `1px solid ${theme.colors.border}`,
            }}
          >
            <DialogComboboxOptionListSearch controlledValue={searchValue} setControlledValue={setSearchValue}>
              <div css={{ padding: `${theme.spacing.xs}px ${theme.spacing.md}px`, marginBottom: theme.spacing.sm }}>
                <PillControl.Root
                  size="small"
                  componentId="mlflow.experiment-scorers.traces-view-judge-type-filter"
                  value="llm"
                >
                  <PillControl.Item value="llm">
                    <FormattedMessage
                      defaultMessage="Custom LLM-as-a-judge"
                      description="Label for custom LLM judge type filter option"
                    />
                  </PillControl.Item>
                  {/* TODO: Add support for pre-built LLM-as-a-judge when default gateway is available */}
                  <PillControl.Item value="template" disabled>
                    <FormattedMessage
                      defaultMessage="Pre-built LLM-as-a-judge"
                      description="Label for pre-built LLM judge type filter option"
                    />
                  </PillControl.Item>
                  {/* TODO: Add support for custom code judges */}
                  <PillControl.Item value="custom-code" disabled>
                    <FormattedMessage
                      defaultMessage="Custom code"
                      description="Label for custom code judge type filter option"
                    />
                  </PillControl.Item>
                </PillControl.Root>
              </div>
              <div
                css={{
                  height: 240,
                  display: 'flex',
                  flexDirection: 'column',
                  overflowY: 'auto',
                  ...getShadowScrollStyles(theme, { orientation: 'vertical' }),
                }}
              >
                {displayedLLMScorers?.map((scorer) => (
                  <ScorerOption scorer={scorer} key={scorer.name} onClick={handleLLMScorerClick} />
                ))}
              </div>
            </DialogComboboxOptionListSearch>
          </DialogComboboxOptionList>
          <DialogComboboxFooter css={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              type="tertiary"
              componentId="mlflow.experiment-scorers.traces-view-create-judge"
              icon={<PlusIcon />}
              css={{ flex: 1 }}
              onClick={() => setIsCreateScorerModalVisible(true)}
            >
              {/* TODO: Add support for creating judges ad-hoc in traces view */}
              <FormattedMessage defaultMessage="Create judge" description="Button to create a new judge" />
            </Button>
          </DialogComboboxFooter>
        </DialogComboboxContent>
      </DialogCombobox>
      {isCreateScorerModalVisible && (
        <ScorerModalRenderer
          visible
          onClose={() => setIsCreateScorerModalVisible(false)}
          experimentId={experimentId}
          mode={SCORER_FORM_MODE.CREATE}
          initialScorerType="llm"
        />
      )}
    </>
  );
};

const ScorerOption = ({ scorer, onClick }: { scorer: LLMScorer; onClick: (scorer: LLMScorer) => void }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <DialogComboboxOptionListSelectItem
      value={scorer.name}
      css={{
        alignItems: 'center',
        height: theme.general.buttonHeight,
        '&>label': { flex: 1 },
        flexShrink: 0,
      }}
      key={scorer.name}
      onChange={() => onClick(scorer)}
    >
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <Typography.Text css={{ flex: 1 }}>{scorer.name}</Typography.Text>
        <Typography.Hint>
          <FormattedMessage defaultMessage="Custom judge" description="Label indicating a custom judge scorer" />
        </Typography.Hint>
      </div>
    </DialogComboboxOptionListSelectItem>
  );
};
