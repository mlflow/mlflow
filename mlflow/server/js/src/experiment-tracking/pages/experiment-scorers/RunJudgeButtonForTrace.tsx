import {
  Button,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxCustomButtonTriggerWrapper,
  DialogComboboxFooter,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  GavelIcon,
  getShadowScrollStyles,
  PlayIcon,
  PlusIcon,
  Popover,
  SparkleDoubleIcon,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { getTypeColor, getTypeDisplayName, getTypeIcon } from './scorerCardUtils';
import { useGetScheduledScorers } from './hooks/useGetScheduledScorers';
import { useExperimentIds } from '../../components/experiment-page/hooks/useExperimentIds';
import { useIntl } from 'react-intl';
import { LLM_TEMPLATE, LLMScorer, ScheduledScorer } from './types';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useEvaluateTraces } from './useEvaluateTraces';
import { useEvaluateTracesUsingKnownScorer } from './useEvaluateTracesUsingKnownScorer';
import ScorerModalRenderer from './ScorerModalRenderer';
import { SCORER_FORM_MODE, ScorerEvaluationScope } from './constants';
import { useTemplateOptions } from './llmScorerUtils';
import { isString } from 'lodash';
import { RunIcon } from '../../components/run-page/assets/RunIcon';

export const RunJudgeButtonForTrace = ({
  traceId,
  onScorerStarted,
  evaluateTraces,
  disabled,
}: {
  traceId: string;
  onScorerStarted?: (scorerName: string) => void;
  evaluateTraces: (scorer: LLMScorer, traceIds: string[]) => void;
  disabled?: boolean;
}) => {
  const [experimentId] = useExperimentIds();
  const { data } = useGetScheduledScorers(experimentId);
  const { templateOptions } = useTemplateOptions(ScorerEvaluationScope.TRACES);

  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [searchValue, setSearchValue] = useState<string>('');
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [isCreateScorerModalVisible, setIsCreateScorerModalVisible] = useState(false);

  const displayedScorers = useMemo(() => {
    return data?.scheduledScorers.filter((scorer) => scorer.name.toLowerCase().includes(searchValue.toLowerCase()));
  }, [data?.scheduledScorers, searchValue]);

  const displayedTemplates = useMemo(() => {
    return templateOptions?.filter((template) => template.label.toLowerCase().includes(searchValue.toLowerCase()));
  }, [templateOptions, searchValue]);

  const handleClick = (scorerOrTemplate: ScheduledScorer | LLM_TEMPLATE) => {
    if (isString(scorerOrTemplate)) {
      // evaluateTraces({ llmTemplate: scorer, instructions: '', name: '', type: 'llm' }, [traceId]);
      // evaluateTraces(scorerOrTemplate, [traceId]);
      // setDropdownOpen(false);
      return;
    }
    if ('instructions' in scorerOrTemplate) {
      evaluateTraces(scorerOrTemplate, [traceId]);
      setDropdownOpen(false);
      onScorerStarted?.(scorerOrTemplate.name);
    }
  };

  return (
    <>
      <DialogCombobox id="TODO" componentId="TODO" open={dropdownOpen} onOpenChange={setDropdownOpen}>
        <DialogComboboxCustomButtonTriggerWrapper asChild>
          <Button componentId="TODO" size="small" icon={<PlayIcon />} disabled={disabled}>
            Run judge
          </Button>
        </DialogComboboxCustomButtonTriggerWrapper>
        <DialogComboboxContent align="end" css={{ minWidth: 400 }}>
          <DialogComboboxOptionList
            css={{
              height: 240,
              display: 'flex',
              flexDirection: 'column',
              overflowY: 'auto',
              ...getShadowScrollStyles(theme, { orientation: 'vertical' }),
              borderTop: `1px solid ${theme.colors.border}`,
            }}
          >
            <DialogComboboxOptionListSearch controlledValue={searchValue} setControlledValue={setSearchValue}>
              {displayedScorers?.map((scorer) => (
                <ScorerOption scorer={scorer} key={scorer.name} onClick={handleClick} />
              ))}
              {displayedTemplates?.map((template) => (
                <TemplateOption
                  template={template}
                  key={template.value}
                  onClick={(template) => handleClick(template)}
                />
              ))}
            </DialogComboboxOptionListSearch>
          </DialogComboboxOptionList>

          <DialogComboboxFooter css={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              type="tertiary"
              componentId="TODO"
              icon={<PlusIcon />}
              css={{ flex: 1 }}
              onClick={() => setIsCreateScorerModalVisible(true)}
            >
              Create judge
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

const ScorerOption = ({ scorer, onClick }: { scorer: ScheduledScorer; onClick: (scorer: ScheduledScorer) => void }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  return (
    <DialogComboboxOptionListSelectItem
      value={scorer.name}
      icon={<GavelIcon />}
      css={{
        alignItems: 'center',
        height: theme.general.buttonHeight,
        '&>label': { flex: 1 },

        flexShrink: 0,
      }}
      key={scorer.name}
      hintColumn={
        <Tag
          componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scorercardrenderer_123"
          color={getTypeColor(scorer)}
          icon={getTypeIcon(scorer)}
        >
          {getTypeDisplayName(scorer, intl)}
        </Tag>
      }
      onChange={() => onClick(scorer)}
    >
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <Typography.Text css={{ flex: 1 }}>{scorer.name}</Typography.Text>
        <Typography.Hint>Custom judge</Typography.Hint>
      </div>
    </DialogComboboxOptionListSelectItem>
  );
};

const TemplateOption = ({
  template,
  onClick,
}: {
  template: {
    value: LLM_TEMPLATE;
    label: string;
    hint: string;
  };
  onClick: (template: LLM_TEMPLATE) => void;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  return (
    <DialogComboboxOptionListSelectItem
      value={template.value}
      icon={<SparkleDoubleIcon />}
      css={{
        alignItems: 'center',
        height: theme.general.buttonHeight,
        '&>label': { flex: 1 },

        flexShrink: 0,
      }}
      key={template.value}
      hintColumn={
        <Tag
          componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scorercardrenderer_123"
          color={'pink'}
          icon={<SparkleDoubleIcon />}
        >
          {intl.formatMessage({
            defaultMessage: 'LLM-as-a-judge',
            description: 'Label for LLM scorer type',
          })}
        </Tag>
      }
      onChange={() => onClick(template.value)}
    >
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <Typography.Text css={{ flex: 1 }}>{template.label}</Typography.Text>
        <Typography.Hint>Built-in judge</Typography.Hint>
      </div>
    </DialogComboboxOptionListSelectItem>
  );
};

export const useRunJudgeButtonForTrace = ({
  // traceId,
  experimentId,
  // onScorerFinished,
  // onScorerStarted,
}: {
  // traceId: string;
  experimentId?: string;
  // onScorerFinished: () => void;
  // onScorerStarted?: (scorerName: string) => void;
}) => {
  const finishedCallbackRef = useRef<(() => void) | undefined>(undefined);
  const [scorerInProgress, setScorerInProgress] = useState<string | undefined>(undefined);
  const { evaluateTraces, isLoading } = useEvaluateTracesUsingKnownScorer({
    experimentId,
    onScorerFinished: () => {
      finishedCallbackRef.current?.();
      setScorerInProgress(undefined);
    },
  });

  const renderRunJudgeButton = useCallback(
    ({
      traceId,
      onRunJudgeFinishedCallback,
      disabled,
    }: {
      traceId: string;
      onRunJudgeFinishedCallback: () => void;
      disabled?: boolean;
    }) => {
      finishedCallbackRef.current = onRunJudgeFinishedCallback;
      return (
        <RunJudgeButtonForTrace
          traceId={traceId}
          onScorerStarted={setScorerInProgress}
          evaluateTraces={evaluateTraces}
          disabled={disabled}
        />
      );
    },
    [evaluateTraces],
  );

  return useMemo(
    () => ({ renderRunJudgeButton, judgeExecutionState: { isLoading, scorerInProgress } }),
    [renderRunJudgeButton, isLoading, scorerInProgress],
  );
};
