import { useCallback, useMemo, useRef, useState } from 'react';
import { LLMScorer } from '../types';
import { useGetScheduledScorers } from './useGetScheduledScorers';
import { useExperimentIds } from '../../../components/experiment-page/hooks/useExperimentIds';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  Alert,
  Button,
  getShadowScrollStyles,
  Input,
  Modal,
  PlusIcon,
  Radio,
  SearchIcon,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { PillControl } from '@databricks/design-system/development';
import ScorerModalRenderer from '../ScorerModalRenderer';
import { SCORER_FORM_MODE } from '../constants';
import { useRunSerializedScorer } from './useRunSerializedScorer';
import { ModelTraceExplorerRunJudgeConfig } from '@databricks/web-shared/model-trace-explorer';
import { ScorerFinishedEvent } from '../useEvaluateTracesAsync';

interface UseRunScorerInTracesViewConfigurationReturnType extends ModelTraceExplorerRunJudgeConfig {
  RunJudgeModalElement: React.ReactNode;
}

export const useRunScorerInTracesViewConfiguration = (): UseRunScorerInTracesViewConfigurationReturnType => {
  const [experimentId] = useExperimentIds();

  const scorerFinishSubscribers = useRef<((event: ScorerFinishedEvent) => void)[]>([]);

  const onScorerFinished = useCallback((event: ScorerFinishedEvent) => {
    scorerFinishSubscribers.current.forEach((callback) => callback(event));
  }, []);

  const subscribeToScorerFinished = useCallback((callback: (event: ScorerFinishedEvent) => void) => {
    scorerFinishSubscribers.current.push(callback);
    return () => {
      scorerFinishSubscribers.current = scorerFinishSubscribers.current.filter(
        (currentCallback) => currentCallback !== callback,
      );
    };
  }, []);

  const { evaluateTraces, allEvaluations } = useRunSerializedScorer({ experimentId, onScorerFinished });

  const renderRunJudgeModal = useCallback<NonNullable<ModelTraceExplorerRunJudgeConfig['renderRunJudgeModal']>>(
    ({ traceId, onClose, visible }) => {
      return (
        <RunJudgeModalImpl visible={visible} traceId={traceId} evaluateTraces={evaluateTraces} onClose={onClose} />
      );
    },
    [evaluateTraces],
  );

  return {
    renderRunJudgeModal,
    evaluations: allEvaluations,
    subscribeToScorerFinished,
  } as UseRunScorerInTracesViewConfigurationReturnType;
};

/**
 * Dropdown for selecting a judge to run against a trace.
 */
const RunJudgeModalImpl = ({
  traceId,
  onScorerStarted,
  evaluateTraces,
  visible,
  onClose,
}: {
  traceId: string;
  onScorerStarted?: (scorerName: string) => void;
  evaluateTraces: (scorer: LLMScorer, traceIds: string[]) => void;
  visible: boolean;
  onClose: () => void;
}) => {
  const [experimentId] = useExperimentIds();
  const { data, isLoading: loadingScorers } = useGetScheduledScorers(experimentId, { enabled: visible });
  const intl = useIntl();

  const { theme } = useDesignSystemTheme();
  const [searchValue, setSearchValue] = useState<string>('');
  const [isCreateScorerModalVisible, setIsCreateScorerModalVisible] = useState(false);

  const displayedLLMScorers = useMemo(() => {
    return data?.scheduledScorers.filter(
      (scorer) => scorer.type === 'llm' && scorer.name.toLowerCase().includes(searchValue.toLowerCase()),
    ) as LLMScorer[];
  }, [data?.scheduledScorers, searchValue]);

  const [error, setError] = useState<Error | undefined>(undefined);
  const [selectedJudge, setSelectedJudge] = useState<LLMScorer | undefined>(undefined);

  const handleModalConfirm = async () => {
    if (!selectedJudge) {
      return;
    }
    setError(undefined);
    try {
      evaluateTraces(selectedJudge, [traceId]);
      onClose();
      onScorerStarted?.(selectedJudge.name);
    } catch (error) {
      setError(error as Error);
    }
  };
  if (!visible) {
    return null;
  }
  return (
    <>
      <Modal
        componentId="mlflow.experiment-scorers.traces-view-judge-select-modal"
        visible
        onCancel={onClose}
        title={<FormattedMessage defaultMessage="Run judge on trace" description="Title for run judge modal" />}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Button text for canceling a judge run',
        })}
        okText={intl.formatMessage({
          defaultMessage: 'Run judge',
          description: 'Button text for running a judge',
        })}
        okButtonProps={{ disabled: !selectedJudge }}
        onOk={handleModalConfirm}
      >
        <div css={{ display: 'flex', gap: theme.spacing.sm, marginBottom: theme.spacing.sm }}>
          <Input
            componentId="mlflow.experiment-scorers.traces-view-judge-search"
            prefix={<SearchIcon />}
            value={searchValue}
            onChange={(e) => setSearchValue(e.target.value)}
            placeholder={intl.formatMessage({
              defaultMessage: 'Search judges',
              description: 'Placeholder for scorer search input',
            })}
          />
          <Button
            componentId="mlflow.experiment-scorers.traces-view-create-judge"
            icon={<PlusIcon />}
            onClick={() => setIsCreateScorerModalVisible(true)}
          >
            <FormattedMessage defaultMessage="Create judge" description="Button to create a new judge" />
          </Button>
        </div>
        <div css={{ marginBottom: theme.spacing.md }}>
          {error && (
            <Alert
              message={error.message}
              type="error"
              componentId="mlflow.experiment-scorers.traces-view-judge-error"
              css={{ marginBottom: theme.spacing.sm }}
              closable={false}
            />
          )}
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
          {loadingScorers && <TableSkeleton lines={3} />}
          {displayedLLMScorers?.map((scorer) => (
            <ScorerOption
              scorer={scorer}
              key={scorer.name}
              onClick={() => setSelectedJudge(scorer)}
              selected={selectedJudge?.name === scorer.name}
            />
          ))}
        </div>
      </Modal>
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

const ScorerOption = ({
  scorer,
  onClick,
  selected,
}: {
  scorer: LLMScorer;
  onClick: (scorer: LLMScorer) => void;
  selected: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      role="radio"
      aria-checked={selected}
      css={{ cursor: 'pointer', height: 48, flexShrink: 0 }}
      onClick={() => onClick(scorer)}
    >
      <Radio componentId={`mlflow.experiment-scorers.traces-view-judge-${scorer.name}`} checked={selected}>
        <div css={{ display: 'flex', flexDirection: 'column', marginLeft: theme.spacing.xs }}>
          <Typography.Text css={{ flex: 1 }}>{scorer.name}</Typography.Text>
          <Typography.Hint>
            <FormattedMessage defaultMessage="Custom judge" description="Label indicating a custom judge scorer" />
          </Typography.Hint>
        </div>
      </Radio>
    </div>
  );
};
