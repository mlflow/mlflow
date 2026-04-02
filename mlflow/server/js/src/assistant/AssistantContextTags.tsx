/**
 * Context tags component for the Assistant chat panel.
 * Displays explicitly selected context (traces, runs, sessions, datasets, prompts, models) as compact tags.
 * Implicit context like experimentId and currentPage is passed to the assistant but not displayed.
 */

import type { TagColors } from '@databricks/design-system';
import {
  DatabaseIcon,
  ForkHorizontalIcon,
  ModelsIcon,
  PlayIcon,
  SparkleDoubleIcon,
  SpeechBubbleIcon,
  Tag,
  TextBoxIcon,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';

import { useAssistantPageContext } from './AssistantPageContext';

const MAX_VISIBLE_ITEMS = 3;

function truncateId(id: string, maxLen = 8): string {
  return id.length > maxLen ? `${id.slice(0, maxLen)}...` : id;
}

interface ContextTagGroupProps {
  ids: string[];
  color: TagColors;
  label: string;
  componentId: string;
  Icon: typeof ForkHorizontalIcon;
}

function ContextTagGroup({ ids, color, label, componentId, Icon }: ContextTagGroupProps): React.ReactElement | null {
  const { theme } = useDesignSystemTheme();

  if (ids.length === 0) return null;

  const visibleIds = ids.slice(0, MAX_VISIBLE_ITEMS);
  const hiddenCount = ids.length - MAX_VISIBLE_ITEMS;

  return (
    <>
      {visibleIds.map((id) => (
        <Tooltip key={id} componentId={`${componentId}.tooltip`} content={`${label}: ${id}`}>
          <Tag componentId={componentId} color={color}>
            <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <Icon css={{ fontSize: 12 }} />
              <span>{truncateId(id)}</span>
            </span>
          </Tag>
        </Tooltip>
      ))}
      {hiddenCount > 0 && (
        <Tooltip componentId={`${componentId}.more.tooltip`} content={`${hiddenCount} more ${label.toLowerCase()}s`}>
          <Tag componentId={`${componentId}.more`} color={color}>
            +{hiddenCount}
          </Tag>
        </Tooltip>
      )}
    </>
  );
}

/**
 * Context tags showing current page context.
 */
export function AssistantContextTags(): React.ReactElement | null {
  const { theme } = useDesignSystemTheme();
  const context = useAssistantPageContext();

  // Traces
  const traceId = context['traceId'] as string | undefined;
  const selectedTraceIds = context['selectedTraceIds'] as string[] | undefined;
  const traceIds = [...new Set([traceId, ...(selectedTraceIds ?? [])].filter(Boolean))] as string[];

  // Runs
  const runId = context['runId'] as string | undefined;
  const selectedRunIds = context['selectedRunIds'] as string[] | undefined;
  const runIds = [...new Set([runId, ...(selectedRunIds ?? [])].filter(Boolean))] as string[];

  // Sessions
  const sessionId = context['sessionId'] as string | undefined;
  const selectedSessionIds = context['selectedSessionIds'] as string[] | undefined;
  const sessionIds = [...new Set([sessionId, ...(selectedSessionIds ?? [])].filter(Boolean))] as string[];

  // Datasets
  const selectedDatasetId = context['selectedDatasetId'] as string | undefined;
  const datasetIds = selectedDatasetId ? [selectedDatasetId] : [];

  // Prompts - combine name and version info
  const promptName = context['promptName'] as string | undefined;
  const promptVersion = context['promptVersion'] as string | undefined;
  const comparedPromptVersion = context['comparedPromptVersion'] as string | undefined;
  const promptIds = promptName
    ? [promptVersion ? `${promptName}@${promptVersion}` : promptName].concat(
        comparedPromptVersion ? [`${promptName}@${comparedPromptVersion}`] : [],
      )
    : [];

  // Models - combine name and versions
  const modelName = context['modelName'] as string | undefined;
  const modelVersion = context['modelVersion'] as string | undefined;
  const selectedModelVersions = context['selectedModelVersions'] as string[] | undefined;
  const modelIds = modelName
    ? [
        // If no version info, just show model name; otherwise show name@version
        ...(modelVersion ? [`${modelName}@${modelVersion}`] : selectedModelVersions?.length ? [] : [modelName]),
        ...(selectedModelVersions?.map((v) => `${modelName}@v${v}`) ?? []),
      ].filter((v, i, arr) => arr.indexOf(v) === i) // Remove duplicates
    : [];

  // Scorers/Judges
  const selectedScorerName = context['selectedScorerName'] as string | undefined;
  const scorerIds = selectedScorerName ? [selectedScorerName] : [];

  const hasContext =
    traceIds.length > 0 ||
    runIds.length > 0 ||
    sessionIds.length > 0 ||
    datasetIds.length > 0 ||
    promptIds.length > 0 ||
    modelIds.length > 0 ||
    scorerIds.length > 0;

  if (!hasContext) return null;

  return (
    <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs, paddingTop: theme.spacing.sm }}>
      <ContextTagGroup
        ids={traceIds}
        color="indigo"
        label="Trace"
        componentId="mlflow.assistant.chat_panel.context.trace"
        Icon={ForkHorizontalIcon}
      />
      <ContextTagGroup
        ids={runIds}
        color="turquoise"
        label="Run"
        componentId="mlflow.assistant.chat_panel.context.run"
        Icon={PlayIcon}
      />
      <ContextTagGroup
        ids={sessionIds}
        color="purple"
        label="Session"
        componentId="mlflow.assistant.chat_panel.context.session"
        Icon={SpeechBubbleIcon}
      />
      <ContextTagGroup
        ids={datasetIds}
        color="brown"
        label="Dataset"
        componentId="mlflow.assistant.chat_panel.context.dataset"
        Icon={DatabaseIcon}
      />
      <ContextTagGroup
        ids={promptIds}
        color="teal"
        label="Prompt"
        componentId="mlflow.assistant.chat_panel.context.prompt"
        Icon={TextBoxIcon}
      />
      <ContextTagGroup
        ids={modelIds}
        color="lemon"
        label="Model"
        componentId="mlflow.assistant.chat_panel.context.model"
        Icon={ModelsIcon}
      />
      <ContextTagGroup
        ids={scorerIds}
        color="pink"
        label="Judge"
        componentId="mlflow.assistant.chat_panel.context.scorer"
        Icon={SparkleDoubleIcon}
      />
    </div>
  );
}
