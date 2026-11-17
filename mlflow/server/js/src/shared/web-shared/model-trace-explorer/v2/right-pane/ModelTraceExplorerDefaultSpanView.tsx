import { isNil } from 'lodash';
import { useMemo, useState } from 'react';

import { Button, ChevronDownIcon, DropdownMenu, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import {
  CodeSnippetRenderMode,
  type ModelTraceChatMessage,
  type ModelTraceExplorerRenderMode,
  type ModelTraceSpanNode,
  type SearchMatch,
} from '../ModelTrace.types';
import {
  createListFromObject,
  hoistLeadingSystemMessages,
  normalizeConversation,
  removeOverlappingInputMessages,
} from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import {
  areModelTraceComparisonValuesEqual,
  ModelTraceExplorerFieldRenderer,
} from '../field-renderers/ModelTraceExplorerFieldRenderer';
import { ModelTraceExplorerIssueEvidence } from '../../ModelTraceExplorerIssueBanner';
import { useModelTraceExplorerContext } from '../ModelTraceExplorerContext';
import { ModelTraceExplorerChatSections } from './ModelTraceExplorerChatSections';
import { ModelTraceExplorerChatTool } from './ModelTraceExplorerChatTool';
import { ModelTraceExplorerConversation } from './ModelTraceExplorerConversation';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { DecisionAnswers, DecisionInputs } from './DecisionView';
import { resolveDecision } from './decision-utils';
import { shouldEnableNewEvalRunsUi } from '../../../../../common/utils/FeatureUtils';

type ModelTraceExplorerSectionRenderMode = 'pretty' | Extract<ModelTraceExplorerRenderMode, 'json' | 'yaml'>;
type ModelTraceExplorerField = { key: string; value: string };

const CHAT_FIELD_KEYS: Record<'inputs' | 'outputs', Set<string>> = {
  inputs: new Set(['messages', 'input', 'inputs', 'prompt', 'prompts', 'tools']),
  outputs: new Set(['messages', 'response', 'output', 'outputs', 'choices', 'generations', 'llm_output']),
};

const getInitialSectionRenderMode = (
  defaultRenderMode: ModelTraceExplorerRenderMode,
): ModelTraceExplorerSectionRenderMode => {
  if (defaultRenderMode === 'json' || defaultRenderMode === 'yaml') {
    return defaultRenderMode;
  }
  return 'pretty';
};

const stringifySectionData = (data: unknown) => JSON.stringify(data ?? null, null, 2);

const getInputChatMessages = (
  activeSpan: ModelTraceSpanNode | undefined,
): { messages: ModelTraceChatMessage[]; hasTopLevelChatPayload: boolean } => {
  if (!activeSpan) {
    return { messages: [], hasTopLevelChatPayload: false };
  }

  let inputMessages = normalizeConversation(activeSpan.inputs, activeSpan.chatMessageFormat) ?? [];
  if (activeSpan.chatMessageFormat === 'pydantic_ai') {
    const outputMessages = normalizeConversation(activeSpan.outputs, activeSpan.chatMessageFormat) ?? [];
    inputMessages = hoistLeadingSystemMessages(inputMessages, outputMessages);
  }
  if (inputMessages.length > 0) {
    return { messages: inputMessages, hasTopLevelChatPayload: true };
  }

  return {
    messages: activeSpan.chatMessages?.filter((message) => message.role === 'user' || message.role === 'system') ?? [],
    hasTopLevelChatPayload: false,
  };
};

const getOutputChatMessages = (
  activeSpan: ModelTraceSpanNode | undefined,
  inputChatMessages: ModelTraceChatMessage[],
): { messages: ModelTraceChatMessage[]; hasTopLevelChatPayload: boolean } => {
  if (!activeSpan) {
    return { messages: [], hasTopLevelChatPayload: false };
  }

  const outputMessages = normalizeConversation(activeSpan.outputs, activeSpan.chatMessageFormat) ?? [];
  const outputOnlyMessages = removeOverlappingInputMessages(outputMessages, inputChatMessages);
  if (outputOnlyMessages.length > 0) {
    return { messages: outputOnlyMessages, hasTopLevelChatPayload: true };
  }

  if (inputChatMessages.length > 0 && typeof activeSpan.outputs === 'string' && activeSpan.outputs.length > 0) {
    return { messages: [{ role: 'assistant', content: activeSpan.outputs }], hasTopLevelChatPayload: true };
  }

  return {
    messages:
      activeSpan.chatMessages?.filter(
        (message) => message.role === 'assistant' || message.role === 'tool' || message.role === 'function',
      ) ?? [],
    hasTopLevelChatPayload: false,
  };
};

// In a comparison, chat messages that did not come from the payload itself are compared as one extra field.
const getEffectiveComparisonFields = (
  fields: ModelTraceExplorerField[],
  chatMessages: { messages: ModelTraceChatMessage[]; hasTopLevelChatPayload: boolean },
): ModelTraceExplorerField[] => {
  if (chatMessages.hasTopLevelChatPayload || chatMessages.messages.length === 0) {
    return fields;
  }

  return [
    ...fields.filter(({ key }) => key !== 'mlflow.chat.messages'),
    { key: 'mlflow.chat.messages', value: JSON.stringify(chatMessages.messages, null, 2) },
  ];
};

// The input and output fields a comparison shows for a span, one entry per rendered field.
export const getSpanComparisonFields = (
  span: ModelTraceSpanNode | undefined,
): { inputs: ModelTraceExplorerField[]; outputs: ModelTraceExplorerField[] } => {
  const inputChatMessages = getInputChatMessages(span);
  return {
    inputs: getEffectiveComparisonFields(createListFromObject(span?.inputs), inputChatMessages),
    outputs: getEffectiveComparisonFields(
      createListFromObject(span?.outputs),
      getOutputChatMessages(span, inputChatMessages.messages),
    ),
  };
};

export function ModelTraceExplorerDefaultSpanView({
  activeSpan,
  className,
  searchFilter,
  activeMatch,
  defaultRenderMode,
}: {
  activeSpan: ModelTraceSpanNode | undefined;
  className?: string;
  searchFilter: string;
  activeMatch: SearchMatch | null;
  defaultRenderMode: ModelTraceExplorerRenderMode;
}): React.ReactElement | null {
  const { theme } = useDesignSystemTheme();
  const { experimentId } = useModelTraceExplorerContext();
  const { selectedComparisonNode, selectedNodeComparisonRole: comparisonRole } = useModelTraceExplorerViewState();
  const [sectionRenderModes, setSectionRenderModes] = useState<
    Record<'inputs' | 'outputs', ModelTraceExplorerSectionRenderMode>
  >({
    inputs: getInitialSectionRenderMode(defaultRenderMode),
    outputs: getInitialSectionRenderMode(defaultRenderMode),
  });
  const [openSectionRenderModeDropdown, setOpenSectionRenderModeDropdown] = useState<'inputs' | 'outputs' | null>(null);
  const inputList = useMemo(() => createListFromObject(activeSpan?.inputs), [activeSpan]);
  const outputList = useMemo(() => createListFromObject(activeSpan?.outputs), [activeSpan]);
  const inputChatMessagesResult = useMemo(() => getInputChatMessages(activeSpan), [activeSpan]);
  const inputChatMessages = inputChatMessagesResult.messages;
  const inputHasTopLevelChatPayload = inputChatMessagesResult.hasTopLevelChatPayload;
  const outputChatMessagesResult = useMemo(
    () => getOutputChatMessages(activeSpan, inputChatMessages),
    [activeSpan, inputChatMessages],
  );
  const outputChatMessages = outputChatMessagesResult.messages;
  const outputHasTopLevelChatPayload = outputChatMessagesResult.hasTopLevelChatPayload;
  const decision = useMemo(() => resolveDecision(activeSpan), [activeSpan]);
  const comparisonInputList = useMemo(
    () => createListFromObject(selectedComparisonNode?.inputs),
    [selectedComparisonNode],
  );
  const comparisonOutputList = useMemo(
    () => createListFromObject(selectedComparisonNode?.outputs),
    [selectedComparisonNode],
  );
  const comparisonInputChat = useMemo(() => getInputChatMessages(selectedComparisonNode), [selectedComparisonNode]);
  const comparisonOutputChat = useMemo(
    () => getOutputChatMessages(selectedComparisonNode, comparisonInputChat.messages),
    [comparisonInputChat, selectedComparisonNode],
  );
  const effectiveInputList = useMemo(
    () => getEffectiveComparisonFields(inputList, inputChatMessagesResult),
    [inputChatMessagesResult, inputList],
  );
  const effectiveOutputList = useMemo(
    () => getEffectiveComparisonFields(outputList, outputChatMessagesResult),
    [outputChatMessagesResult, outputList],
  );
  const effectiveComparisonInputList = useMemo(
    () => getEffectiveComparisonFields(comparisonInputList, comparisonInputChat),
    [comparisonInputChat, comparisonInputList],
  );
  const effectiveComparisonOutputList = useMemo(
    () => getEffectiveComparisonFields(comparisonOutputList, comparisonOutputChat),
    [comparisonOutputChat, comparisonOutputList],
  );

  if (isNil(activeSpan)) {
    return null;
  }

  const containsInputs = comparisonRole
    ? effectiveInputList.length > 0 || effectiveComparisonInputList.length > 0
    : inputList.length > 0 || inputChatMessages.length > 0;
  const containsOutputs = comparisonRole
    ? effectiveOutputList.length > 0 || effectiveComparisonOutputList.length > 0
    : outputList.length > 0 || outputChatMessages.length > 0;
  const containsTools = (activeSpan.chatTools?.length ?? 0) > 0;

  const isActiveMatchSpan = !isNil(activeMatch) && activeMatch.span.key === activeSpan.key;

  const renderModeDropdown = (
    section: 'inputs' | 'outputs',
    renderMode: ModelTraceExplorerSectionRenderMode,
    setRenderMode: (mode: ModelTraceExplorerSectionRenderMode) => void,
  ) => (
    <DropdownMenu.Root
      open={openSectionRenderModeDropdown === section}
      onOpenChange={(open) => setOpenSectionRenderModeDropdown(open ? section : null)}
    >
      <DropdownMenu.Trigger asChild>
        <Button
          size="small"
          componentId="shared.model-trace-explorer.default-span-view.render-mode"
          type="tertiary"
          endIcon={<ChevronDownIcon />}
          css={{
            color: `${theme.colors.textSecondary} !important`,
            fontSize: theme.typography.fontSizeSm,
            paddingInline: theme.spacing.xs,
            '& > span, svg': {
              color: `${theme.colors.textSecondary} !important`,
            },
          }}
        >
          {renderMode === 'pretty' ? (
            <FormattedMessage
              defaultMessage="Pretty"
              description="Label for the pretty render mode in the model trace explorer inputs/outputs section"
            />
          ) : renderMode === 'json' ? (
            <FormattedMessage
              defaultMessage="JSON"
              description="Label for the JSON render mode in the model trace explorer inputs/outputs section"
            />
          ) : (
            <FormattedMessage
              defaultMessage="YAML"
              description="Label for the YAML render mode in the model trace explorer inputs/outputs section"
            />
          )}
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align="end">
        <DropdownMenu.RadioGroup
          componentId="shared.model-trace-explorer.default-span-view.render-mode-radio"
          value={renderMode}
          onValueChange={(value) => {
            if (value === 'pretty' || value === 'json' || value === 'yaml') {
              setRenderMode(value);
            }
          }}
        >
          <DropdownMenu.RadioItem value="pretty">
            <DropdownMenu.ItemIndicator />
            <FormattedMessage
              defaultMessage="Pretty"
              description="Label for the pretty render mode dropdown item in the model trace explorer inputs/outputs section"
            />
          </DropdownMenu.RadioItem>
          <DropdownMenu.RadioItem value="yaml">
            <DropdownMenu.ItemIndicator />
            <FormattedMessage
              defaultMessage="YAML"
              description="Label for the YAML render mode dropdown item in the model trace explorer inputs/outputs section"
            />
          </DropdownMenu.RadioItem>
          <DropdownMenu.RadioItem value="json">
            <DropdownMenu.ItemIndicator />
            <FormattedMessage
              defaultMessage="JSON"
              description="Label for the JSON render mode dropdown item in the model trace explorer inputs/outputs section"
            />
          </DropdownMenu.RadioItem>
        </DropdownMenu.RadioGroup>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );

  const renderPrettyFields = (
    section: 'inputs' | 'outputs',
    fields: typeof inputList,
    comparisonFields: typeof inputList = [],
  ) => {
    if (!comparisonRole && !searchFilter && decision) {
      if (section === 'outputs') {
        return <DecisionAnswers decision={decision} />;
      }
      return <DecisionInputs decision={decision} fields={fields} assessments={activeSpan.assessments} />;
    }

    // In a comparison, fields are the union of both sides, with changed fields first.
    const activeFields = new Map(fields.map(({ key, value }) => [key, value]));
    const counterpartFields = new Map(comparisonFields.map(({ key, value }) => [key, value]));
    const baselineFields = comparisonRole === 'candidate' ? comparisonFields : fields;
    const candidateFields = comparisonRole === 'candidate' ? fields : comparisonFields;
    const fieldKeys = [
      ...baselineFields.map(({ key }) => key),
      ...candidateFields.map(({ key }) => key).filter((key) => !baselineFields.some((field) => field.key === key)),
    ];
    const sortedFieldKeys = comparisonRole
      ? [...fieldKeys].sort(
          (firstKey, secondKey) =>
            Number(areModelTraceComparisonValuesEqual(activeFields.get(firstKey), counterpartFields.get(firstKey))) -
            Number(areModelTraceComparisonValuesEqual(activeFields.get(secondKey), counterpartFields.get(secondKey))),
        )
      : fieldKeys;
    const displayedFields = comparisonRole
      ? sortedFieldKeys.map((key) => ({ key, value: activeFields.get(key) }))
      : fields;
    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {displayedFields.map(({ key, value }, index) => (
          <ModelTraceExplorerFieldRenderer
            key={key || index}
            title={key}
            data={value}
            comparisonData={counterpartFields.get(key)}
            comparisonRole={comparisonRole}
            comparisonEnabled={Boolean(comparisonRole)}
            renderMode="default"
            assessments={activeSpan?.assessments}
            searchFilter={searchFilter}
            activeMatch={activeMatch}
            containsActiveMatch={isActiveMatchSpan && activeMatch?.section === section && activeMatch.key === key}
          />
        ))}
      </div>
    );
  };

  // PydanticAI agent spans take `message_history` and `user_prompt` arguments and return the
  // run's new messages under `_new_messages_serialized`. Each is already part of the chat
  // transcript when that transcript was parsed from the same section, so hide the raw field
  // rather than showing the same content twice.
  const isRenderedPydanticAIField = (section: 'inputs' | 'outputs', key: string) => {
    if (activeSpan.chatMessageFormat !== 'pydantic_ai') {
      return false;
    }
    if (section === 'inputs' && key === 'message_history') {
      return inputHasTopLevelChatPayload;
    }
    if (section === 'inputs' && key === 'user_prompt') {
      // Compare the normalized form so multimodal prompts (text plus image parts) also match.
      const promptMessage = normalizeConversation({ user_prompt: activeSpan.inputs?.user_prompt }, 'pydantic_ai')?.[0];
      return (
        !isNil(promptMessage) &&
        inputChatMessages.some((message) => JSON.stringify(message) === JSON.stringify(promptMessage))
      );
    }
    if (section === 'outputs' && key === '_new_messages_serialized') {
      return outputHasTopLevelChatPayload;
    }
    return false;
  };

  const filterNonChatFields = (
    section: 'inputs' | 'outputs',
    fields: typeof inputList,
    skipAnonymousTopLevelField = false,
  ) =>
    fields.filter(
      ({ key }) =>
        !(skipAnonymousTopLevelField && key === '') &&
        !CHAT_FIELD_KEYS[section].has(key.toLowerCase()) &&
        !(!searchFilter && isRenderedPydanticAIField(section, key)),
    );

  const renderNonChatFields = (
    section: 'inputs' | 'outputs',
    fields: typeof inputList,
    skipAnonymousTopLevelField = false,
  ) => {
    const nonChatFields = filterNonChatFields(section, fields, skipAnonymousTopLevelField);
    return nonChatFields.length > 0 ? renderPrettyFields(section, nonChatFields) : null;
  };

  const renderSectionPayload = (section: 'inputs' | 'outputs', data: unknown) => {
    if (comparisonRole) {
      return renderPrettyFields(
        section,
        section === 'inputs' ? effectiveInputList : effectiveOutputList,
        section === 'inputs' ? effectiveComparisonInputList : effectiveComparisonOutputList,
      );
    }
    if (sectionRenderModes[section] === 'pretty') {
      if (decision && !searchFilter) {
        return renderPrettyFields(section, section === 'inputs' ? inputList : outputList);
      }

      if (section === 'inputs' && inputChatMessages.length > 0) {
        return (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <ModelTraceExplorerChatSections messages={inputChatMessages} />
            {renderNonChatFields(section, inputList, inputHasTopLevelChatPayload)}
          </div>
        );
      }

      if (section === 'outputs' && outputChatMessages.length > 0) {
        return (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <ModelTraceExplorerConversation messages={outputChatMessages} />
            {renderNonChatFields(section, outputList, outputHasTopLevelChatPayload)}
          </div>
        );
      }

      return renderPrettyFields(section, section === 'inputs' ? inputList : outputList);
    }

    const stringifiedData = stringifySectionData(data);
    return (
      <ModelTraceExplorerCodeSnippet
        title=""
        data={stringifiedData}
        initialRenderMode={
          sectionRenderModes[section] === 'yaml' ? CodeSnippetRenderMode.YAML : CodeSnippetRenderMode.JSON
        }
        searchFilter={searchFilter}
        activeMatch={activeMatch}
        containsActiveMatch={isActiveMatchSpan && activeMatch?.section === section}
        hideRenderModeDropdown
      />
    );
  };

  const renderSectionTitle = (section: 'inputs' | 'outputs', title: React.ReactNode) => (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        gap: theme.spacing.sm,
        width: '100%',
      }}
    >
      {title}
      {!comparisonRole &&
        renderModeDropdown(section, sectionRenderModes[section], (mode) =>
          setSectionRenderModes((current) => ({ ...current, [section]: mode })),
        )}
    </div>
  );

  return (
    <div
      css={{
        paddingLeft: theme.spacing.md + theme.spacing.xs,
        paddingRight: theme.spacing.md + theme.spacing.xs,
        paddingTop: theme.spacing.sm,
        ...(shouldEnableNewEvalRunsUi() ? { paddingBottom: theme.spacing.xl } : {}),
      }}
      data-testid="model-trace-explorer-default-span-view"
    >
      {activeSpan && <ModelTraceExplorerIssueEvidence activeSpan={activeSpan} experimentId={experimentId} />}
      {containsTools && (
        <ModelTraceExplorerCollapsibleSection
          withBorder
          initialExpanded={false}
          css={{
            borderBottom: `1px solid ${theme.colors.borderDecorative}`,
            marginBottom: theme.spacing.sm,
            paddingBottom: theme.spacing.sm,
          }}
          headerPadding={`${theme.spacing.xs}px 0`}
          contentPadding={`${theme.spacing.xs}px 0`}
          sectionKey="tools"
          title={
            <span css={{ display: 'flex', alignItems: 'center', width: '100%' }}>
              <FormattedMessage defaultMessage="Tools" description="Title of the available chat tools section" />
              <Typography.Text color="secondary" css={{ marginLeft: theme.spacing.xs }}>
                <FormattedMessage
                  defaultMessage="({count})"
                  description="Count shown beside the available chat tools section title"
                  values={{ count: activeSpan.chatTools?.length ?? 0 }}
                />
              </Typography.Text>
              <Typography.Text data-tools-expand-hint color="secondary" size="sm" css={{ marginLeft: 'auto' }}>
                <FormattedMessage
                  defaultMessage="Click to Expand"
                  description="Hint next to the tools section chevron that the section can be expanded"
                />
              </Typography.Text>
            </span>
          }
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {activeSpan.chatTools?.map((tool) => (
              <ModelTraceExplorerChatTool key={tool.function.name} tool={tool} />
            ))}
          </div>
        </ModelTraceExplorerCollapsibleSection>
      )}
      {containsInputs && (
        <ModelTraceExplorerCollapsibleSection
          // A comparison remounts the section per span so a collapsed section opens again on the next span.
          key={comparisonRole ? `${activeSpan.key}-input` : undefined}
          withBorder
          css={{ marginBottom: theme.spacing.sm }}
          headerPadding={`${theme.spacing.xs}px 0`}
          contentPadding={`${theme.spacing.xs}px 0`}
          sectionKey="input"
          title={renderSectionTitle(
            'inputs',
            <>
              <FormattedMessage
                defaultMessage="Inputs"
                description="Model trace explorer > selected span > inputs header"
              />
            </>,
          )}
        >
          {renderSectionPayload('inputs', activeSpan.inputs)}
        </ModelTraceExplorerCollapsibleSection>
      )}
      {containsOutputs && (
        <ModelTraceExplorerCollapsibleSection
          key={comparisonRole ? `${activeSpan.key}-output` : undefined}
          withBorder
          css={{ borderTop: `1px solid ${theme.colors.borderDecorative}`, paddingTop: theme.spacing.sm }}
          headerPadding={`${theme.spacing.xs}px 0`}
          contentPadding={`${theme.spacing.xs}px 0`}
          sectionKey="output"
          title={renderSectionTitle(
            'outputs',
            <>
              <FormattedMessage
                defaultMessage="Outputs"
                description="Model trace explorer > selected span > outputs header"
              />
            </>,
          )}
        >
          {renderSectionPayload('outputs', activeSpan.outputs)}
        </ModelTraceExplorerCollapsibleSection>
      )}
    </div>
  );
}
