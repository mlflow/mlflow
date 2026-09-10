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
import { createListFromObject, normalizeConversation } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';
import { ModelTraceExplorerChatSections } from './ModelTraceExplorerChatSections';
import { ModelTraceExplorerChatTool } from './ModelTraceExplorerChatTool';
import { ModelTraceExplorerConversation } from './ModelTraceExplorerConversation';

type ModelTraceExplorerSectionRenderMode = 'pretty' | Extract<ModelTraceExplorerRenderMode, 'json' | 'yaml'>;

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

const areChatMessagesEquivalent = (first: ModelTraceChatMessage, second: ModelTraceChatMessage): boolean => {
  return JSON.stringify(first) === JSON.stringify(second);
};

const removeInputMessagesPrefix = (
  outputMessages: ModelTraceChatMessage[],
  inputMessages: ModelTraceChatMessage[],
): ModelTraceChatMessage[] => {
  if (
    outputMessages.length >= inputMessages.length &&
    inputMessages.every((message, index) => areChatMessagesEquivalent(message, outputMessages[index]))
  ) {
    return outputMessages.slice(inputMessages.length);
  }

  return outputMessages;
};

const getInputChatMessages = (activeSpan: ModelTraceSpanNode | undefined): ModelTraceChatMessage[] => {
  if (!activeSpan) {
    return [];
  }

  const inputMessages = normalizeConversation(activeSpan.inputs, activeSpan.chatMessageFormat) ?? [];
  if (inputMessages.length > 0) {
    return inputMessages;
  }

  return activeSpan.chatMessages?.filter((message) => message.role === 'user' || message.role === 'system') ?? [];
};

const getOutputChatMessages = (
  activeSpan: ModelTraceSpanNode | undefined,
  inputChatMessages: ModelTraceChatMessage[],
): ModelTraceChatMessage[] => {
  if (!activeSpan) {
    return [];
  }

  const outputMessages = normalizeConversation(activeSpan.outputs, activeSpan.chatMessageFormat) ?? [];
  const outputOnlyMessages = removeInputMessagesPrefix(outputMessages, inputChatMessages);
  if (outputOnlyMessages.length > 0) {
    return outputOnlyMessages;
  }

  if (inputChatMessages.length > 0 && typeof activeSpan.outputs === 'string' && activeSpan.outputs.length > 0) {
    return [{ role: 'assistant', content: activeSpan.outputs }];
  }

  return (
    activeSpan.chatMessages?.filter(
      (message) => message.role === 'assistant' || message.role === 'tool' || message.role === 'function',
    ) ?? []
  );
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
  const [sectionRenderModes, setSectionRenderModes] = useState<
    Record<'inputs' | 'outputs', ModelTraceExplorerSectionRenderMode>
  >({
    inputs: getInitialSectionRenderMode(defaultRenderMode),
    outputs: getInitialSectionRenderMode(defaultRenderMode),
  });
  const [openSectionRenderModeDropdown, setOpenSectionRenderModeDropdown] = useState<'inputs' | 'outputs' | null>(null);
  const inputList = useMemo(() => createListFromObject(activeSpan?.inputs), [activeSpan]);
  const outputList = useMemo(() => createListFromObject(activeSpan?.outputs), [activeSpan]);
  const inputChatMessages = useMemo(() => getInputChatMessages(activeSpan), [activeSpan]);
  const outputChatMessages = useMemo(
    () => getOutputChatMessages(activeSpan, inputChatMessages),
    [activeSpan, inputChatMessages],
  );

  if (isNil(activeSpan)) {
    return null;
  }

  const containsInputs = inputList.length > 0 || inputChatMessages.length > 0;
  const containsOutputs = outputList.length > 0 || outputChatMessages.length > 0;
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

  const renderPrettyFields = (section: 'inputs' | 'outputs', fields: typeof inputList) => (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {fields.map(({ key, value }, index) => (
        <ModelTraceExplorerFieldRenderer
          key={key || index}
          title={key}
          data={value}
          renderMode="default"
          assessments={activeSpan?.assessments}
          searchFilter={searchFilter}
          activeMatch={activeMatch}
          containsActiveMatch={isActiveMatchSpan && activeMatch?.section === section && activeMatch.key === key}
        />
      ))}
    </div>
  );

  const renderNonChatFields = (section: 'inputs' | 'outputs', fields: typeof inputList) => {
    const nonChatFields = fields.filter(({ key }) => !CHAT_FIELD_KEYS[section].has(key.toLowerCase()));
    return nonChatFields.length > 0 ? renderPrettyFields(section, nonChatFields) : null;
  };

  const renderSectionPayload = (section: 'inputs' | 'outputs', data: unknown) => {
    if (sectionRenderModes[section] === 'pretty') {
      if (isActiveMatchSpan && activeMatch.section === section) {
        return renderPrettyFields(section, section === 'inputs' ? inputList : outputList);
      }

      if (section === 'inputs' && inputChatMessages.length > 0) {
        return (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <ModelTraceExplorerChatSections messages={inputChatMessages} />
            {renderNonChatFields(section, inputList)}
          </div>
        );
      }

      if (section === 'outputs' && outputChatMessages.length > 0) {
        return (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <ModelTraceExplorerConversation messages={outputChatMessages} />
            {renderNonChatFields(section, outputList)}
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
      {renderModeDropdown(section, sectionRenderModes[section], (mode) =>
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
      }}
      data-testid="model-trace-explorer-default-span-view"
    >
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
