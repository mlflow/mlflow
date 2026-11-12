import { useDesignSystemTheme } from '@databricks/design-system';
import { useLayoutEffect, useRef, useState } from 'react';
import { EndpointSelector } from '../../../components/EndpointSelector';
import type { ChatMessage, PlaygroundParams, PlaygroundTool, ResponseFormatType, ToolChoice } from '../types';
import { AddToDatasetButton } from './AddToDatasetButton';
import { ParametersButton } from './ParametersButton';
import { RegistryButton } from './RegistryButton';
import { SaveToRegistryButton } from './SaveToRegistryButton';
import { VariablesButton } from './VariablesButton';

// At or below this bar width the prompt actions (Load, Save, Add to datasets) show only
// their icons, so the bar stays on one row. It sits just above the ~1010px the bar needs
// with full labels and the widest (400px) endpoint selector.
const COMPACT_ACTIONS_MAX_BAR_WIDTH = 1040;

interface Props {
  endpointName: string;
  onEndpointSelect: (name: string) => void;
  params: PlaygroundParams;
  onParamsChange: (next: PlaygroundParams) => void;
  tools: PlaygroundTool[];
  onAddTool: () => void;
  onRemoveTool: (id: string) => void;
  onUpdateTool: (id: string, patch: Partial<PlaygroundTool>) => void;
  toolChoice: ToolChoice;
  onToolChoiceChange: (next: ToolChoice) => void;
  responseFormatType: ResponseFormatType;
  onResponseFormatTypeChange: (next: ResponseFormatType) => void;
  responseFormatSchemaText: string;
  onResponseFormatSchemaChange: (next: string) => void;
  responseFormatSchemaError?: string | null;
  messages: ChatMessage[];
  variables: Record<string, string>;
  onVariablesChange: (next: Record<string, string>) => void;
  onOpenRegistry: () => void;
  onOpenSave: () => void;
  saveDisabled?: boolean;
  onOpenAddToDataset: () => void;
  addToDatasetDisabled?: boolean;
}

export const PlaygroundTopBar = ({
  endpointName,
  onEndpointSelect,
  params,
  onParamsChange,
  tools,
  onAddTool,
  onRemoveTool,
  onUpdateTool,
  toolChoice,
  onToolChoiceChange,
  responseFormatType,
  onResponseFormatTypeChange,
  responseFormatSchemaText,
  onResponseFormatSchemaChange,
  responseFormatSchemaError,
  messages,
  variables,
  onVariablesChange,
  onOpenRegistry,
  onOpenSave,
  saveDisabled,
  onOpenAddToDataset,
  addToDatasetDisabled,
}: Props) => {
  const { theme } = useDesignSystemTheme();

  // Measured on the bar, not the viewport, because the assistant panel narrows the bar
  // without the window changing. A layout effect, so a narrow bar never paints wrapped.
  const barRef = useRef<HTMLDivElement>(null);
  const [barWidth, setBarWidth] = useState<number | null>(null);
  useLayoutEffect(() => {
    const bar = barRef.current;
    if (!bar) return;
    const measure = () => {
      if (bar.clientWidth) setBarWidth(bar.clientWidth);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(bar);
    return () => observer.disconnect();
  }, []);
  const compactActions = barWidth !== null && barWidth <= COMPACT_ACTIONS_MAX_BAR_WIDTH;

  return (
    <div
      ref={barRef}
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        flexWrap: 'wrap',
      }}
    >
      <EndpointSelector
        componentIdPrefix="mlflow.playground.endpoint-selector"
        currentEndpointName={endpointName}
        onEndpointSelect={onEndpointSelect}
        showCreateButton
        triggerMaxWidth={400}
      />
      <ParametersButton
        value={params}
        onChange={onParamsChange}
        tools={tools}
        onAddTool={onAddTool}
        onRemoveTool={onRemoveTool}
        onUpdateTool={onUpdateTool}
        toolChoice={toolChoice}
        onToolChoiceChange={onToolChoiceChange}
        responseFormatType={responseFormatType}
        onResponseFormatTypeChange={onResponseFormatTypeChange}
        responseFormatSchemaText={responseFormatSchemaText}
        onResponseFormatSchemaChange={onResponseFormatSchemaChange}
        responseFormatSchemaError={responseFormatSchemaError}
      />
      <VariablesButton messages={messages} value={variables} onChange={onVariablesChange} />
      <RegistryButton onOpen={onOpenRegistry} compact={compactActions} />
      <SaveToRegistryButton onOpen={onOpenSave} disabled={saveDisabled} compact={compactActions} />
      <AddToDatasetButton onOpen={onOpenAddToDataset} disabled={addToDatasetDisabled} compact={compactActions} />
    </div>
  );
};
