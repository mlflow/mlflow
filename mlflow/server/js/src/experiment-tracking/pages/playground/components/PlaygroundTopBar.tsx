import { useDesignSystemTheme } from '@databricks/design-system';
import { EndpointSelector } from '../../../components/EndpointSelector';
import type { ChatMessage, PlaygroundParams, ResponseFormatType, ToolChoice } from '../types';
import { ParametersButton } from './ParametersButton';
import { RegistryButton } from './RegistryButton';
import { SaveToRegistryButton } from './SaveToRegistryButton';
import { VariablesButton } from './VariablesButton';

interface Props {
  endpointName: string;
  onEndpointSelect: (name: string) => void;
  params: PlaygroundParams;
  onParamsChange: (next: PlaygroundParams) => void;
  toolsText: string;
  onToolsChange: (next: string) => void;
  toolsError?: string | null;
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
}

export const PlaygroundTopBar = ({
  endpointName,
  onEndpointSelect,
  params,
  onParamsChange,
  toolsText,
  onToolsChange,
  toolsError,
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
}: Props) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
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
        showCreateButton={false}
        triggerMaxWidth={400}
      />
      <ParametersButton
        value={params}
        onChange={onParamsChange}
        toolsText={toolsText}
        onToolsChange={onToolsChange}
        toolsError={toolsError}
        toolChoice={toolChoice}
        onToolChoiceChange={onToolChoiceChange}
        responseFormatType={responseFormatType}
        onResponseFormatTypeChange={onResponseFormatTypeChange}
        responseFormatSchemaText={responseFormatSchemaText}
        onResponseFormatSchemaChange={onResponseFormatSchemaChange}
        responseFormatSchemaError={responseFormatSchemaError}
      />
      <VariablesButton messages={messages} value={variables} onChange={onVariablesChange} />
      <RegistryButton onOpen={onOpenRegistry} />
      <SaveToRegistryButton onOpen={onOpenSave} disabled={saveDisabled} />
    </div>
  );
};
