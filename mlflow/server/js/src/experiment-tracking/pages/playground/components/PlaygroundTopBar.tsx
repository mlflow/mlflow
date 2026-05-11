import { useDesignSystemTheme } from '@databricks/design-system';
import { EndpointSelector } from '../../../components/EndpointSelector';
import type { ChatMessage, PlaygroundParams } from '../types';
import { ParametersButton } from './ParametersButton';
import { RegistryButton } from './RegistryButton';
import { VariablesButton } from './VariablesButton';

interface Props {
  endpointName: string;
  onEndpointSelect: (name: string) => void;
  params: PlaygroundParams;
  onParamsChange: (next: PlaygroundParams) => void;
  messages: ChatMessage[];
  variables: Record<string, string>;
  onVariablesChange: (next: Record<string, string>) => void;
  onOpenRegistry: () => void;
}

export const PlaygroundTopBar = ({
  endpointName,
  onEndpointSelect,
  params,
  onParamsChange,
  messages,
  variables,
  onVariablesChange,
  onOpenRegistry,
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
      <div css={{ flex: '0 0 auto', maxWidth: 320 }}>
        <EndpointSelector
          componentIdPrefix="mlflow.playground.endpoint-selector"
          currentEndpointName={endpointName}
          onEndpointSelect={onEndpointSelect}
          showCreateButton={false}
        />
      </div>
      <ParametersButton value={params} onChange={onParamsChange} />
      <VariablesButton messages={messages} value={variables} onChange={onVariablesChange} />
      <RegistryButton onOpen={onOpenRegistry} />
    </div>
  );
};
