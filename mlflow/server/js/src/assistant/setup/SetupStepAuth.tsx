import type { AuthState } from '../types';
import { ClaudeCodeAuth } from './ClaudeCodeAuth';
import { CodexAuth } from './CodexAuth';
import { MLflowGatewayAuth } from './MLflowGatewayAuth';
import { OllamaAuth } from './OllamaAuth';

interface SetupStepAuthProps {
  provider: string;
  cachedAuthStatus?: AuthState;
  onAuthStatusChange: (status: AuthState) => void;
  onBack: () => void;
  onContinue: () => void;
}

export const SetupStepAuth = ({
  provider,
  cachedAuthStatus,
  onAuthStatusChange,
  onBack,
  onContinue,
}: SetupStepAuthProps) => {
  if (provider === 'ollama') {
    return (
      <OllamaAuth
        cachedAuthStatus={cachedAuthStatus}
        onAuthStatusChange={onAuthStatusChange}
        onBack={onBack}
        onContinue={onContinue}
      />
    );
  }

  if (provider === 'codex') {
    return (
      <CodexAuth
        cachedAuthStatus={cachedAuthStatus}
        onAuthStatusChange={onAuthStatusChange}
        onBack={onBack}
        onContinue={onContinue}
      />
    );
  }

  if (provider === 'mlflow_gateway') {
    // MLflowGatewayAuth has no auth status to cache — the gateway lives
    // in the same MLflow server, so we drop the cachedAuthStatus /
    // onAuthStatusChange props that the other Auth components consume.
    return <MLflowGatewayAuth onBack={onBack} onContinue={onContinue} />;
  }

  return (
    <ClaudeCodeAuth
      cachedAuthStatus={cachedAuthStatus}
      onAuthStatusChange={onAuthStatusChange}
      onBack={onBack}
      onContinue={onContinue}
    />
  );
};
