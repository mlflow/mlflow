import { createMLflowRoutePath } from './RoutingUtils';

export const getExperimentChatSessionPageRoute = (experimentId: string, sessionId: string) => {
  return createMLflowRoutePath(`/experiments/${experimentId}/chat-sessions/${sessionId}`);
};
