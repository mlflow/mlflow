import type { ReactNode } from 'react';
import { Button, BeakerIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { getExperimentPagePlaygroundRoute } from './routes';
import { useInRouterContext, useParams, useWorkspaceNavigate } from './RoutingUtils';

interface OpenInPlaygroundTarget {
  traceId: string | undefined;
  spanId?: string;
  // Falls back to the experiment id in the current route when not provided.
  experimentId?: string;
}

interface OpenInPlaygroundActionProps extends OpenInPlaygroundTarget {
  children: (openInPlayground: () => void) => ReactNode;
}

const OpenInPlaygroundActionImpl = ({
  traceId,
  spanId,
  experimentId: experimentIdProp,
  children,
}: OpenInPlaygroundActionProps) => {
  const navigate = useWorkspaceNavigate();
  const { experimentId: experimentIdFromRoute } = useParams<{ experimentId: string }>();
  const experimentId = experimentIdProp ?? experimentIdFromRoute;

  if (!experimentId || !traceId) {
    return null;
  }

  return <>{children(() => navigate(getExperimentPagePlaygroundRoute(experimentId, { traceId, spanId })))}</>;
};

/**
 * "Open in Playground" action shown on the trace UI (both trace-level and span-level). Renders
 * `children` with a callback that opens the playground, so each surface keeps its own trigger style.
 *
 * The callback navigates to the experiment's playground page, passing the trace id (and, at the
 * span level, the span id) as query params. The playground reads those params to pre-fill itself
 * with the captured input + prompt/model so users can iterate on the prompt without leaving MLflow.
 *
 * Renders nothing when the experiment id cannot be resolved from the current route (e.g. the trace
 * explorer is embedded outside an experiment page) or when there is no trace id to link to. The
 * router-context guard also keeps router-less embeddings (e.g. the OSS notebook renderer) working,
 * since `useWorkspaceNavigate`/`useParams` throw outside a `<Router>`.
 */
export const OpenInPlaygroundAction = (props: OpenInPlaygroundActionProps) => {
  const inRouterContext = useInRouterContext();

  if (!inRouterContext) {
    return null;
  }

  return <OpenInPlaygroundActionImpl {...props} />;
};

export const OpenInPlaygroundButton = ({ size, ...target }: OpenInPlaygroundTarget & { size?: 'small' | 'middle' }) => (
  <OpenInPlaygroundAction {...target}>
    {(openInPlayground) => (
      <Button
        componentId="shared.model-trace-explorer.open-in-playground"
        icon={<BeakerIcon />}
        size={size}
        onClick={openInPlayground}
      >
        <FormattedMessage
          defaultMessage="Open in Playground"
          description="Button on the trace UI that opens the MLflow playground pre-filled with the trace's input and prompt so the user can test different prompts"
        />
      </Button>
    )}
  </OpenInPlaygroundAction>
);
