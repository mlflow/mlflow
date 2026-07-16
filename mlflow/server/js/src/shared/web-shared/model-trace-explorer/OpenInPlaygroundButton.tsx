import { Button, BeakerIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { getExperimentPagePlaygroundRoute } from './routes';
import { useInRouterContext, useNavigate, useParams } from './RoutingUtils';

interface OpenInPlaygroundButtonProps {
  traceId: string | undefined;
  spanId?: string;
  // Falls back to the experiment id in the current route when not provided.
  experimentId?: string;
  size?: 'small' | 'middle';
}

const OpenInPlaygroundButtonImpl = ({
  traceId,
  spanId,
  experimentId: experimentIdProp,
  size,
}: OpenInPlaygroundButtonProps) => {
  const navigate = useNavigate();
  const { experimentId: experimentIdFromRoute } = useParams<{ experimentId: string }>();
  const experimentId = experimentIdProp ?? experimentIdFromRoute;

  if (!experimentId || !traceId) {
    return null;
  }

  return (
    <Button
      componentId="shared.model-trace-explorer.open-in-playground"
      icon={<BeakerIcon />}
      size={size}
      onClick={() => navigate(getExperimentPagePlaygroundRoute(experimentId, { traceId, spanId }))}
    >
      <FormattedMessage
        defaultMessage="Open in Playground"
        description="Button on the trace UI that opens the MLflow playground pre-filled with the trace's input and prompt so the user can test different prompts"
      />
    </Button>
  );
};

/**
 * "Open in Playground" action shown on the trace UI (both trace-level and span-level).
 *
 * Clicking it navigates to the experiment's playground page, passing the trace id (and, at the
 * span level, the span id) as query params. The playground reads those params to pre-fill itself
 * with the captured input + prompt/model so users can iterate on the prompt without leaving MLflow.
 *
 * Renders nothing when the experiment id cannot be resolved from the current route (e.g. the trace
 * explorer is embedded outside an experiment page) or when there is no trace id to link to. The
 * router-context guard also keeps router-less embeddings (e.g. the OSS notebook renderer) working,
 * since `useNavigate`/`useParams` throw outside a `<Router>`.
 */
export const OpenInPlaygroundButton = (props: OpenInPlaygroundButtonProps) => {
  const inRouterContext = useInRouterContext();

  if (!inRouterContext) {
    return null;
  }

  return <OpenInPlaygroundButtonImpl {...props} />;
};
