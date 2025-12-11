import { useParams } from '../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { useEditEndpointForm, getReadableErrorMessage } from '../hooks/useEditEndpointForm';
import { EditEndpointFormRenderer } from '../components/edit-endpoint/EditEndpointFormRenderer';

/**
 * Container component for editing endpoints.
 * Uses the container/renderer pattern:
 * - useEditEndpointForm: Contains all business logic (state, mutations, handlers)
 * - EditEndpointFormRenderer: Pure presentational component
 *
 * This separation allows easy swapping of renderers (full page, modal, etc.)
 * while reusing the business logic.
 */
const EditEndpointPage = () => {
  const { endpointId } = useParams<{ endpointId: string }>();

  const {
    form,
    isLoadingEndpoint,
    isSubmitting,
    loadError,
    mutationError,
    resetErrors,
    endpoint,
    isFormComplete,
    handleSubmit,
    handleCancel,
    handleNameBlur,
  } = useEditEndpointForm(endpointId ?? '');

  return (
    <EditEndpointFormRenderer
      form={form}
      isLoadingEndpoint={isLoadingEndpoint}
      isSubmitting={isSubmitting}
      loadError={loadError}
      mutationError={mutationError}
      errorMessage={getReadableErrorMessage(mutationError)}
      resetErrors={resetErrors}
      endpointId={endpointId ?? ''}
      endpointName={endpoint?.name}
      isFormComplete={isFormComplete}
      onSubmit={handleSubmit}
      onCancel={handleCancel}
      onNameBlur={handleNameBlur}
    />
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, EditEndpointPage);
