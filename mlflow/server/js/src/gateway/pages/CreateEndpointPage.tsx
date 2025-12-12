import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { useCreateEndpointForm, getReadableErrorMessage } from '../hooks/useCreateEndpointForm';
import { CreateEndpointFormRenderer } from '../components/create-endpoint/CreateEndpointFormRenderer';

/**
 * Container component for creating endpoints.
 * Uses the container/renderer pattern:
 * - useCreateEndpointForm: Contains all business logic (state, mutations, handlers)
 * - CreateEndpointFormRenderer: Pure presentational component
 *
 * This separation allows easy swapping of renderers (full page, modal, etc.)
 * while reusing the business logic.
 */
const CreateEndpointPage = () => {
  const {
    form,
    isLoading,
    error,
    resetErrors,
    selectedModel,
    isFormComplete,
    handleSubmit,
    handleCancel,
    handleNameBlur,
  } = useCreateEndpointForm();

  return (
    <CreateEndpointFormRenderer
      form={form}
      isLoading={isLoading}
      error={error}
      errorMessage={getReadableErrorMessage(error)}
      resetErrors={resetErrors}
      selectedModel={selectedModel}
      isFormComplete={isFormComplete}
      onSubmit={handleSubmit}
      onCancel={handleCancel}
      onNameBlur={handleNameBlur}
    />
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, CreateEndpointPage);
