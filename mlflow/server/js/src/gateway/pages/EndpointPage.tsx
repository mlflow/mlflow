import { useParams } from '../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { FormProvider } from 'react-hook-form';
import { useEditEndpointForm, getReadableErrorMessage } from '../hooks/useEditEndpointForm';
import { EditEndpointFormRenderer } from '../components/edit-endpoint/EditEndpointFormRenderer';

const EndpointPage = () => {
  const { endpointId } = useParams<{ endpointId: string }>();

  const {
    form,
    isLoadingEndpoint,
    isSubmitting,
    loadError,
    mutationError,
    endpoint,
    existingEndpoints,
    isFormComplete,
    hasChanges,
    handleSubmit,
    handleCancel,
    handleNameUpdate,
  } = useEditEndpointForm(endpointId ?? '');

  return (
    <FormProvider {...form}>
      <EditEndpointFormRenderer
        form={form}
        isLoadingEndpoint={isLoadingEndpoint}
        isSubmitting={isSubmitting}
        loadError={loadError}
        mutationError={mutationError}
        errorMessage={getReadableErrorMessage(mutationError)}
        endpoint={endpoint}
        existingEndpoints={existingEndpoints}
        isFormComplete={isFormComplete}
        hasChanges={hasChanges}
        onSubmit={handleSubmit}
        onCancel={handleCancel}
        onNameUpdate={handleNameUpdate}
      />
    </FormProvider>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, EndpointPage);
