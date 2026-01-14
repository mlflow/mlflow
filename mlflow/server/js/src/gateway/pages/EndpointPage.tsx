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
    resetErrors,
    endpoint,
    isFormComplete,
    hasChanges,
    handleSubmit,
    handleCancel,
    handleNameBlur,
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
        resetErrors={resetErrors}
        endpointName={endpoint?.name}
        experimentId={endpoint?.experiment_id}
        isFormComplete={isFormComplete}
        hasChanges={hasChanges}
        onSubmit={handleSubmit}
        onCancel={handleCancel}
        onNameBlur={handleNameBlur}
      />
    </FormProvider>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, EndpointPage);
