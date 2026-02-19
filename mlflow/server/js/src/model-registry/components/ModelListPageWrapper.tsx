import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { ModelListPage } from './ModelListPage';
const ModelListPageWrapperImpl = () => {
  return <ModelListPage />;
};
export const ModelListPageWrapper = withErrorBoundary(
  ErrorUtils.mlflowServices.MODEL_REGISTRY,
  ModelListPageWrapperImpl,
);

export default ModelListPageWrapper;
