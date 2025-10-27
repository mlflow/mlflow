import { PageWrapper, LegacySkeleton } from '@databricks/design-system';
import { useEffect, useState } from 'react';
import { connect, useDispatch } from 'react-redux';
import { useParams, useNavigate } from '../../common/utils/RoutingUtils';
import type { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import Utils from '../../common/utils/Utils';
import type { ReduxState } from '../../redux-types';
import { getRunApi } from '../actions';
import Routes from '../routes';
import { PageNotFoundView } from '../../common/components/PageNotFoundView';
import type { WithRouterNextProps } from '../../common/utils/withRouterNext';
import { withRouterNext } from '../../common/utils/withRouterNext';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';

const DirectRunPageImpl = (props: any) => {
  const { runUuid } = useParams<{ runUuid: string }>();
  const [error, setError] = useState<ErrorWrapper>();
  const navigate = useNavigate();

  const dispatch = useDispatch();

  // Reset error after changing requested run
  useEffect(() => {
    setError(undefined);
  }, [runUuid]);

  useEffect(() => {
    // Start fetching run info if it doesn't exist in the store yet
    if (!props.runInfo && runUuid) {
      const action = getRunApi(runUuid);
      action.payload.catch((e) => {
        Utils.logErrorAndNotifyUser(e);
        setError(e);
      });
      dispatch(action);
    }
  }, [dispatch, runUuid, props.runInfo]);

  useEffect(() => {
    if (props.runInfo?.experimentId) {
      navigate(Routes.getRunPageRoute(props.runInfo.experimentId, props.runInfo.runUuid), {
        replace: true,
      });
    }
  }, [navigate, props.runInfo]);

  // If encountered 404 error, display a proper component
  if (error?.status === 404) {
    return <PageNotFoundView />;
  }

  // If the run is loading, display skeleton
  return (
    <PageWrapper>
      <LegacySkeleton />
    </PageWrapper>
  );
};

const DirectRunPageWithRouter = withRouterNext(
  connect((state: ReduxState, ownProps: WithRouterNextProps<{ runUuid: string }>) => {
    return { runInfo: state.entities.runInfosByUuid[ownProps.params.runUuid] };
  })(DirectRunPageImpl),
);

export const DirectRunPage = withErrorBoundary(ErrorUtils.mlflowServices.RUN_TRACKING, DirectRunPageWithRouter);

export default DirectRunPage;
