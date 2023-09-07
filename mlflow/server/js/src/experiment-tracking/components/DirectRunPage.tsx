import { PageWrapper, Skeleton } from '@databricks/design-system';
import { useEffect, useState } from 'react';
import { connect, useDispatch } from 'react-redux';
import { RouteComponentProps } from 'react-router';
import { useParams, Navigate } from 'react-router-dom-v5-compat';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import Utils from '../../common/utils/Utils';
import { StateWithEntities } from '../../redux-types';
import { getRunApi } from '../actions';
import Routes from '../routes';
import { PageNotFoundView } from './PageNotFoundView';
import { WithRouterNextProps, withRouterNext } from '../../common/utils/withRouterNext';

export const DirectRunPageImpl = (props: any) => {
  const { runUuid } = useParams<{ runUuid: string }>();
  const [error, setError] = useState<ErrorWrapper>();

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

  // If encountered 404 error, display a proper component
  if (error?.status === 404) {
    return <PageNotFoundView />;
  }

  // If the run info is loaded, redirect to the run page
  if (props.runInfo?.experiment_id) {
    return (
      <Navigate
        to={Routes.getRunPageRoute(props.runInfo.experiment_id, props.runInfo.run_uuid)}
        replace
      />
    );
  }

  // If the run is loading, display skeleton
  return (
    <PageWrapper>
      <Skeleton />
    </PageWrapper>
  );
};

export const DirectRunPage = withRouterNext(
  connect((state: StateWithEntities, ownProps: WithRouterNextProps<{ runUuid: string }>) => {
    return { runInfo: state.entities.runInfosByUuid[ownProps.params.runUuid] };
  })(DirectRunPageImpl),
);
