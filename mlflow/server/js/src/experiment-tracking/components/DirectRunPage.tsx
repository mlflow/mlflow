import { PageWrapper, Skeleton } from '@databricks/design-system';
import { useEffect, useState } from 'react';
import { connect, useDispatch } from 'react-redux';
import { RouteComponentProps } from 'react-router';
import { Redirect, useRouteMatch } from 'react-router-dom';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import Utils from '../../common/utils/Utils';
import { StateWithEntities } from '../../redux-types';
import { getRunApi } from '../actions';
import Routes from '../routes';
import { PageNotFoundView } from './PageNotFoundView';

export const DirectRunPageImpl = (props: any) => {
  const { params } = useRouteMatch<{ runUuid: string }>();
  const [error, setError] = useState<ErrorWrapper>();

  const dispatch = useDispatch();

  // Reset error after changing requested run
  useEffect(() => {
    setError(undefined);
  }, [params.runUuid]);

  useEffect(() => {
    // Start fetching run info if it doesn't exist in the store yet
    if (!props.runInfo && params.runUuid) {
      const action = getRunApi(params.runUuid);
      action.payload.catch((e) => {
        Utils.logErrorAndNotifyUser(e);
        setError(e);
      });
      dispatch(action);
    }
  }, [dispatch, params.runUuid, props.runInfo]);

  // If encountered 404 error, display a proper component
  if (error?.status === 404) {
    return <PageNotFoundView />;
  }

  // If the run info is loaded, redirect to the run page
  if (props.runInfo?.experiment_id) {
    return (
      <Redirect
        to={Routes.getRunPageRoute(props.runInfo.experiment_id, props.runInfo.run_uuid)}
        push={false}
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

export const DirectRunPage = connect(
  (state: StateWithEntities, ownProps: RouteComponentProps<{ runUuid: string }>) => {
    return { runInfo: state.entities.runInfosByUuid[ownProps.match.params.runUuid] };
  },
)(DirectRunPageImpl);
