import { useDispatch, useSelector } from 'react-redux';
import type { ReduxState, ThunkDispatch } from '../../../../redux-types';
import { useEffect, useMemo } from 'react';
import { getRunApi } from '../../../actions';
import { ParagraphSkeleton } from '@databricks/design-system';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { FormattedMessage } from 'react-intl';
import { shouldEnableGraphQLRunDetailsPage } from '../../../../common/utils/FeatureUtils';
import { useGetRunQuery } from '../hooks/useGetRunQuery';

export const RunViewParentRunBox = ({ parentRunUuid }: { parentRunUuid: string }) => {
  const dispatch = useDispatch<ThunkDispatch>();

  const parentRunInfoRedux = useSelector(({ entities }: ReduxState) => {
    return entities.runInfosByUuid[parentRunUuid];
  });

  const parentRunInfoGraphql = useGetRunQuery({
    runUuid: parentRunUuid,
    disabled: !shouldEnableGraphQLRunDetailsPage(),
  });

  const parentRunInfo = useMemo(() => {
    return shouldEnableGraphQLRunDetailsPage() ? parentRunInfoGraphql?.data?.info : parentRunInfoRedux;
  }, [parentRunInfoGraphql, parentRunInfoRedux]);

  useEffect(() => {
    // Don't call REST API if GraphQL is enabled
    if (shouldEnableGraphQLRunDetailsPage()) {
      return;
    }
    if (!parentRunInfo) {
      dispatch(getRunApi(parentRunUuid));
    }
  }, [dispatch, parentRunUuid, parentRunInfo]);

  if (!parentRunInfo) {
    return (
      <ParagraphSkeleton
        loading
        label={
          <FormattedMessage
            defaultMessage="Parent run name loading"
            description="Run page > Overview > Parent run name loading"
          />
        }
      />
    );
  }

  if (!parentRunInfo.experimentId || !parentRunInfo.runUuid) {
    return null;
  }

  return (
    <Link to={Routes.getRunPageRoute(parentRunInfo.experimentId, parentRunInfo.runUuid)}>{parentRunInfo.runName}</Link>
  );
};
