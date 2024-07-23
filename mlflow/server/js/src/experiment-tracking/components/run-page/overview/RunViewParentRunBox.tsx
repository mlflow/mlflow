import { useDispatch, useSelector } from 'react-redux';
import { KeyValueEntity } from '../../../types';
import { ReduxState, ThunkDispatch } from '../../../../redux-types';
import { useEffect } from 'react';
import { getRunApi } from '../../../actions';
import { ParagraphSkeleton } from '@databricks/design-system';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { FormattedMessage } from 'react-intl';

export const RunViewParentRunBox = ({ parentRunUuid }: { parentRunUuid: string }) => {
  const parentRunInfo = useSelector(({ entities }: ReduxState) => {
    return entities.runInfosByUuid[parentRunUuid];
  });
  const dispatch = useDispatch<ThunkDispatch>();
  useEffect(() => {
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
  return (
    <Link to={Routes.getRunPageRoute(parentRunInfo.experimentId, parentRunInfo.runUuid)}>{parentRunInfo.runName}</Link>
  );
};
