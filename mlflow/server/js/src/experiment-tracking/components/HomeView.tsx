/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { connect } from 'react-redux';
import { Navigate } from '../../common/utils/RoutingUtils';
import { PageWrapper, LegacySkeleton } from '@databricks/design-system';
import ExperimentListView from './ExperimentListView';
import { getExperiments } from '../reducers/Reducers';
import { NoExperimentView } from './NoExperimentView';
import Utils from '../../common/utils/Utils';
import Routes from '../routes';

// Lazy load experiment page in order to promote bundle splitting
const ExperimentPage = React.lazy(() => import('./experiment-page/ExperimentPage'));

export const getFirstActiveExperiment = (experiments: any) => {
  const sorted = experiments.concat().sort(Utils.compareExperiments);
  return sorted.find((e: any) => e.lifecycle_stage === 'active');
};

type HomeViewProps = {
  experiments?: any[];
  experimentIds?: string[];
  compareExperiments?: boolean;
};

class HomeView extends Component<HomeViewProps> {
  render() {
    const { experimentIds, experiments, compareExperiments } = this.props;
    // @ts-expect-error TS(2532): Object is possibly 'undefined'.
    const hasExperiments = experimentIds?.length > 0;

    if (experimentIds === undefined) {
      const firstExp = getFirstActiveExperiment(experiments);
      if (firstExp) {
        return <Navigate to={Routes.getExperimentPageRoute(firstExp.experiment_id)} replace />;
      }
    }

    // @ts-expect-error TS(4111): Property 'HIDE_EXPERIMENT_LIST' comes from an inde... Remove this comment to see the full error message
    if (process.env.HIDE_EXPERIMENT_LIST === 'true') {
      return (
        <>
          {hasExperiments ? (
            <PageWrapper css={{ height: `100%`, paddingTop: 16, width: '100%' }}>
              <React.Suspense fallback={<LegacySkeleton />}>
                <ExperimentPage
                  // @ts-expect-error TS(2322): Type '{ experimentIds: string[] | undefined; compa... Remove this comment to see the full error message
                  experimentIds={experimentIds}
                  compareExperiments={compareExperiments}
                />
              </React.Suspense>
            </PageWrapper>
          ) : (
            <NoExperimentView />
          )}
        </>
      );
    }
    return (
      <div css={{ display: 'flex', height: 'calc(100% - 60px)' }}>
        <div css={{ height: '100%', paddingTop: 24, display: 'flex' }}>
          {/* @ts-expect-error TS(2322): Type '{ activeExperimentIds: string[]; experiments... Remove this comment to see the full error message */}
          <ExperimentListView activeExperimentIds={experimentIds || []} experiments={experiments} />
        </div>
        <PageWrapper css={{ height: '100%', flex: '1', paddingTop: 24 }}>
          {hasExperiments ? (
            <React.Suspense fallback={<LegacySkeleton />}>
              <ExperimentPage
                // @ts-expect-error TS(2322): Type '{ experimentIds: string[] | undefined; compa... Remove this comment to see the full error message
                experimentIds={experimentIds}
                compareExperiments={compareExperiments}
              />
            </React.Suspense>
          ) : (
            <NoExperimentView />
          )}
        </PageWrapper>
      </div>
    );
  }
}

const mapStateToProps = (state: any) => {
  const experiments = getExperiments(state);
  return { experiments };
};

export default connect(mapStateToProps)(HomeView);
