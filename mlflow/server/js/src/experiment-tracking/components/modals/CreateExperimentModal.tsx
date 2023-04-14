import React, { Component } from 'react';
import { connect } from 'react-redux';
import { withRouter } from 'react-router-dom';
import debounce from 'lodash/debounce';

import Routes from '../../routes';
import { GenericInputModal } from './GenericInputModal';
import { CreateExperimentForm, EXP_NAME_FIELD, ARTIFACT_LOCATION } from './CreateExperimentForm';
import { getExperimentNameValidator } from '../../../common/forms/validations';

import { createExperimentApi, searchExperimentsApi } from '../../actions';
import { getExperiments } from '../../reducers/Reducers';

type CreateExperimentModalImplProps = {
  isOpen?: boolean;
  onClose: (...args: any[]) => any;
  experimentNames: string[];
  createExperimentApi: (...args: any[]) => any;
  searchExperimentsApi: (...args: any[]) => any;
  history: any;
};

export class CreateExperimentModalImpl extends Component<CreateExperimentModalImplProps> {
  handleCreateExperiment = async (values: any) => {
    // get values of input fields
    const experimentName = values[EXP_NAME_FIELD];
    const artifactLocation = values[ARTIFACT_LOCATION];

    // Both createExperimentApi and searchExperimentsApi calls need to be fulfilled sequentially
    // before redirecting the user to the newly created experiment page (history.push())
    const response = await this.props.createExperimentApi(experimentName, artifactLocation);
    await this.props.searchExperimentsApi();

    const {
      value: { experiment_id: newExperimentId },
    } = response;
    if (newExperimentId) {
      this.props.history.push(Routes.getExperimentPageRoute(newExperimentId));
    }
  };

  debouncedExperimentNameValidator = debounce(
    getExperimentNameValidator(() => this.props.experimentNames),
    400,
  );

  render() {
    const { isOpen } = this.props;
    return (
      <GenericInputModal
        title='Create Experiment'
        okText='Create'
        isOpen={isOpen}
        handleSubmit={this.handleCreateExperiment}
        onClose={this.props.onClose}
      >
        {/* @ts-expect-error TS(2322): Type '{ validator: ((rule: any, value: any, callba... Remove this comment to see the full error message */}
        <CreateExperimentForm validator={this.debouncedExperimentNameValidator} />
      </GenericInputModal>
    );
  }
}

const mapStateToProps = (state: any) => {
  const experiments = getExperiments(state);
  const experimentNames = experiments.map((e) => (e as any).getName());
  return { experimentNames };
};

const mapDispatchToProps = {
  createExperimentApi,
  searchExperimentsApi,
};

export const CreateExperimentModal: TODOBrokenReactRouterType = withRouter(
  // @ts-expect-error TS(2345): Argument of type 'ConnectedComponent<typeof Create... Remove this comment to see the full error message
  connect(mapStateToProps, mapDispatchToProps)(CreateExperimentModalImpl),
);
