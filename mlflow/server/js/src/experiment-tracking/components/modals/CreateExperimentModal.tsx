/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { connect } from 'react-redux';
import type { NavigateFunction } from '../../../common/utils/RoutingUtils';
import debounce from 'lodash/debounce';

import Routes from '../../routes';
import { GenericInputModal } from './GenericInputModal';
import { CreateExperimentForm, EXP_NAME_FIELD, ARTIFACT_LOCATION } from './CreateExperimentForm';
import { getExperimentNameValidator } from '../../../common/forms/validations';

import { createExperimentApi } from '../../actions';
import { getExperiments } from '../../reducers/Reducers';
import { withRouterNext } from '../../../common/utils/withRouterNext';

type CreateExperimentModalImplProps = {
  isOpen?: boolean;
  onClose: (...args: any[]) => any;
  experimentNames: string[];
  createExperimentApi: (...args: any[]) => any;
  onExperimentCreated: () => void;
  navigate: NavigateFunction;
};

export class CreateExperimentModalImpl extends Component<CreateExperimentModalImplProps> {
  handleCreateExperiment = async (values: any) => {
    // get values of input fields
    const experimentName = values[EXP_NAME_FIELD];
    const artifactLocation = values[ARTIFACT_LOCATION];

    // createExperimentApi call needs to be fulfilled before redirecting the user to the newly
    // created experiment page (history.push())
    const response = await this.props.createExperimentApi(experimentName, artifactLocation);
    this.props.onExperimentCreated();

    const {
      value: { experiment_id: newExperimentId },
    } = response;
    if (newExperimentId) {
      this.props.navigate(Routes.getExperimentPageRoute(newExperimentId));
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
        title="Create Experiment"
        okText="Create"
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
  const experimentNames = experiments.map((e) => e.name);
  return { experimentNames };
};

const mapDispatchToProps = {
  createExperimentApi,
};

export const CreateExperimentModal = withRouterNext(
  connect(mapStateToProps, mapDispatchToProps)(CreateExperimentModalImpl),
);
