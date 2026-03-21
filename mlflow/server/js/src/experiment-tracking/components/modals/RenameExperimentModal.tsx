/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { connect } from 'react-redux';
import debounce from 'lodash/debounce';

import { GenericInputModal } from './GenericInputModal';
import { RenameForm, NEW_NAME_FIELD } from './RenameForm';
import { getExperimentNameValidator } from '../../../common/forms/validations';

import { updateExperimentApi, getExperimentApi } from '../../actions';
import { getExperiments } from '../../reducers/Reducers';
import Utils from '../../../common/utils/Utils';

type RenameExperimentModalImplProps = {
  isOpen?: boolean;
  experimentId?: string;
  experimentName?: string;
  experimentNames: string[];
  onClose: (...args: any[]) => any;
  updateExperimentApi: (...args: any[]) => any;
  getExperimentApi: (...args: any[]) => any;
  onExperimentRenamed: () => void;
};

class RenameExperimentModalImpl extends Component<RenameExperimentModalImplProps> {
  handleRenameExperiment = (values: any) => {
    // get value of input field
    const newExperimentName = values[NEW_NAME_FIELD];
    const updateExperimentPromise = this.props
      .updateExperimentApi(this.props.experimentId, newExperimentName)
      .then(() => {
        this.props.getExperimentApi(this.props.experimentId);
        this.props.onExperimentRenamed();
      })
      .catch((e: any) => Utils.logErrorAndNotifyUser(e));

    return updateExperimentPromise;
  };

  debouncedExperimentNameValidator = debounce(
    getExperimentNameValidator(() => this.props.experimentNames),
    400,
  );

  render() {
    const { isOpen, experimentName } = this.props;
    return (
      <GenericInputModal
        title="Rename Experiment"
        okText="Save"
        isOpen={isOpen}
        handleSubmit={this.handleRenameExperiment}
        onClose={this.props.onClose}
      >
        {/* @ts-expect-error TS(2769): No overload matches this call. */}
        <RenameForm
          type="experiment"
          name={experimentName}
          visible={isOpen}
          validator={this.debouncedExperimentNameValidator}
        />
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
  updateExperimentApi,
  getExperimentApi,
};

export const RenameExperimentModal = connect(mapStateToProps, mapDispatchToProps)(RenameExperimentModalImpl);
