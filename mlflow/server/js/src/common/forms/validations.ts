import { MlflowService } from '../../experiment-tracking/sdk/MlflowService';
import { Services as ModelRegistryService } from '../../model-registry/services';
import { ErrorCodes } from '../constants';

const isResourceDoesNotExistError = (error: unknown) =>
  typeof error === 'object' &&
  error !== null &&
  'getErrorCode' in error &&
  typeof error.getErrorCode === 'function' &&
  error.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST;

export const getExperimentNameValidator = (getExistingExperimentNames: () => string[]) => {
  return (rule: unknown, value: string | undefined, callback: (arg?: string) => void) => {
    if (!value) {
      // no need to execute below validations when no value is entered
      callback(undefined);
    } else if (getExistingExperimentNames().includes(value)) {
      // getExistingExperimentNames returns the names of all active experiments
      // check whether the passed value is part of the list
      callback(`Experiment "${value}" already exists.`);
    } else {
      // on-demand validation whether experiment already exists (active or deleted)
      MlflowService.getExperimentByName({ experiment_name: value })
        .then((res) => {
          if (res.experiment.lifecycleStage === 'deleted') {
            callback(`Experiment "${value}" already exists in deleted state.
                                 You can restore the experiment, or permanently delete the
                                 experiment from the .trash folder (under tracking server's
                                 root folder) in order to use this experiment name again.`);
          } else {
            callback(`Experiment "${value}" already exists.`);
          }
        })
        .catch((e) => {
          if (isResourceDoesNotExistError(e)) {
            callback(undefined);
          } else {
            callback('Could not validate experiment name. Please try again.');
          }
        });
    }
  };
};

export const modelNameValidator = (rule: unknown, name: string | undefined, callback: (arg?: string) => void) => {
  if (!name) {
    callback(undefined);
    return;
  }

  ModelRegistryService.getRegisteredModel({ name: name })
    .then(() => callback(`Model "${name}" already exists.`))
    .catch((e) => callback(undefined));
};
