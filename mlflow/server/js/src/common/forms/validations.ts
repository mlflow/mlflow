import { MlflowService } from '../../experiment-tracking/sdk/MlflowService';
import { Services as ModelRegistryService } from '../../model-registry/services';

export const getExperimentNameValidator = (getExistingExperimentNames: () => string[]) => {
  return (rule: unknown, value: string | undefined, callback: (arg?: string) => void) => {
    if (!value) {
      // no need to execute below validations when no value is entered
      // eslint-disable-next-line callback-return
      callback(undefined);
    } else if (getExistingExperimentNames().includes(value)) {
      // getExistingExperimentNames returns the names of all active experiments
      // check whether the passed value is part of the list
      // eslint-disable-next-line callback-return
      callback(`Experiment "${value}" already exists.`);
    } else {
      // on-demand validation whether experiment already exists in deleted state
      MlflowService.getExperimentByName({ experiment_name: value })
        .then((res) =>
          callback(`Experiment "${value}" already exists in deleted state.
                                 You can restore the experiment, or permanently delete the
                                 experiment from the .trash folder (under tracking server's
                                 root folder) in order to use this experiment name again.`),
        )
        .catch((e) => callback(undefined)); // no experiment returned
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
