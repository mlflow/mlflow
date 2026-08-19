import { useCallback, useEffect, useRef, useState } from 'react';
import { shallowEqual, useDispatch, useSelector } from 'react-redux';
import { FormattedMessage, useIntl } from 'react-intl';
import debounce from 'lodash/debounce';

import { Modal } from '@databricks/design-system';
import { useNavigate } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { createExperimentApi } from '../../actions';
import { getExperiments } from '../../reducers/Reducers';
import { getExperimentNameValidator } from '../../../common/forms/validations';
import Utils from '../../../common/utils/Utils';
import { CreateExperimentForm } from './CreateExperimentForm';
import type { ReduxState, ThunkDispatch } from '@mlflow/mlflow/src/redux-types';

type CreateExperimentModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onExperimentCreated: () => void;
};

export const CreateExperimentModal = ({ isOpen, onClose, onExperimentCreated }: CreateExperimentModalProps) => {
  const intl = useIntl();
  const navigate = useNavigate();
  const dispatch = useDispatch<ThunkDispatch>();
  const experimentNames = useSelector((state: ReduxState) => getExperiments(state).map((e) => e.name), shallowEqual);

  const [experimentName, setExperimentName] = useState('');
  const [artifactLocation, setArtifactLocation] = useState('');
  const [nameError, setNameError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const experimentNamesRef = useRef(experimentNames);
  experimentNamesRef.current = experimentNames;

  const debouncedValidatorRef = useRef(
    debounce((value: string) => {
      const validator = getExperimentNameValidator(() => experimentNamesRef.current);
      validator(null, value, (error?: string) => {
        setNameError(error ?? '');
      });
    }, 400),
  );

  const resetFormState = useCallback(() => {
    debouncedValidatorRef.current.cancel();
    setExperimentName('');
    setArtifactLocation('');
    setNameError('');
    setIsSubmitting(false);
  }, []);

  useEffect(() => {
    if (isOpen) {
      resetFormState();
    }
    const cleanup = debouncedValidatorRef.current;
    return () => cleanup.cancel();
  }, [isOpen, resetFormState]);

  const handleClose = () => {
    resetFormState();
    onClose();
  };

  const handleNameChange = (value: string) => {
    setExperimentName(value);
    if (!value.trim()) {
      debouncedValidatorRef.current.cancel();
      setNameError(
        intl.formatMessage({
          defaultMessage: 'Please input a new name for the new experiment.',
          description: 'Error message for name requirement in create experiment for MLflow',
        }),
      );
    } else {
      debouncedValidatorRef.current(value);
    }
  };

  const isCreateDisabled = !experimentName.trim() || !!nameError || isSubmitting;

  const validateExperimentName = async (value: string): Promise<string> => {
    const validator = getExperimentNameValidator(() => experimentNamesRef.current);
    return new Promise((resolve) => {
      validator(undefined, value, (error?: string) => resolve(error ?? ''));
    });
  };

  const handleSubmit = async () => {
    if (isCreateDisabled) return;
    debouncedValidatorRef.current.cancel();
    setIsSubmitting(true);
    const validationError = await validateExperimentName(experimentName);
    if (validationError) {
      setNameError(validationError);
      setIsSubmitting(false);
      return;
    }
    setNameError('');
    try {
      // @ts-expect-error -- createExperimentApi has loosely typed params and thunk return
      const response: { value?: { experiment_id?: string } } = await dispatch(
        // @ts-expect-error -- artifactPath param is typed as `undefined` due to default value
        createExperimentApi(experimentName, artifactLocation || undefined),
      );
      onExperimentCreated();
      handleClose();
      const newExperimentId = response?.value?.experiment_id;
      if (newExperimentId) {
        navigate(Routes.getExperimentPageRoute(newExperimentId));
      }
    } catch (e) {
      setIsSubmitting(false);
      Utils.logErrorAndNotifyUser(e);
    }
  };

  return (
    <Modal
      componentId="mlflow.experiment.create_experiment_modal"
      data-testid="mlflow-input-modal"
      title={
        <FormattedMessage
          defaultMessage="Create Experiment"
          description="Title for create experiment modal in MLflow"
        />
      }
      visible={isOpen}
      onOk={handleSubmit}
      okText={intl.formatMessage({
        defaultMessage: 'Create',
        description: 'Confirm button text for create experiment modal',
      })}
      okButtonProps={{ disabled: isCreateDisabled }}
      confirmLoading={isSubmitting}
      onCancel={handleClose}
    >
      <CreateExperimentForm
        experimentName={experimentName}
        artifactLocation={artifactLocation}
        nameError={nameError}
        onNameChange={handleNameChange}
        onArtifactLocationChange={setArtifactLocation}
      />
    </Modal>
  );
};
