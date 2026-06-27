import { useCallback, useEffect, useRef, useState } from 'react';
import { shallowEqual, useDispatch, useSelector } from 'react-redux';
import { FormattedMessage, useIntl } from 'react-intl';
import debounce from 'lodash/debounce';

import { Alert, Button, Modal, Spacer, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { useNavigate } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { createExperimentApi } from '../../actions';
import { getExperiments } from '../../reducers/Reducers';
import { getExperimentNameValidator } from '../../../common/forms/validations';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import { CreateExperimentForm } from './CreateExperimentForm';
import type { ReduxState, ThunkDispatch } from '@mlflow/mlflow/src/redux-types';

const MAX_EXPERIMENT_NAME_LENGTH = 500;

type CreateExperimentModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onExperimentCreated: () => void;
};

export const CreateExperimentModal = ({ isOpen, onClose, onExperimentCreated }: CreateExperimentModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const dispatch = useDispatch<ThunkDispatch>();
  const experimentNames = useSelector((state: ReduxState) => getExperiments(state).map((e) => e.name), shallowEqual);

  const [experimentName, setExperimentName] = useState('');
  const [artifactLocation, setArtifactLocation] = useState('');
  const [nameError, setNameError] = useState('');
  const [submitError, setSubmitError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const experimentNamesRef = useRef(experimentNames);
  experimentNamesRef.current = experimentNames;

  const debouncedValidatorRef = useRef(
    (() => {
      let latestValidationToken = 0;
      const debounced = debounce((value: string, token: number) => {
        const validator = getExperimentNameValidator(() => experimentNamesRef.current);
        validator(null, value, (error?: string) => {
          if (token === latestValidationToken) {
            setNameError(error ?? '');
          }
        });
      }, 400);
      const validate = (value: string) => {
        const token = ++latestValidationToken;
        debounced(value, token);
      };
      validate.cancel = () => {
        // Invalidate any in-flight validator callbacks.
        latestValidationToken++;
        debounced.cancel();
      };
      return validate;
    })(),
  );

  const resetFormState = useCallback(() => {
    debouncedValidatorRef.current.cancel();
    setExperimentName('');
    setArtifactLocation('');
    setNameError('');
    setSubmitError('');
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
    setSubmitError('');
    if (!value.trim()) {
      debouncedValidatorRef.current.cancel();
      setNameError(
        intl.formatMessage({
          defaultMessage: 'Please input a new name for the new experiment.',
          description: 'Error message for name requirement in create experiment for MLflow',
        }),
      );
    } else if (value.length > MAX_EXPERIMENT_NAME_LENGTH) {
      debouncedValidatorRef.current.cancel();
      setNameError(
        intl.formatMessage(
          {
            defaultMessage: 'Must be {maxLength} characters or less',
            description: 'Error message when experiment name exceeds maximum length in create experiment for MLflow',
          },
          { maxLength: MAX_EXPERIMENT_NAME_LENGTH },
        ),
      );
    } else {
      setNameError('');
      debouncedValidatorRef.current(value);
    }
  };

  const handleArtifactLocationChange = (value: string) => {
    setArtifactLocation(value);
    setSubmitError('');
  };

  const isCreateDisabled = !experimentName.trim() || !!nameError || isSubmitting;

  const disabledReason = !experimentName.trim()
    ? intl.formatMessage({
        defaultMessage: 'Please enter an experiment name',
        description: 'Tooltip when Create button is disabled because experiment name is empty',
      })
    : nameError || '';

  const validateExperimentName = async (value: string): Promise<string> => {
    const validator = getExperimentNameValidator(() => experimentNamesRef.current);
    return new Promise((resolve) => {
      validator(undefined, value, (error?: string) => resolve(error ?? ''));
    });
  };

  const handleSubmit = async () => {
    if (isCreateDisabled) return;
    debouncedValidatorRef.current.cancel();
    setSubmitError('');
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
      let message: string;
      if (e instanceof ErrorWrapper) {
        message = e.renderHttpError();
      } else if (e instanceof Error) {
        message = e.message;
      } else {
        message = intl.formatMessage({
          defaultMessage: 'Failed to create experiment',
          description: 'Fallback error message when experiment creation fails in MLflow',
        });
      }
      setSubmitError(message);
    }
  };

  const createButton = (
    <Button
      componentId="mlflow.experiment.create_experiment_modal.submit"
      type="primary"
      onClick={handleSubmit}
      loading={isSubmitting}
      disabled={isCreateDisabled}
    >
      <FormattedMessage defaultMessage="Create" description="Confirm button text for create experiment modal" />
    </Button>
  );

  const wrappedCreateButton = disabledReason ? (
    <Tooltip componentId="mlflow.experiment.create_experiment_modal.submit_tooltip" content={disabledReason}>
      <span css={{ display: 'inline-flex' }}>{createButton}</span>
    </Tooltip>
  ) : (
    createButton
  );

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
      onCancel={handleClose}
      footer={
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          <Button componentId="mlflow.experiment.create_experiment_modal.cancel" onClick={handleClose}>
            <FormattedMessage defaultMessage="Cancel" description="Cancel button text for create experiment modal" />
          </Button>
          {wrappedCreateButton}
        </div>
      }
    >
      <div
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey && !isCreateDisabled) {
            e.preventDefault();
            handleSubmit();
          }
        }}
      >
        {submitError && (
          <>
            <Alert
              componentId="mlflow.experiment.create_experiment_modal.error"
              closable={false}
              message={submitError}
              type="error"
            />
            <Spacer />
          </>
        )}
        <CreateExperimentForm
          experimentName={experimentName}
          artifactLocation={artifactLocation}
          nameError={nameError}
          onNameChange={handleNameChange}
          onArtifactLocationChange={handleArtifactLocationChange}
        />
      </div>
    </Modal>
  );
};
