import { Button, Modal, Typography } from '@databricks/design-system';
import { debounce } from 'lodash';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';

import { useDispatch, useSelector } from 'react-redux';
import { getUUID } from '../../common/utils/ActionUtils';
import { useNavigate } from '../../common/utils/RoutingUtils';
import Utils from '../../common/utils/Utils';
import { getModelNameFilter } from '../utils/SearchUtils';
import type { ReduxState, ThunkDispatch } from '../../redux-types';
import { createModelVersionApi, createRegisteredModelApi, searchRegisteredModelsApi } from '../actions';
import { ModelRegistryRoutes } from '../routes';
import {
  CREATE_NEW_MODEL_OPTION_VALUE,
  MODEL_NAME_FIELD,
  RegisterModelForm,
  SELECTED_MODEL_FIELD,
} from './RegisterModelForm';
import type { ModelVersionInfoEntity } from '../../experiment-tracking/types';

const MAX_SEARCH_REGISTERED_MODELS = 5;

type PromoteModelButtonImplProps = {
  modelVersion: ModelVersionInfoEntity;
};

export const PromoteModelButton = (props: PromoteModelButtonImplProps) => {
  const intl = useIntl();
  const navigate = useNavigate();

  const createRegisteredModelRequestId = useRef(getUUID());
  const createModelVersionRequestId = useRef(getUUID());

  const { modelVersion } = props;
  const [visible, setVisible] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const dispatch = useDispatch<ThunkDispatch>();

  const modelByName = useSelector((state: ReduxState) => state.entities.modelByName);

  const form = useRef<any>();
  const showRegisterModal = () => {
    setVisible(true);
  };

  const hideRegisterModal = () => {
    setVisible(false);
  };

  const resetAndClearModalForm = () => {
    setVisible(false);
    setConfirmLoading(false);
  };

  const handleRegistrationFailure = (e: any) => {
    setConfirmLoading(false);
    Utils.logErrorAndNotifyUser(e);
  };

  const handleSearchRegisteredModels = useCallback(
    (input: any) => {
      dispatch(searchRegisteredModelsApi(getModelNameFilter(input), MAX_SEARCH_REGISTERED_MODELS));
    },
    [dispatch],
  );

  const debouncedHandleSearchRegisteredModels = useMemo(
    () => debounce(handleSearchRegisteredModels, 300),
    [handleSearchRegisteredModels],
  );

  const handleCopyModel = () => {
    form.current.validateFields().then((values: any) => {
      setConfirmLoading(true);
      const selectedModelName = values[SELECTED_MODEL_FIELD];
      const copySource = 'models:/' + modelVersion.name + '/' + modelVersion.version;
      if (selectedModelName === CREATE_NEW_MODEL_OPTION_VALUE) {
        const newModelName = values[MODEL_NAME_FIELD];
        dispatch(createRegisteredModelApi(newModelName, createRegisteredModelRequestId.current))
          .then(() =>
            dispatch(
              createModelVersionApi(
                newModelName,
                copySource,
                modelVersion.run_id,
                modelVersion.tags,
                createModelVersionRequestId.current,
              ),
            ),
          )
          .then((mvResult: any) => {
            resetAndClearModalForm();
            const { version } = mvResult.value['model_version'];
            navigate(ModelRegistryRoutes.getModelVersionPageRoute(newModelName, version));
          })
          .catch(handleRegistrationFailure);
      } else {
        dispatch(
          createModelVersionApi(
            selectedModelName,
            copySource,
            modelVersion.run_id,
            modelVersion.tags,
            createModelVersionRequestId.current,
          ),
        )
          .then((mvResult: any) => {
            resetAndClearModalForm();
            const { version } = mvResult.value['model_version'];
            navigate(ModelRegistryRoutes.getModelVersionPageRoute(selectedModelName, version));
          })
          .catch(handleRegistrationFailure);
      }
    });
  };

  useEffect(() => {
    dispatch(searchRegisteredModelsApi());
  }, [dispatch]);

  useEffect(() => {
    if (visible) {
      dispatch(searchRegisteredModelsApi());
    }
  }, [dispatch, visible]);

  const renderRegisterModelForm = () => {
    return (
      <>
        <Typography.Paragraph css={{ marginTop: '-12px' }}>
          <FormattedMessage
            defaultMessage="Copy your MLflow models to another registered model for
            simple model promotion across environments. For more mature production-grade setups, we
            recommend setting up automated model training workflows to produce models in controlled
            environments. <link>Learn more</link>"
            description="Model registry > OSS Promote model modal > description paragraph body"
            values={{
              link: (chunks) => (
                <Typography.Link
                  componentId="codegen_mlflow_app_src_model-registry_components_promotemodelbutton.tsx_140"
                  href={
                    'https://mlflow.org/docs/latest/model-registry.html' +
                    '#promoting-an-mlflow-model-across-environments'
                  }
                  openInNewTab
                >
                  {chunks}
                </Typography.Link>
              ),
            }}
          />
        </Typography.Paragraph>
        <RegisterModelForm
          modelByName={modelByName}
          innerRef={form}
          onSearchRegisteredModels={debouncedHandleSearchRegisteredModels}
          isCopy
        />
      </>
    );
  };

  return (
    <div className="promote-model-btn-wrapper">
      <Button
        componentId="codegen_mlflow_app_src_model-registry_components_promotemodelbutton.tsx_165"
        className="promote-model-btn"
        type="primary"
        onClick={showRegisterModal}
      >
        <FormattedMessage
          defaultMessage="Promote model"
          description="Button text to pomote the model to a different registered model"
        />
      </Button>
      <Modal
        title={
          <FormattedMessage
            defaultMessage="Promote {sourceModelName} version {sourceModelVersion}"
            description="Modal title to pomote the model to a different registered model"
            values={{ sourceModelName: modelVersion.name, sourceModelVersion: modelVersion.version }}
          />
        }
        // @ts-expect-error TS(2322): Type '{ children: Element; title: any; width: numb... Remove this comment to see the full error message
        width={640}
        visible={visible}
        onOk={handleCopyModel}
        okText={intl.formatMessage({
          defaultMessage: 'Promote',
          description: 'Confirmation text to promote the model',
        })}
        cancelText={intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Cancel text to cancel the flow to copy the model',
        })}
        confirmLoading={confirmLoading}
        onCancel={hideRegisterModal}
        centered
      >
        {renderRegisterModelForm()}
      </Modal>
    </div>
  );
};
