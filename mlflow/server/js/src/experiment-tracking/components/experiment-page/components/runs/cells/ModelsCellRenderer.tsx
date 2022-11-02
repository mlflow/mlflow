import React from 'react';
import { TrimmedText } from '../../../../../../common/components/TrimmedText';
import loggedModelSvg from '../../../../../../common/static/logged-model.svg';
import registeredModelSvg from '../../../../../../common/static/registered-model.svg';
import Utils from '../../../../../../common/utils/Utils';
import { getModelVersionPageRoute } from '../../../../../../model-registry/routes';
import Routes from '../../../../../routes';
import { RunRowModelsInfo } from '../../../utils/experimentPage.row-types';

const EMPTY_CELL_PLACEHOLDER = '-';

export interface ModelsCellRendererProps {
  value: RunRowModelsInfo;
}

export const ModelsCellRenderer = React.memo((props: ModelsCellRendererProps) => {
  const { registeredModels, loggedModels, experimentId, runUuid } = props.value;
  const models = Utils.mergeLoggedAndRegisteredModels(loggedModels, registeredModels);

  if (models && models.length) {
    const modelToRender = models[0];
    let modelDiv;
    if (modelToRender.registeredModelName) {
      const { registeredModelName, registeredModelVersion } = modelToRender;
      modelDiv = (
        <>
          <img
            data-test-id='registered-model-icon'
            alt=''
            title='Registered Model'
            src={registeredModelSvg}
          />
          <a
            href={Utils.getIframeCorrectedRoute(
              getModelVersionPageRoute(registeredModelName, registeredModelVersion),
            )}
            className='registered-model-link'
            target='_blank'
            rel='noreferrer'
          >
            <TrimmedText text={registeredModelName} maxSize={10} className={'model-name'} />
            {`/${registeredModelVersion}`}
          </a>
        </>
      );
    } else if (modelToRender.flavors) {
      const loggedModelFlavorText = modelToRender.flavors ? modelToRender.flavors[0] : 'Model';
      const loggedModelLink = Utils.getIframeCorrectedRoute(
        `${Routes.getRunPageRoute(experimentId, runUuid)}/artifactPath/${
          modelToRender.artifactPath
        }`,
      );
      modelDiv = (
        <>
          <img data-test-id='logged-model-icon' alt='' title='Logged Model' src={loggedModelSvg} />
          {/* Reported during ESLint upgrade */}
          {/* eslint-disable-next-line react/jsx-no-target-blank */}
          <a href={loggedModelLink} target='_blank' className='logged-model-link'>
            {loggedModelFlavorText}
          </a>
        </>
      );
    }

    return (
      <div className='logged-model-cell' css={styles.imageWrapper}>
        {modelDiv}
        {loggedModels.length > 1 ? `, ${loggedModels.length - 1} more` : ''}
      </div>
    );
  }
  return <>{EMPTY_CELL_PLACEHOLDER}</>;
});

const styles = {
  imageWrapper: {
    img: {
      height: '15px',
      position: 'relative' as const,
      marginRight: '4px',
    },
  },
};
