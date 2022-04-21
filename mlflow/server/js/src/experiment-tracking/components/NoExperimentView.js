import React from 'react';
import { css } from 'emotion';
import { useDesignSystemTheme } from '@databricks/design-system';
import noExperiments from '../../common/static/no-experiments.svg';
import { ExperimentCliDocUrl } from '../../common/constants';

function NoExperimentViewImpl() {
  const { theme } = useDesignSystemTheme();
  return (
    <div className='center'>
      <img
        alt='No experiments found.'
        style={{ height: '200px', marginTop: '80px' }}
        src={noExperiments}
      />
      <h1 style={{ paddingTop: '10px' }}>No Experiments Exist</h1>
      <h2 className={css({ color: theme.colors.textSecondary })}>
        To create an experiment use the <a href={ExperimentCliDocUrl}>mlflow experiments</a> CLI.
      </h2>
    </div>
  );
}

export const NoExperimentView = NoExperimentViewImpl;
