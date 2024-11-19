import React from 'react';
import { DirectTransitionForm } from './DirectTransitionForm';
import { ACTIVE_STAGES, Stages } from '../constants';
import { Checkbox } from '@databricks/design-system';
import _ from 'lodash';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event-14';

const minimalProps = {
  innerRef: React.createRef(),
};

const checkboxDataTestId = 'direct-transition-form-check-box';
const modelVersionUpdateFormTestId = 'model-version-update-form';

const [activeStages, nonActiveStages] = _.partition(Stages, (s) => ACTIVE_STAGES.includes(s));

describe('DirectTransitionForm', () => {
  test('should render with minimal props without exploding', () => {
    renderWithIntl(<DirectTransitionForm {...minimalProps} />);
    expect(screen.getByTestId(modelVersionUpdateFormTestId)).toBeInTheDocument();
  });

  test.each(activeStages)('should render checkbox for active stage %s', (toStage) => {
    const props = { ...minimalProps, toStage };
    renderWithIntl(<DirectTransitionForm {...props} />);
    expect(screen.getByTestId(modelVersionUpdateFormTestId)).toBeInTheDocument();
    expect(screen.getByTestId(checkboxDataTestId)).toBeInTheDocument();
  });

  test.each(nonActiveStages)('should not render checkbox for non-active stage %s', (toStage) => {
    const props = { ...minimalProps, toStage };
    renderWithIntl(<DirectTransitionForm {...props} />);
    expect(screen.getByTestId(modelVersionUpdateFormTestId)).toBeInTheDocument();
    expect(screen.queryByTestId(checkboxDataTestId)).not.toBeInTheDocument();
  });
});
