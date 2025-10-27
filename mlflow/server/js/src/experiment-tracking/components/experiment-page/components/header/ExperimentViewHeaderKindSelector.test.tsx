import { render } from '@testing-library/react';
import { ExperimentKind } from '../../../../constants';
import { ExperimentViewHeaderKindSelector } from './ExperimentViewHeaderKindSelector';
import { IntlProvider } from 'react-intl';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider, DesignSystemThemeProvider } from '@databricks/design-system';

describe('ExperimentViewHeaderKindSelector', () => {
  const renderTestComponent = (props: Partial<React.ComponentProps<typeof ExperimentViewHeaderKindSelector>> = {}) => {
    return render(
      <ExperimentViewHeaderKindSelector
        value={ExperimentKind.NO_INFERRED_TYPE}
        onChange={jest.fn()}
        isUpdating={false}
        readOnly={false}
        {...props}
      />,
      {
        wrapper: ({ children }) => (
          <IntlProvider locale="en">
            <DesignSystemProvider>{children}</DesignSystemProvider>
          </IntlProvider>
        ),
      },
    );
  };

  test('it should render the component in edit mode', async () => {
    const onChange = jest.fn();
    const { getByText, findByText } = renderTestComponent({
      value: ExperimentKind.CUSTOM_MODEL_DEVELOPMENT,
      onChange,
    });
    expect(getByText('Machine learning')).toBeInTheDocument();

    await userEvent.click(getByText('Machine learning'));

    await userEvent.click(await findByText('GenAI apps & agents'));

    expect(onChange).toHaveBeenCalledWith(ExperimentKind.GENAI_DEVELOPMENT);
  });

  test('it should render the component in read only mode', async () => {
    const { getByText, getByRole } = renderTestComponent({
      value: ExperimentKind.CUSTOM_MODEL_DEVELOPMENT,
      readOnly: true,
    });
    expect(getByText('Machine learning')).toBeInTheDocument();

    // No dropdown menu for the tag element
    expect(getByRole('status')).not.toHaveAttribute('aria-haspopup');
  });
});
