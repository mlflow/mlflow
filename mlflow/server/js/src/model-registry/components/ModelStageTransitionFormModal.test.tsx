import { render, screen } from '@testing-library/react';
import { ModelStageTransitionFormModal, ModelStageTransitionFormModalMode } from './ModelStageTransitionFormModal';
import type { ComponentProps } from 'react';
import { Stages } from '../constants';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

describe('ModelStageTransitionFormModal', () => {
  const renderTestComponent = (props: Partial<ComponentProps<typeof ModelStageTransitionFormModal>>) => {
    render(
      <ModelStageTransitionFormModal
        visible
        transitionDescription={null}
        allowArchivingExistingVersions
        toStage={Stages.STAGING}
        {...props}
      />,
      {
        wrapper: ({ children }) => (
          <DesignSystemProvider>
            <IntlProvider locale="en">{children}</IntlProvider>
          </DesignSystemProvider>
        ),
      },
    );
  };

  it('should handle form submission', async () => {
    const onConfirm = jest.fn();
    renderTestComponent({ onConfirm });

    expect(await screen.findByText('Stage transition')).toBeInTheDocument();

    await userEvent.type(screen.getByLabelText('Comment'), 'test comment');
    await userEvent.click(screen.getByLabelText(/Transition existing/));
    await userEvent.click(screen.getByRole('button', { name: 'OK' }));
    expect(onConfirm).toHaveBeenCalledWith(
      { comment: 'test comment', archiveExistingVersions: true },
      expect.anything(),
    );
  });

  it.each([
    [ModelStageTransitionFormModalMode.Approve, 'Approve pending request'],
    [ModelStageTransitionFormModalMode.Reject, 'Reject pending request'],
    [ModelStageTransitionFormModalMode.Cancel, 'Cancel pending request'],
  ])('should display proper modal title when particular operating mode is selected', async (mode, expectedTitle) => {
    const onConfirm = jest.fn();
    renderTestComponent({ onConfirm, mode });

    expect(await screen.findByText(expectedTitle)).toBeInTheDocument();
  });
});
