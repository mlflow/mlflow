import { jest, describe, test, expect } from '@jest/globals';
import { renderWithIntl, screen, within, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { ModelsNextUIPromoModal } from './ModelsNextUIPromoModal';

describe('ModelsNextUIPromoModal', () => {
  test('renders the promo content when visible', () => {
    renderWithIntl(<ModelsNextUIPromoModal visible onClose={jest.fn()} onTryItNow={jest.fn()} />);
    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByText(/Flexible, governed deployments/)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: /Try it now/ })).toBeInTheDocument();
  });

  // Regression for #24257: on open, focus must land on a real control (the primary button)
  // rather than resting on the dialog's aria-hidden focus-trap sentinel.
  test('moves focus to the primary action when opened', async () => {
    renderWithIntl(<ModelsNextUIPromoModal visible onClose={jest.fn()} onTryItNow={jest.fn()} />);
    const tryItNow = screen.getByRole('button', { name: /Try it now/ });
    await waitFor(() => expect(tryItNow).toHaveFocus());
  });
});
