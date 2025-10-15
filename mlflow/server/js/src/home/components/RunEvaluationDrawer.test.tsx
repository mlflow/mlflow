import React, { useEffect } from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { RunEvaluationDrawer } from './RunEvaluationDrawer';
import { HomePageViewStateProvider, useHomePageViewState } from '../HomePageViewStateContext';

const OpenOnMount = () => {
  const { openRunEvaluationDrawer } = useHomePageViewState();
  useEffect(() => {
    openRunEvaluationDrawer();
  }, [openRunEvaluationDrawer]);
  return null;
};

describe('RunEvaluationDrawer', () => {
  it('renders the drawer with default framework selected', () => {
    renderWithDesignSystem(
      <MemoryRouter>
        <HomePageViewStateProvider>
          <OpenOnMount />
          <RunEvaluationDrawer />
        </HomePageViewStateProvider>
      </MemoryRouter>,
    );

    expect(
      screen.getByRole('dialog', {
        name: 'Run evaluation',
      }),
    ).toBeInTheDocument();

    const openAiButton = screen.getByRole('button', { name: 'OpenAI' });
    expect(openAiButton).toHaveAttribute('aria-pressed', 'true');
    expect(
      screen.getByText('Select an example and follow the steps to evaluate your models with MLflow.'),
    ).toBeVisible();
    expect(
      Array.from(document.querySelectorAll('code')).some((node) =>
        node.textContent?.includes('mlflow.genai.evaluate('),
      ),
    ).toBe(true);
  });

  it('updates content when selecting a different framework', async () => {
    renderWithDesignSystem(
      <MemoryRouter>
        <HomePageViewStateProvider>
          <OpenOnMount />
          <RunEvaluationDrawer />
        </HomePageViewStateProvider>
      </MemoryRouter>,
    );

    const scikitButton = screen.getByRole('button', { name: 'Scikit-learn' });
    await userEvent.click(scikitButton);

    expect(scikitButton).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'OpenAI' })).toHaveAttribute('aria-pressed', 'false');
    expect(
      Array.from(document.querySelectorAll('code')).some((node) =>
        node.textContent?.includes('mlflow.models.evaluate('),
      ),
    ).toBe(true);
  });
});
