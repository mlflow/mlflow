import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { HoverCard } from '.';
import { Button } from '../Button';
import { DesignSystemProvider } from '../DesignSystemProvider';

describe('HoverCard', function () {
  function renderComponent() {
    return render(
      <DesignSystemProvider>
        <HoverCard
          trigger={
            <Button
              componentId="codegen_design-system_src_design-system_hovercard_hovercard.test.tsx_14"
              data-testid="test-trigger"
            >
              Hover to see content
            </Button>
          }
          content={<div>HoverCard content</div>}
          align="start"
        />
      </DesignSystemProvider>,
    );
  }

  it('renders HoverCard on hover and hides on mouse leave', async () => {
    renderComponent();

    // Trigger hover event
    await userEvent.hover(screen.getByTestId('test-trigger'));

    // Wait for content to appear
    await waitFor(() => {
      expect(screen.getByText('HoverCard content')).toBeInTheDocument();
    });

    // Trigger unhover event
    await userEvent.unhover(screen.getByTestId('test-trigger'));

    // Wait for content to disappear
    await waitFor(() => {
      expect(screen.queryByText('HoverCard content')).toBeNull();
    });
  });
});
