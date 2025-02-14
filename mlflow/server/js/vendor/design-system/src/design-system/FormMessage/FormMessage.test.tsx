import { render, screen } from '@testing-library/react';

import type { FormMessageProps } from './FormMessage';
import { FormMessage } from './FormMessage';
import { DesignSystemProvider } from '../DesignSystemProvider';

describe('FormMessage', function () {
  function renderComponent({ type = 'warning', message = 'some message', ...rest }: Partial<FormMessageProps> = {}) {
    return render(
      <DesignSystemProvider>
        <FormMessage type={type} message={message} {...rest} />
      </DesignSystemProvider>,
    );
  }

  it('renders message', async () => {
    renderComponent();
    expect(screen.getByText('some message')).toBeInTheDocument();
  });
});
