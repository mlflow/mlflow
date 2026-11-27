import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import { AddToolsModal } from './AddToolsModal';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('AddToolsModal', () => {
  it('pre-fills the code snippet with the server name and version', () => {
    render(
      <Wrapper>
        <AddToolsModal visible serverName="io.github.test/my-server" version="2.0.0" onClose={jest.fn()} />
      </Wrapper>,
    );
    expect(screen.getByText(/io.github.test\/my-server/)).toBeInTheDocument();
    expect(screen.getByText(/2.0.0/)).toBeInTheDocument();
  });
});
