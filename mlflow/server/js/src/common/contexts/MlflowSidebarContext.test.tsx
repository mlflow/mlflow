import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import React from 'react';

import { MlflowSidebarContext, useMlflowSidebar } from './MlflowSidebarContext';

const Consumer = () => {
  const sidebar = useMlflowSidebar();
  return <div>{sidebar ? `show=${String(sidebar.showSidebar)}` : 'no-context'}</div>;
};

describe('useMlflowSidebar', () => {
  it('returns undefined outside the provider', () => {
    render(<Consumer />);
    expect(screen.getByText('no-context')).toBeInTheDocument();
  });

  it('exposes the sidebar state from the provider', () => {
    render(
      <MlflowSidebarContext.Provider value={{ showSidebar: false, setShowSidebar: () => {} }}>
        <Consumer />
      </MlflowSidebarContext.Provider>,
    );
    expect(screen.getByText('show=false')).toBeInTheDocument();
  });
});
