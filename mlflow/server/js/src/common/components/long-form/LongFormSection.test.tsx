import { describe, it, expect } from '@jest/globals';
import userEventGlobal from '@testing-library/user-event';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { LongFormSection } from './LongFormSection';

const userEvent = userEventGlobal.setup();

describe('LongFormSection — collapsible mode', () => {
  it('renders children when ``collapsible`` is not set (back-compat: existing call sites stay open)', () => {
    renderWithDesignSystem(
      <LongFormSection title="Always open">
        <div>inner content</div>
      </LongFormSection>,
    );
    expect(screen.getByText('inner content')).toBeInTheDocument();
  });

  it('hides children when ``collapsible`` + ``defaultCollapsed`` are set', () => {
    renderWithDesignSystem(
      <LongFormSection title="Permissions" collapsible defaultCollapsed>
        <div>inner content</div>
      </LongFormSection>,
    );
    expect(screen.queryByText('inner content')).not.toBeInTheDocument();
    // Title button stays in the DOM so the user can expand the section.
    expect(screen.getByRole('button', { name: /Permissions/ })).toHaveAttribute('aria-expanded', 'false');
  });

  it('toggles visibility on title click when collapsible', async () => {
    renderWithDesignSystem(
      <LongFormSection title="Permissions" collapsible defaultCollapsed>
        <div>inner content</div>
      </LongFormSection>,
    );
    const toggle = screen.getByRole('button', { name: /Permissions/ });
    await userEvent.click(toggle);
    expect(screen.getByText('inner content')).toBeInTheDocument();
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    await userEvent.click(toggle);
    expect(screen.queryByText('inner content')).not.toBeInTheDocument();
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
  });
});
