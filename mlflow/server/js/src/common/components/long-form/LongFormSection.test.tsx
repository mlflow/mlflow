import { describe, it, expect } from '@jest/globals';
import userEventGlobal from '@testing-library/user-event';
import React, { useState } from 'react';
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
    expect(screen.getByText('inner content')).toBeVisible();
  });

  it('hides children when ``collapsible`` + ``defaultCollapsed`` are set', () => {
    renderWithDesignSystem(
      <LongFormSection title="Permissions" collapsible defaultCollapsed>
        <div>inner content</div>
      </LongFormSection>,
    );
    // Stays mounted but not visible so any in-progress draft state survives.
    expect(screen.getByText('inner content')).not.toBeVisible();
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
    expect(screen.getByText('inner content')).toBeVisible();
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    await userEvent.click(toggle);
    expect(screen.getByText('inner content')).not.toBeVisible();
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
  });

  it('preserves child state across a collapse / expand round-trip', async () => {
    // Pins the reason we use ``hidden`` (not conditional render): a draft
    // typed into ``RolePermissionsSection`` / ``RoleUsersSection`` must
    // survive the user collapsing the section and re-opening it.
    const Inner = () => {
      const [text, setText] = useState('');
      return <input aria-label="draft" value={text} onChange={(e) => setText(e.target.value)} />;
    };
    renderWithDesignSystem(
      <LongFormSection title="Permissions" collapsible>
        <Inner />
      </LongFormSection>,
    );
    const input = screen.getByRole('textbox', { name: 'draft' });
    await userEvent.type(input, 'in-progress');
    const toggle = screen.getByRole('button', { name: /Permissions/ });
    await userEvent.click(toggle);
    await userEvent.click(toggle);
    expect(screen.getByRole('textbox', { name: 'draft' })).toHaveValue('in-progress');
  });
});
