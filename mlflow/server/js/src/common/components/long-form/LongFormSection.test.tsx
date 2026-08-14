import { describe, it, expect, beforeEach } from '@jest/globals';
import userEventGlobal from '@testing-library/user-event';
import React, { useState } from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { LongFormSection } from './LongFormSection';

// ``userEvent.setup()`` v14 holds stateful pointer/keyboard state; if a test
// throws mid-gesture, leftover state can leak into the next test. Re-create
// per test so each case starts clean.
let userEvent: ReturnType<typeof userEventGlobal.setup>;
beforeEach(() => {
  userEvent = userEventGlobal.setup();
});

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

  it('wires up the disclosure-widget ARIA relationship + ``inert`` when collapsed', () => {
    // ARIA: ``aria-controls`` on the toggle button must point at the content
    // region's ``id`` so screen readers announce what the toggle expands.
    // ``inert`` on the collapsed wrapper is a belt-and-suspenders guarantee:
    // ``hidden`` alone can be overridden by CSS resets, but ``inert``
    // removes the subtree from tab order regardless.
    renderWithDesignSystem(
      <LongFormSection title="Permissions" collapsible defaultCollapsed>
        <button type="button">deep button</button>
      </LongFormSection>,
    );
    const toggle = screen.getByRole('button', { name: /Permissions/ });
    const controlsId = toggle.getAttribute('aria-controls');
    expect(controlsId).toBeTruthy();
    // ``getElementById`` (not ``querySelector('#…')``) because React 18's
    // ``useId`` produces colon-bracketed ids (`:r0:`) that aren't valid in
    // CSS selectors without escaping.
    const region = document.getElementById(controlsId!);
    expect(region).not.toBeNull();
    expect(region).toHaveAttribute('hidden');
    expect(region).toHaveAttribute('inert');
  });

  it('drops ``inert`` when expanded so inner controls are focusable again', async () => {
    renderWithDesignSystem(
      <LongFormSection title="Permissions" collapsible defaultCollapsed>
        <button type="button">deep button</button>
      </LongFormSection>,
    );
    const toggle = screen.getByRole('button', { name: /Permissions/ });
    await userEvent.click(toggle);
    const controlsId = toggle.getAttribute('aria-controls')!;
    const region = document.getElementById(controlsId)!;
    expect(region).not.toHaveAttribute('hidden');
    expect(region).not.toHaveAttribute('inert');
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

  it('preserves child state across a collapse / expand round-trip (starts expanded)', async () => {
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

  it('preserves child state across an expand / collapse round-trip (starts collapsed)', async () => {
    // Same guarantee from the other direction: a section that opens with
    // ``defaultCollapsed`` mounts its children hidden + inert, and any
    // state added after the first expand must survive a re-collapse.
    const Inner = () => {
      const [text, setText] = useState('');
      return <input aria-label="draft" value={text} onChange={(e) => setText(e.target.value)} />;
    };
    renderWithDesignSystem(
      <LongFormSection title="Permissions" collapsible defaultCollapsed>
        <Inner />
      </LongFormSection>,
    );
    // Confirm we really started in the hidden state. ``hidden: true`` opts
    // into matching hidden elements — by default ``getByRole`` filters them
    // out, which would mask the very state we want to assert on.
    expect(screen.getByRole('textbox', { name: 'draft', hidden: true })).not.toBeVisible();
    const toggle = screen.getByRole('button', { name: /Permissions/ });
    await userEvent.click(toggle);
    await userEvent.type(screen.getByRole('textbox', { name: 'draft' }), 'opened-then-typed');
    await userEvent.click(toggle);
    await userEvent.click(toggle);
    expect(screen.getByRole('textbox', { name: 'draft' })).toHaveValue('opened-then-typed');
  });
});
