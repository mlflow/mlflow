import { describe, expect, it } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';

import { createMockSkill } from '../test-utils';
import { SkillAction } from '../types';
import { UseSkillButton } from './UseSkillButton';

const renderUseSkillButton = (allowed_actions?: SkillAction[]) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <UseSkillButton skill={createMockSkill({ allowed_actions })} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('UseSkillButton permission gating', () => {
  it('disables the Use button and explains the permission when USE is not allowed', async () => {
    renderUseSkillButton([]);

    const button = screen.getByRole('button', { name: 'Use' });
    expect(button).toBeDisabled();

    await userEvent.hover(button.parentElement!);
    expect(await screen.findByRole('tooltip')).toHaveTextContent('You do not have permission to use this skill.');
  });

  it('shows the Use button when the skill allows USE', () => {
    renderUseSkillButton([SkillAction.USE]);

    expect(screen.getByRole('button', { name: 'Use' })).toBeInTheDocument();
  });

  it('keeps the Use button when allowed_actions is omitted', () => {
    renderUseSkillButton(undefined);

    expect(screen.getByRole('button', { name: 'Use' })).toBeInTheDocument();
  });

  it('pins the selected version and changes the install destination', async () => {
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <UseSkillButton skill={createMockSkill({ organization: 'acme', name: 'code-review' })} version={2} />
        </DesignSystemProvider>
      </IntlProvider>,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Use' }));
    expect(await screen.findByText('Use @acme/code-review')).toBeInTheDocument();
    expect(screen.getByText('Pinned version: v2')).toBeInTheDocument();
    expect(document.body.textContent).toContain('skills:/@acme/code-review/2');
    expect(document.body.textContent).toContain('--destination .claude/skills');

    await userEvent.click(screen.getByRole('radio', { name: 'Python' }));
    expect(document.body.textContent).toContain('version=2');
    await userEvent.click(screen.getByRole('radio', { name: 'CLI' }));

    const trigger = document.querySelector<HTMLElement>(
      '[data-component-id="mlflow.skill_registry.use_modal.install_target"]',
    );
    if (!trigger) throw new Error('Install target select not found');
    await userEvent.click(trigger);
    await userEvent.click(await screen.findByRole('option', { name: 'GitHub Copilot' }));
    expect(document.body.textContent).toContain('--destination .github/skills');
    expect(document.body.textContent).toContain('skills:/@acme/code-review/2');
  });
});
