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
});
