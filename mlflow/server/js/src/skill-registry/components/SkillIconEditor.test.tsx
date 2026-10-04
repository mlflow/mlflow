import { describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import { SkillIconEditor } from './SkillIconEditor';

describe('SkillIconEditor', () => {
  it('gives each existing icon URL input its own accessible name', () => {
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <SkillIconEditor
            icons={[{ src: 'https://example.com/light.svg', theme: 'light' }, { src: 'https://example.com/dark.svg' }]}
            onChange={jest.fn()}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );

    expect(screen.getByRole('textbox', { name: 'Icon URL 1' })).toHaveValue('https://example.com/light.svg');
    expect(screen.getByRole('textbox', { name: 'Icon URL 2' })).toHaveValue('https://example.com/dark.svg');
  });
});
