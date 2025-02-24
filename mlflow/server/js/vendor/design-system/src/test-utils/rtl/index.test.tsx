import { render, screen, within } from '@testing-library/react';

import { expect } from '@databricks/config-jest';

import { openDropdownMenu } from '.';
import { Button, DesignSystemProvider, DropdownMenu } from '../../design-system';

describe('openDropdownMenu', () => {
  it('opens dropdown menu', async () => {
    render(
      <DesignSystemProvider>
        <DropdownMenu.Root>
          <DropdownMenu.Trigger asChild>
            <Button componentId="codegen_design-system_src_test-utils_rtl_index.test.tsx_14">Open menu</Button>
          </DropdownMenu.Trigger>
          <DropdownMenu.Content>
            <DropdownMenu.Item componentId="codegen_design-system_src_test-utils_rtl_index.test.tsx_17">
              Option 1
            </DropdownMenu.Item>
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      </DesignSystemProvider>,
    );
    await openDropdownMenu(screen.getByText('Open menu'));
    const dropdownMenu = await screen.findByRole('menu');
    expect(within(dropdownMenu).getByText('Option 1')).toBeInTheDocument();
  });
});
