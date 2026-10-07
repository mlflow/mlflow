import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { SkillVersionList } from './SkillVersionList';
import { createMockSkillVersion } from '../test-utils';
import { SkillStatus } from '../types';

const renderVersionList = (props: Partial<React.ComponentProps<typeof SkillVersionList>> = {}) => {
  const defaultProps = {
    versions: [],
    onSelectVersion: jest.fn(),
    ...props,
  };
  return render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <SkillVersionList {...defaultProps} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
};

describe('SkillVersionList', () => {
  it('renders empty state when no versions exist', () => {
    renderVersionList({ versions: [] });
    expect(screen.getByText('No versions')).toBeInTheDocument();
  });

  it('renders version labels, status tags, and highlights the selected row', () => {
    renderVersionList({
      versions: [
        createMockSkillVersion({ version: 2, status: SkillStatus.ACTIVE }),
        createMockSkillVersion({ version: 1, status: SkillStatus.DRAFT }),
      ],
      selectedVersion: 1,
    });
    expect(screen.getByText('Version 2')).toBeInTheDocument();
    expect(screen.getByText('Version 1')).toBeInTheDocument();
    expect(screen.getByText('Active')).toBeInTheDocument();
    expect(screen.getByText('Draft')).toBeInTheDocument();
    expect(screen.getByRole('row', { selected: true })).toHaveTextContent('Version 1');
  });

  it('warns when more versions exist than the list shows', () => {
    renderVersionList({ versions: [createMockSkillVersion()], hasMoreVersions: true });
    expect(screen.getByText('Only the most recent 100 versions are shown.')).toBeInTheDocument();
  });

  it('selects a version with click and keyboard', async () => {
    const onSelectVersion = jest.fn();
    renderVersionList({
      versions: [createMockSkillVersion({ version: 1 })],
      onSelectVersion,
    });

    await userEvent.click(screen.getByText('Version 1'));
    expect(onSelectVersion).toHaveBeenCalledWith(1);

    onSelectVersion.mockClear();
    const row = screen.getByText('Version 1').closest('[role="row"]') as HTMLElement | null;
    row?.focus();
    await userEvent.keyboard('{Enter}');
    expect(onSelectVersion).toHaveBeenCalledWith(1);
  });
});
