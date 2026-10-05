import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { SkillVersionDetail } from './SkillVersionDetail';
import { createMockSkill, createMockSkillVersion } from '../test-utils';
import { SkillStatus } from '../types';

const renderDetail = (props: Partial<React.ComponentProps<typeof SkillVersionDetail>> = {}) => {
  const skill = props.skill ?? createMockSkill();
  return render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <SkillVersionDetail skill={skill} {...props} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
};

describe('SkillVersionDetail', () => {
  it('renders version metadata, aliases, tags, and a safe source link', () => {
    renderDetail({
      version: createMockSkillVersion({
        version: 2,
        status: SkillStatus.ACTIVE,
        source: 'https://github.com/acme/skills',
        aliases: ['prod'],
        tags: { env: 'prod' },
      }),
    });

    expect(screen.getByText('Viewing version 2')).toBeInTheDocument();
    expect(screen.getByText('Active')).toBeInTheDocument();
    expect(screen.getByText('Git')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'https://github.com/acme/skills' })).toHaveAttribute(
      'href',
      'https://github.com/acme/skills',
    );
    expect(screen.getByText('Path: code-review')).toBeInTheDocument();
    expect(screen.getByText('Ref: main')).toBeInTheDocument();
    // The source row and the Files notice both point at the browse URL, as in the prototype.
    expect(screen.getAllByRole('link', { name: 'https://github.com/acme/skills/tree/main/code-review' })).toHaveLength(
      2,
    );
    expect(screen.getAllByText(/Opens a third-party site/)).toHaveLength(2);
    expect(screen.getByText('Content is read from a remote source.')).toBeInTheDocument();
    expect(screen.getByText('sha256:abc123')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Copy content digest' })).toBeInTheDocument();
    expect(screen.getByText('skills:/@acme/code-review/2')).toBeInTheDocument();
    expect(screen.getByText('skills:/@acme/code-review@prod')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /reference uri/i })).not.toBeInTheDocument();
    expect(screen.queryByText('Last updated:')).not.toBeInTheDocument();
    expect(screen.queryByText('Subpath:')).not.toBeInTheDocument();
    expect(document.body.textContent).toContain('env');
    expect(document.body.textContent).toContain('prod');
  });

  it('allows deleting a deprecated version and explains why an active one cannot be deleted', async () => {
    const { rerender } = renderDetail({
      version: createMockSkillVersion({ status: SkillStatus.DEPRECATED }),
      onDelete: jest.fn(),
    });
    expect(screen.getByRole('button', { name: 'Delete version' })).toBeEnabled();

    rerender(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <SkillVersionDetail
            skill={createMockSkill()}
            version={createMockSkillVersion({ status: SkillStatus.ACTIVE })}
            onDelete={jest.fn()}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );
    const button = screen.getByRole('button', { name: 'Delete version' });
    expect(button).toBeDisabled();
    await userEvent.hover(button.parentElement as HTMLElement);
    expect(await screen.findAllByText(/Unpublish or deprecate this version first/)).not.toHaveLength(0);
  });

  it.each([
    [{ digest: 'a'.repeat(64) }, /Pulling this version fails if the fetched content no longer matches/],
    [{ digest: null, source_type: 'git' as const, ref: 'main' }, /This version points at main; if that is a branch/],
    [{ digest: null, source_type: 'git' as const, ref: null }, /points at the repository's default branch/],
    [{ digest: null, source_type: 'git' as const, ref: '0123abcd' }, /pulls aren't checked against the content/],
    [{ digest: null, source_type: 'mlflow' as const }, /files are stored in MLflow/],
  ])('explains the content digest for %p', async (overrides, expected) => {
    renderDetail({ version: createMockSkillVersion(overrides) });
    await userEvent.hover(screen.getByLabelText('About the content digest'));
    expect(await screen.findAllByText(expected)).not.toHaveLength(0);
  });

  it('hides the creator row when no user was recorded', () => {
    renderDetail({ version: createMockSkillVersion({ created_by: null }) });
    expect(screen.queryByText('Created by:')).not.toBeInTheDocument();
  });

  it('does not link unsafe sources', () => {
    renderDetail({
      version: createMockSkillVersion({ source: 'mlflow-artifacts:/skills/@acme/code-review/2' }),
    });
    expect(screen.getByText('mlflow-artifacts:/skills/@acme/code-review/2')).toBeInTheDocument();
    expect(
      screen.queryByRole('link', { name: 'mlflow-artifacts:/skills/@acme/code-review/2' }),
    ).not.toBeInTheDocument();
  });

  it('renders empty, missing, and error states', () => {
    const { rerender } = renderDetail({});
    expect(screen.getByText('Select a version to view details.')).toBeInTheDocument();

    rerender(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <SkillVersionDetail skill={createMockSkill()} isMissing />
        </DesignSystemProvider>
      </IntlProvider>,
    );
    expect(screen.getByText('This version is no longer available.')).toBeInTheDocument();

    rerender(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <SkillVersionDetail skill={createMockSkill()} error={new Error('Version lookup failed')} />
        </DesignSystemProvider>
      </IntlProvider>,
    );
    expect(screen.getByText('Version lookup failed')).toBeInTheDocument();
  });
});
