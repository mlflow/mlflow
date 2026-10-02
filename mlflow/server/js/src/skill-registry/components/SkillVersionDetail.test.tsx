import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
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
    expect(screen.getByText('active')).toBeInTheDocument();
    expect(screen.getByText('Git')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'https://github.com/acme/skills' })).toHaveAttribute(
      'href',
      'https://github.com/acme/skills',
    );
    expect(screen.getByText('Path: code-review')).toBeInTheDocument();
    expect(screen.getByText('Ref: main')).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: 'https://github.com/acme/skills/tree/main/code-review' }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Opens a third-party site/)).toBeInTheDocument();
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
