import { describe, expect, it, jest } from '@jest/globals';
import { renderWithDesignSystem, act } from '../../../../common/utils/TestUtils.react18';
import { ApiKeyConfigurator } from './ApiKeyConfigurator';
import type { ApiKeyConfiguration } from '../types';
import type { SecretInfo } from '../../../types';

const emptyNewSecret = { name: '', authMode: '', secretFields: {}, configFields: {} };

const baseValue: ApiKeyConfiguration = {
  mode: 'new',
  existingSecretId: '',
  newSecret: emptyNewSecret,
};

const existingSecret: SecretInfo = {
  secret_id: 'secret-1',
  secret_name: 'my-openai-key',
  masked_values: {},
  provider: 'openai',
  created_at: 1704067200,
  last_updated_at: 1704067200,
};

const defaultProps = {
  provider: 'openai',
  isLoadingSecrets: false,
  authModes: [],
  defaultAuthMode: undefined,
  isLoadingProviderConfig: false,
};

describe('ApiKeyConfigurator auto-switch to existing mode', () => {
  it('switches to existing mode when secrets are available', () => {
    const onChange = jest.fn();
    renderWithDesignSystem(
      <ApiKeyConfigurator
        {...defaultProps}
        value={baseValue}
        onChange={onChange}
        existingSecrets={[existingSecret]}
      />,
    );

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ mode: 'existing' }));
  });

  it('stays on new mode when no existing secrets', () => {
    const onChange = jest.fn();
    renderWithDesignSystem(
      <ApiKeyConfigurator {...defaultProps} value={baseValue} onChange={onChange} existingSecrets={[]} />,
    );

    expect(onChange).not.toHaveBeenCalledWith(expect.objectContaining({ mode: 'existing' }));
  });

  it('stays on new mode while secrets are still loading', () => {
    const onChange = jest.fn();
    renderWithDesignSystem(
      <ApiKeyConfigurator
        {...defaultProps}
        value={baseValue}
        onChange={onChange}
        existingSecrets={[]}
        isLoadingSecrets
      />,
    );

    expect(onChange).not.toHaveBeenCalledWith(expect.objectContaining({ mode: 'existing' }));
  });

  it('does not switch if user has started filling in a new secret name', () => {
    const onChange = jest.fn();
    const valueWithName: ApiKeyConfiguration = {
      ...baseValue,
      newSecret: { ...emptyNewSecret, name: 'my-key' },
    };
    renderWithDesignSystem(
      <ApiKeyConfigurator
        {...defaultProps}
        value={valueWithName}
        onChange={onChange}
        existingSecrets={[existingSecret]}
      />,
    );

    expect(onChange).not.toHaveBeenCalledWith(expect.objectContaining({ mode: 'existing' }));
  });
});
