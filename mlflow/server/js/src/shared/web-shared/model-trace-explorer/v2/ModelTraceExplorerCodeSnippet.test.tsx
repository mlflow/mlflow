import { describe, expect, it, jest } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { render } from '@databricks/testing-library';
import { ProvidersWrapper } from '../../test-utils/testUtilProviderWrappers';
import { CodeSnippetRenderMode } from './ModelTrace.types';
import { ModelTraceExplorerCodeSnippet } from './ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCodeSnippetBody } from './ModelTraceExplorerCodeSnippetBody';

const DATA = JSON.stringify('nested value');

describe('ModelTraceExplorerCodeSnippetBody', () => {
  it('shows an expansion control when rendered JSON exceeds the preview height', async () => {
    jest.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockReturnValue(200);

    render(<ModelTraceExplorerCodeSnippetBody data={DATA} />, { wrapper: ProvidersWrapper });

    expect(await screen.findByRole('button', { name: 'See more' })).toBeInTheDocument();
  });
});

describe('ModelTraceExplorerCodeSnippet', () => {
  it('keeps expansion controls available after switching from YAML to overflowing JSON', async () => {
    jest.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockReturnValue(200);
    render(
      <ModelTraceExplorerCodeSnippet title="attribute" data={DATA} initialRenderMode={CodeSnippetRenderMode.YAML} />,
      { wrapper: ProvidersWrapper },
    );

    await userEvent.click(screen.getByText('YAML'));
    await userEvent.click(await screen.findByRole('menuitemradio', { name: 'JSON' }));
    await userEvent.click(await screen.findByRole('button', { name: 'See more' }));

    expect(screen.getByRole('button', { name: 'See less' })).toBeInTheDocument();
  });
});
