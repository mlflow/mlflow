import { DesignSystemProvider } from '@databricks/design-system';
import { describe, expect, it } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { ArtifactPreviewTooLargeError } from '../../../common/utils/ArtifactUtils';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';

const renderError = (error: unknown, description?: React.ReactNode) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ArtifactViewErrorState error={error} description={description} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('ArtifactViewErrorState', () => {
  it('explains the runtime preview size limit', () => {
    renderError(new ArtifactPreviewTooLargeError());
    expect(screen.getByText(/Maximum file size for preview: 50 MiB/)).toBeInTheDocument();
  });

  it('shows a reload prompt when server artifact settings change', () => {
    renderError(new ErrorWrapper({ artifacts_presigned_only: true }, 403));
    expect(screen.getByText(/Reload this page to continue/)).toBeInTheDocument();
  });

  it('keeps unexpected errors generic', () => {
    renderError(new Error('private implementation detail'));
    expect(screen.getByText('Loading artifact failed')).toBeInTheDocument();
    expect(screen.queryByText('private implementation detail')).not.toBeInTheDocument();
  });
  it.each([
    [new ArtifactPreviewTooLargeError(), /Maximum file size for preview: 50 MiB/],
    [new ErrorWrapper({ artifacts_presigned_only: true }, 403), /Reload this page to continue/],
    [
      new ErrorWrapper({ external_artifact_service: true, message: 'Use a same-origin artifact service.' }, 403),
      /Use a same-origin artifact service/,
    ],
    [
      new ErrorWrapper(
        { invalid_artifact_root: true, message: 'The artifact root URI cannot be decoded safely.' },
        400,
      ),
      /The artifact root URI cannot be decoded safely/,
    ],
  ])('prioritizes actionable artifact guidance over caller descriptions: %s', (error, guidance) => {
    renderError(error, 'Could not load model information.');
    expect(screen.getByText(guidance)).toBeInTheDocument();
    expect(screen.queryByText('Could not load model information.')).not.toBeInTheDocument();
  });

  it('preserves caller descriptions for ordinary errors', () => {
    renderError(new Error('private implementation detail'), 'Could not load model information.');
    expect(screen.getByText('Could not load model information.')).toBeInTheDocument();
    expect(screen.queryByText('private implementation detail')).not.toBeInTheDocument();
  });
});
