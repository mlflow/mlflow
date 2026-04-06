import { describe, expect, it } from '@jest/globals';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { FallbackConnectorLine } from './FallbackConnectorLine';

describe('FallbackConnectorLine', () => {
  it('renders the Fallback label by default', () => {
    renderWithDesignSystem(<FallbackConnectorLine />);
    expect(screen.getByText('Fallback')).toBeInTheDocument();
  });

  it('renders the Fallback label when showLabel is true', () => {
    renderWithDesignSystem(<FallbackConnectorLine showLabel />);
    expect(screen.getByText('Fallback')).toBeInTheDocument();
  });

  it('hides the Fallback label when showLabel is false', () => {
    renderWithDesignSystem(<FallbackConnectorLine showLabel={false} />);
    expect(screen.queryByText('Fallback')).not.toBeInTheDocument();
  });
});
