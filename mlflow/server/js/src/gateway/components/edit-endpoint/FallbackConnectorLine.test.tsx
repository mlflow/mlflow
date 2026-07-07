import { describe, expect, it } from '@jest/globals';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { ConnectorLine, FallbackConnectorLine } from './FallbackConnectorLine';

describe('ConnectorLine', () => {
  it('renders without a label', () => {
    renderWithDesignSystem(<ConnectorLine />);
    expect(screen.queryByText('Fallback')).not.toBeInTheDocument();
  });
});

describe('FallbackConnectorLine', () => {
  it('renders the Fallback label', () => {
    renderWithDesignSystem(<FallbackConnectorLine />);
    expect(screen.getByText('Fallback')).toBeInTheDocument();
  });
});
