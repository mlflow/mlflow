import userEvent from '@testing-library/user-event';
import { type Location, MemoryRouter, useLocation } from '../../../common/utils/RoutingUtils';
import { act, renderWithIntl, screen, within } from '../../../common/utils/TestUtils';
import { ModelEntity } from '../../../experiment-tracking/types';
import { ModelsTableAliasedVersionsCell } from './ModelsTableAliasedVersionsCell';
import { openDropdownMenu } from '@databricks/design-system/test-utils/rtl';
import {
  DesignSystemThemeProvider,
  DesignSystemThemeContext,
  DesignSystemProvider,
} from '@databricks/design-system';
import { first } from 'lodash';

describe('ModelListTableAliasedVersionsCell', () => {
  const modelWithOneAlias: ModelEntity = {
    name: 'test-model',
    aliases: [{ alias: 'alias-version-1', version: '1' }],
  } as any;

  const modelWithMultipleAliases: ModelEntity = {
    name: 'test-model',
    aliases: [
      { alias: 'alias-version-1-another', version: '1' },
      { alias: 'alias-version-1', version: '1' },
      { alias: 'alias-version-2', version: '2' },
      { alias: 'alias-version-10', version: '10' },
    ],
  } as any;

  const renderWithRouterWrapper = (element: React.ReactElement) => {
    let _location: Location | null = null;
    const Component = () => {
      _location = useLocation();

      return <>{element}</>;
    };
    renderWithIntl(
      <DesignSystemProvider>
        <MemoryRouter>
          <Component />
        </MemoryRouter>
      </DesignSystemProvider>,
    );
    return { getLocation: () => _location };
  };

  test('display a single version and navigate to it', () => {
    const { getLocation } = renderWithRouterWrapper(
      <ModelsTableAliasedVersionsCell model={modelWithOneAlias} />,
    );
    expect(screen.getByText(/@ alias-version-1/)).toBeInTheDocument();
    userEvent.click(screen.getByRole('link', { name: 'Version 1' }));
    expect(getLocation()?.pathname).toEqual('/models/test-model/versions/1');
  });

  test('display multiple versions and navigate there', async () => {
    const { getLocation } = renderWithRouterWrapper(
      <ModelsTableAliasedVersionsCell model={modelWithMultipleAliases} />,
    );
    expect(screen.getByText(/@ alias-version-10/)).toBeInTheDocument();
    userEvent.click(screen.getByRole('link', { name: 'Version 10' }));
    expect(getLocation()?.pathname).toEqual('/models/test-model/versions/10');

    await act(async () => {
      await openDropdownMenu(screen.getByRole('button', { name: '+3' }));
    });

    userEvent.click(within(screen.getByRole('menu')).getByText(/Version 2/));

    expect(getLocation()?.pathname).toEqual('/models/test-model/versions/2');
  });
});
