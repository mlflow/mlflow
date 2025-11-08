import userEvent from '@testing-library/user-event';
import { type Location, useLocation } from '../../../common/utils/RoutingUtils';
import {
  setupTestRouter,
  testRoute,
  waitForRoutesToBeRendered,
  TestRouter,
} from '../../../common/utils/RoutingTestUtils';
import { renderWithIntl, act, screen, within } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { ModelEntity } from '../../../experiment-tracking/types';
import { ModelsTableAliasedVersionsCell } from './ModelsTableAliasedVersionsCell';
import { openDropdownMenu } from '@databricks/design-system/test-utils/rtl';
import { DesignSystemProvider } from '@databricks/design-system';

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

  const renderWithRouterWrapper = async (element: React.ReactElement) => {
    let _location: Location | null = null;
    const Component = () => {
      _location = useLocation();

      return <>{element}</>;
    };
    renderWithIntl(
      <DesignSystemProvider>
        <TestRouter routes={[testRoute(<Component />)]} />
      </DesignSystemProvider>,
    );
    return { getLocation: () => _location };
  };

  test('display a single version and navigate to it', async () => {
    const { getLocation } = await renderWithRouterWrapper(<ModelsTableAliasedVersionsCell model={modelWithOneAlias} />);
    expect(screen.getByText(/@ alias-version-1/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('link', { name: /alias-version-1 : Version 1/ }));
    expect(getLocation()?.pathname).toMatch('/models/test-model/versions/1');
  });

  test('display multiple versions and navigate there', async () => {
    const { getLocation } = await renderWithRouterWrapper(
      <ModelsTableAliasedVersionsCell model={modelWithMultipleAliases} />,
    );
    expect(screen.getByText(/@ alias-version-10/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('link', { name: /alias-version-10 : Version 10/ }));
    expect(getLocation()?.pathname).toMatch('/models/test-model/versions/10');

    await act(async () => {
      await openDropdownMenu(screen.getByRole('button', { name: '+3' }));
    });

    await userEvent.click(within(screen.getByRole('menu')).getByText(/Version 2/));

    expect(getLocation()?.pathname).toMatch('/models/test-model/versions/2');
  });
});
