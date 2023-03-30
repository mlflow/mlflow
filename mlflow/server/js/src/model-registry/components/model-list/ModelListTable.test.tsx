import { BrowserRouter } from 'react-router-dom';
import {
  getTableRowByCellText,
  getTableRows,
} from '@databricks/design-system/dist/test-utils/enzyme';
import { mountWithIntl } from '../../../common/utils/TestUtils';
import { ModelListTable, ModelListTableProps } from './ModelListTable';

import { Stages } from '../../constants';
import Utils from '../../../common/utils/Utils';

const MODELS = [
  {
    name: 'test_model_1',
    creation_timestamp: 1234,
    email_subscription_status: 'active',
    last_updated_timestamp: 1234,
    permission_level: '',
    source: 'notebook',
    status: 'active',
    user_id: '123',
    latest_versions: [{ version: 1 } as any],
    tags: [],
  },
  {
    name: 'test_model_2',
    creation_timestamp: 100000,
    email_subscription_status: 'active',
    last_updated_timestamp: 100000,
    permission_level: '',
    source: 'notebook',
    status: 'active',
    user_id: '123',
    latest_versions: [
      { version: 2, current_stage: Stages.PRODUCTION },
      { version: 3, current_stage: Stages.STAGING },
      { version: 1 },
    ] as any,
    tags: [],
  },
];

describe('ModelListTable', () => {
  const minimalProps: ModelListTableProps = {
    isLoading: false,
    modelsData: MODELS as any,
    onSortChange: jest.fn(),
    orderByAsc: false,
    orderByKey: 'name',
    pagination: <div data-testid='pagination' />,
    isFiltered: false,
  };

  const createComponentWrapper = (moreProps: Partial<ModelListTableProps> = {}) => {
    return mountWithIntl(
      <BrowserRouter>
        <ModelListTable {...minimalProps} {...moreProps} />
      </BrowserRouter>,
    );
  };

  it('mounts the component and checks if rows are rendered', () => {
    const wrapper = createComponentWrapper({});

    // One header row and two data rows
    expect(wrapper.find('div[role="row"]').length).toBe(3);

    // Our mocked pagination exists
    expect(wrapper.find('[data-testid="pagination"]').exists()).toBeTruthy();
  });

  it('checks if the modification date column is rendered', () => {
    const wrapper = createComponentWrapper({});

    const {
      bodyRows: [firstRow],
    } = getTableRows(wrapper);
    expect(
      firstRow
        .findWhere((column) =>
          column.text().includes(Utils.formatTimestamp(MODELS[0].last_updated_timestamp)),
        )
        .exists(),
    ).toBeTruthy();
  });

  it('checks if the model link is rendered', () => {
    const wrapper = createComponentWrapper({});
    expect(
      getTableRowByCellText(wrapper, 'test_model_1')
        .find('a[href="/models/test_model_1"]')
        .exists(),
    ).toBeTruthy();
    expect(
      getTableRowByCellText(wrapper, 'test_model_2')
        .find('a[href="/models/test_model_2"]')
        .exists(),
    ).toBeTruthy();
  });

  it('checks if the simple model version links are rendered', () => {
    const wrapper = createComponentWrapper({});
    // Model #1 contains only one version
    expect(
      getTableRowByCellText(wrapper, 'test_model_1')
        .find('a[href="/models/test_model_1/versions/1"]')
        .exists(),
    ).toBeTruthy();
  });

  it('checks if the staged model version links are rendered', () => {
    const wrapper = createComponentWrapper({});
    // Model #2 contains versions 2 and 3 in staging in production, but version 1 is not shown
    const row = getTableRowByCellText(wrapper, 'test_model_2');
    expect(row.find('a[href="/models/test_model_2/versions/1"]').exists()).toBeFalsy();
    expect(row.find('a[href="/models/test_model_2/versions/2"]').exists()).toBeTruthy();
    expect(row.find('a[href="/models/test_model_2/versions/3"]').exists()).toBeTruthy();
  });
});
