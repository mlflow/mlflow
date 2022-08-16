import { Descriptions } from './Descriptions';

export default {
  title: 'Common/Descriptions',
  component: Descriptions,
  argTypes: {},
};

const renderItems = () => (
  <>
    <Descriptions.Item label='The label'>The value</Descriptions.Item>
    <Descriptions.Item label='Another label'>Another value</Descriptions.Item>
    <Descriptions.Item label='A label'>A value</Descriptions.Item>
    <Descriptions.Item label='Extra label'>Extra value</Descriptions.Item>
  </>
);

export const SimpleUse = () => <Descriptions>{renderItems()}</Descriptions>;

export const TwoColumns = () => <Descriptions columns={2}>{renderItems()}</Descriptions>;

export const ManyItems = () => (
  <Descriptions>
    {renderItems()}
    {renderItems()}
    {renderItems()}
  </Descriptions>
);
