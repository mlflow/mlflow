import { Form, Button, Header } from '@databricks/design-system';
import { useNavigate } from 'react-router-dom-v5-compat';

export const Signout = () => {
  const navigate = useNavigate();

  const onFinish = () => {
    localStorage.removeItem('access_token');
    navigate('/', { replace: true });
  };

  return (
    <div css={{ marginLeft: 20 }}>
      <Header title='Sign out' />
      <Form onFinish={onFinish}>
        <Form.Item>
          <Button htmlType='submit'>Sign out</Button>
        </Form.Item>
      </Form>
    </div>
  );
};
