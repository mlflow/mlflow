import { Form, Input, Button, Header } from '@databricks/design-system';
import { useNavigate } from 'react-router-dom-v5-compat';
import { MlflowService } from '../sdk/MlflowService';

export const Signin = () => {
  const navigate = useNavigate();

  const onFinish = (values: any) => {
    const { email, password } = values;
    MlflowService.signin({ email, password }).then((response) => {
      const { access_token } = response;
      localStorage.setItem('access_token', access_token);
      navigate('/', { replace: true });
    });
  };

  return (
    <div css={{ marginLeft: 20 }}>
      <Header title='Sign-in' />
      <Form onFinish={onFinish}>
        <Form.Item
          label='Email'
          name='email'
          rules={[{ required: true, message: 'Please input your email!' }]}
        >
          <Input />
        </Form.Item>

        <Form.Item
          label='Password'
          name='password'
          rules={[{ required: true, message: 'Please input your password!' }]}
        >
          <Input.Password />
        </Form.Item>

        <Form.Item>
          <Button htmlType='submit'>Submit</Button>
        </Form.Item>
      </Form>
    </div>
  );
};
