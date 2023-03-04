import { Form, Input, Button, Header } from '@databricks/design-system';
import { useNavigate } from 'react-router-dom-v5-compat';
import { MlflowService } from '../sdk/MlflowService';

export const Signup = () => {
  const navigate = useNavigate();

  const onFinish = (values: any) => {
    const { email, password } = values;
    MlflowService.signup({ email, password }).then(() => {
      navigate('/signin', { replace: true });
    });
  };

  return (
    <div css={{ marginLeft: 20 }}>
      <Header title='Sign-up' />
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
