import { string, object } from 'yup';

export const validationSchema = object().shape({
  newRunName: string().required(),
});
