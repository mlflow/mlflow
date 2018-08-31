import * as yup from 'yup';

export const validationSchema = yup.object().shape({
  newRunName: yup.string().required(),
});
