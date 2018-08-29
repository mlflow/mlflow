import * as yup from 'yup';

export const validationSchema = yup.object().shape({
  firstName: yup.string().required(),
  lastName: yup.string().required(),
  age: yup.number().positive().integer(),
  email: yup.string().email(),
  favoriteColor: yup.string(),
})
