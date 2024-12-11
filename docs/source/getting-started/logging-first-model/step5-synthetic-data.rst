Create a dataset about apples
=============================

In order to produce some meaningful data (and a model) for us to log to MLflow, we'll need a dataset.
In the interests of sticking with our theme of modeling demand for produce sales, this data will
actually need to be about apples.

There's a distinctly miniscule probability of finding an actual dataset on the internet about this,
so we can just roll up our sleeves and make our own.

Defining a dataset generator
----------------------------

For our examples to work, we're going to need something that can actually fit, but not something that
fits too well. We're going to be training multiple iterations in order to show the effect of modifying
our model's hyperparameters, so there needs to be some amount of unexplained variance in the feature set.
However, we need some degree of correlation between our target variable (``demand``, in the case of our
apples sales data that we want to predict) and the feature set.

We can introduce this correlation by crafting a relationship between our features and our target.
The random elements of some of the factors will handle the unexplained variance portion.

.. code-section::

    .. code-block:: python

        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta


        def generate_apple_sales_data_with_promo_adjustment(
            base_demand: int = 1000, n_rows: int = 5000
        ):
            """
            Generates a synthetic dataset for predicting apple sales demand with seasonality
            and inflation.

            This function creates a pandas DataFrame with features relevant to apple sales.
            The features include date, average_temperature, rainfall, weekend flag, holiday flag,
            promotional flag, price_per_kg, and the previous day's demand. The target variable,
            'demand', is generated based on a combination of these features with some added noise.

            Args:
                base_demand (int, optional): Base demand for apples. Defaults to 1000.
                n_rows (int, optional): Number of rows (days) of data to generate. Defaults to 5000.

            Returns:
                pd.DataFrame: DataFrame with features and target variable for apple sales prediction.

            Example:
                >>> df = generate_apple_sales_data_with_seasonality(base_demand=1200, n_rows=6000)
                >>> df.head()
            """

            # Set seed for reproducibility
            np.random.seed(9999)

            # Create date range
            dates = [datetime.now() - timedelta(days=i) for i in range(n_rows)]
            dates.reverse()

            # Generate features
            df = pd.DataFrame(
                {
                    "date": dates,
                    "average_temperature": np.random.uniform(10, 35, n_rows),
                    "rainfall": np.random.exponential(5, n_rows),
                    "weekend": [(date.weekday() >= 5) * 1 for date in dates],
                    "holiday": np.random.choice([0, 1], n_rows, p=[0.97, 0.03]),
                    "price_per_kg": np.random.uniform(0.5, 3, n_rows),
                    "month": [date.month for date in dates],
                }
            )

            # Introduce inflation over time (years)
            df["inflation_multiplier"] = (
                1 + (df["date"].dt.year - df["date"].dt.year.min()) * 0.03
            )

            # Incorporate seasonality due to apple harvests
            df["harvest_effect"] = np.sin(2 * np.pi * (df["month"] - 3) / 12) + np.sin(
                2 * np.pi * (df["month"] - 9) / 12
            )

            # Modify the price_per_kg based on harvest effect
            df["price_per_kg"] = df["price_per_kg"] - df["harvest_effect"] * 0.5

            # Adjust promo periods to coincide with periods lagging peak harvest by 1 month
            peak_months = [4, 10]  # months following the peak availability
            df["promo"] = np.where(
                df["month"].isin(peak_months),
                1,
                np.random.choice([0, 1], n_rows, p=[0.85, 0.15]),
            )

            # Generate target variable based on features
            base_price_effect = -df["price_per_kg"] * 50
            seasonality_effect = df["harvest_effect"] * 50
            promo_effect = df["promo"] * 200

            df["demand"] = (
                base_demand
                + base_price_effect
                + seasonality_effect
                + promo_effect
                + df["weekend"] * 300
                + np.random.normal(0, 50, n_rows)
            ) * df[
                "inflation_multiplier"
            ]  # adding random noise

            # Add previous day's demand
            df["previous_days_demand"] = df["demand"].shift(1)
            df["previous_days_demand"].fillna(
                method="bfill", inplace=True
            )  # fill the first row

            # Drop temporary columns
            df.drop(columns=["inflation_multiplier", "harvest_effect", "month"], inplace=True)

            return df

Generate the data using the method we just prepared and save its result.

.. code-section::

    .. code-block:: python
        :name: client

        data = generate_apple_sales_data_with_promo_adjustment(base_demand=1_000, n_rows=1_000)

        data[-20:]

In the next section, we'll both use this generator for its output (the data set), and as an example
for how to leverage MLflow Tracking as part of a prototyping phase for a project.
