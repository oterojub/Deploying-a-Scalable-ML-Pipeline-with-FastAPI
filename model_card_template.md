# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained to predict whether an individual's
annual income exceeds $50,000 based on demographic and employment information from
the U.S. Census Bureau dataset. The model was built using scikit-learn's
RandomForestClassifier with 100 estimators and a random state of 42.

## Intended Use

This model is intended for educational and research purposes to demonstrate how
demographic and employment features can be used to classify income levels. It is
designed to predict whether a person earns more or less than $50,000 per year.
It should not be used for making real-world financial or employment decisions.

## Training Data

The model was trained on the Census Bureau dataset (census.csv), which contains
32,561 records with 14 features including age, workclass, education, marital status,
occupation, relationship, race, sex, capital gain, capital loss, hours per week, and
native country. The dataset was split 80/20 into training and test sets using
scikit-learn's train_test_split with a random state of 42. Categorical features were
encoded using a OneHotEncoder, and the label was binarized using a LabelBinarizer.

## Evaluation Data

The model was evaluated on the remaining 20% of the Census Bureau dataset, which
was held out during training. The test set contains approximately 6,513 records and
uses the same preprocessing pipeline as the training data, with the encoder and label
binarizer fitted only on the training set.

## Metrics

The model was evaluated using precision, recall, and F1 score on the test dataset.

- **Precision:** 0.7419
- **Recall:** 0.6384
- **F1 Score:** 0.6863

The model also computes performance on slices of the data across all categorical
features. Results are saved in slice_output.txt. As an example, performance on
the workclass feature varies notably, with Federal-gov workers achieving an F1
of 0.7914 and workers in the ? category achieving an F1 of 0.5000.

## Ethical Considerations

This model is trained on demographic data that includes sensitive attributes such
as race, sex, and native country. These features may introduce bias into the
model's predictions. For example, the model may perform differently across
demographic groups, which could lead to unfair outcomes if used in real-world
decision-making. Users should be aware of these limitations and avoid using this
model in any context where biased predictions could cause harm.

## Caveats and Recommendations

- The model is trained on U.S. Census data from the 1990s and may not reflect
  current income distributions or demographic patterns.
- The model should not be used for hiring, lending, or other consequential
  decisions without careful fairness analysis.
- Performance varies across demographic slices, as shown in slice_output.txt.
  Users are encouraged to review these results before deploying the model.
- Future improvements could include hyperparameter tuning, feature engineering,
  or using a more recent dataset to improve both accuracy and fairness.