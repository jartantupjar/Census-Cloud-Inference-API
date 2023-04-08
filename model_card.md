# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Model Used: scikit-learn's GradientBoostingClassifier
- Model Creation Date: April 8, 2023, 3:50 PM

## Intended Use
- The model was developed for educational purposes and is used to predict whether an individual earns more than $50k or less than or equal to $50k.

## Training Data
- The model was trained on a dataset obtained from the Census Bureau, which was publicly available.
- The training dataset consisted of labeled data with features such as age, education level, occupation, etc., and corresponding labels indicating whether the individual's income was greater than $50k or less than or equal to $50k.

## Evaluation Data
- The model's performance was evaluated on a test dataset that was not used during training but is from the same source where 20% was split off randomly.

## Metrics
- Accuracy metrics were used to evaluate the model's performance.
- Training Dataset Results:
  - Precision: 0.797
  - Recall: 0.610
  - F-beta Score: 0.691
- Test Dataset Results:
  - Precision: 0.806
  - Recall: 0.627
  - F-beta Score: 0.705
- Precision, Recall, and F-beta score were used as they are commonly used metrics for binary classification tasks like this one. Precision measures the proportion of true positive predictions out of the total positive predictions, providing insight into the model's ability to correctly identify positive instances. Recall measures the proportion of true positive predictions out of the actual positive instances in the dataset, providing information about the model's ability to capture all the positive instances. F-beta score combines both precision and recall, allowing for a trade-off between precision and recall by using a beta parameter. In this case, F-beta score with a beta value of 1 was used to equally weigh precision and recall. These metrics were chosen to provide a comprehensive evaluation of the model's accuracy, ability to identify positive instances, and balance between precision and recall, which are important considerations in this classification task.

## Ethical Considerations
- As the model uses publicly available Census Bureau data, there are no specific ethical concerns related to data privacy or bias. However, it is important to note that using this model for any real-world applications may require additional ethical considerations, such as ensuring fairness, accountability, and transparency in decision-making.

## Caveats and Recommendations
- The model's performance may vary depending on the quality and representativeness of the data used for training and evaluation.
- As with any machine learning model, it is important to thoroughly validate the model's predictions and consider other factors before making any decisions based solely on the model's output.
- It is recommended to periodically update the model with new data to maintain its accuracy and relevance.
- Further investigation and analysis may be required to identify and mitigate any potential biases or limitations in the model's predictions.
- It is important to use the model responsibly and in compliance with applicable laws and regulations.
