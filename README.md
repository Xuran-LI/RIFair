# RIFair

Code for RIFair: "Perturbation Effects on Accuracy and Individual Fairness among Similar Individuals"

# Package Requirements:

Python 3.8,

tensorflow 2.4.1,

numpy 1.19.5,

keras 2.4.3,

scikit-learn 1.0.2,

pandas 1.4.3,


# RIFair for evaluating Robust Individual Fairness

This package provides code for evaluating robust individual fairness using the RIFair method. 
The following steps outline the process for conducting experiments on the Band dataset:

1.Run 1.prepare_data.py to prepare train and test data for the experiment.

2.Run 2.train_model.py to train the baseline AutoInt classification model.

3.Run 3.RIFair_FB.py, 4.RIFair_FF.py, and 5.RIFair_TB.py to generate false biased, false fair, and true biased adversarial instances.

4.Run 6.get_result.py translate generated false biased, false fair, and true biased adversarial instances into text formats.

4.Run 7.retrain_model.py to retrain the model utilizing generated adversarial instances by RIFair from training dataset.


# Exporting experiment results

5.Run 8.check_model.py and 9.get_social_impact_Acc_IF.py to export the experiment results as worksheets.


If you have any questions or need further assistance, please reach out to us.
