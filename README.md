# RIFair

This repository provides the implementation of **RIFair** for *Perturbation Effects on Accuracy and Individual Fairness*. 

RIFair is a framework for evaluating **robust individual fairness** in NLP models by generating *semantically equivalent and imperceptible adversarial similar instances* to expose violations of robustness and fairness under controlled perturbations.

### Package Requirements  
Python 3.10, TensorFlow 2.12.0, Keras 2.12.0, and PyTorch 2.0.1.

### Usage  

RIFair generates inaccurate or unfair adversarial similar instances through the following pipeline. 

First, run `1.get_attack_token.py` and `2.get_attack_token_candidates.py` to extract high-frequency tokens and construct their **semantically equivalent replacement candidates**. Next, run `3.get_attack_candidates_similar_scores.py` to compute similarity scores using **NLI-based entailment** and **cosine distance**, ensuring that perturbations preserve semantics. 

Then, run `5.1.get_importance_test_data.py`, `5.2.get_token_importance.py`, and `5.3.calculate_perturbation_importance.py` to estimate **black-box token importance** for guiding perturbations. 


Finally, run `7.1.get_RIF_perturbation_result.py` to generate **unrobust or unfair adversarial similar instances** under the RIFair framework.

If you have any questions or need further assistance, feel free to reach out.
