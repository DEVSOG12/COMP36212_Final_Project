# Submission Checklist for COMP36212 EX3

## Code Implementation
- [x] Part I: Stochastic Gradient Descent
  - [x] Implemented SGD update rule
  - [x] Implemented gradient verification using numerical differentiation
  - [x] Tested with batch size 10 and learning rate 0.1

- [x] Part II: Improving Convergence
  - [x] Implemented and explored batch size variations
  - [x] Implemented and explored learning rate variations
  - [x] Implemented learning rate decay
  - [x] Implemented momentum
  - [x] Combined learning rate decay and momentum

- [x] Part III: Adaptive Learning Rate
  - [x] Implemented Adam optimizer with bias correction

## Experiments and Analysis
- [x] Run experiments for all optimization methods
- [x] Generated plots for loss and accuracy
- [x] Analyzed performance of each method
- [x] Compared different approaches

## Report
- [x] Introduction and problem statement
- [x] Part I description and analysis
  - [x] SGD implementation details
  - [x] Gradient verification analysis
  - [x] SGD performance results
- [x] Part II description and analysis
  - [x] Effect of batch size and learning rate
  - [x] Learning rate decay implementation and results
  - [x] Momentum implementation and results
  - [x] Combined approaches and their benefits
- [x] Part III description and analysis
  - [x] Adam implementation details
  - [x] Adam performance results and comparison
- [x] Discussion and conclusions
  - [x] Summary of findings
  - [x] Trade-offs between methods
  - [x] Optimal approach for the given problem
  - [x] Generalization to other problems
- [x] References

## Submission Files
- [x] Report in PDF format (max 8 pages)
- [x] Code files (zipped):
  - [x] main.c
  - [x] mnist_helper.c
  - [x] mnist_helper.h
  - [x] neural_network.c
  - [x] neural_network.h
  - [x] optimiser.c
  - [x] optimiser.h
  - [x] experiment_utils.c
  - [x] experiment_utils.h
  - [x] makefile
  - [x] run.sh
  - [x] analyze_results.py

## Final Check
- [ ] Report is within the page limit (8 pages max)
- [ ] Code is well-commented and readable
- [ ] All figures/tables are properly referenced in the text
- [ ] Report follows the required structure
- [ ] Code runs successfully with provided scripts 