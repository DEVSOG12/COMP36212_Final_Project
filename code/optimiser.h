#ifndef OPTIMISER_H
#define OPTIMISER_H

#include <stdio.h>

// Optimization method flags
typedef enum {
    SGD,
    SGD_MOMENTUM,
    SGD_LR_DECAY,
    SGD_MOMENTUM_LR_DECAY,
    ADAM,
    RMSPROP
} optimisation_method_t;

void initialise_optimiser(double learning_rate, int batch_size, int total_epochs);
void run_optimisation(void);
double evaluate_objective_function(unsigned int sample);
void update_parameters(unsigned int batch_size);

// Functions for advanced optimization methods
void update_learning_rate(unsigned int epoch);
void set_optimisation_method(optimisation_method_t method, double momentum_param, double final_lr);
void set_adam_parameters(double beta1, double beta2, double epsilon);
void set_rmsprop_parameters(double decay_rate, double epsilon);

#endif /* OPTMISER_H */
