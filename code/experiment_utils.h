#ifndef EXPERIMENT_UTILS_H
#define EXPERIMENT_UTILS_H

#include <stdio.h>
#include "optimiser.h"

// Import function from optimiser.c
extern void update_parameters(unsigned int batch_size);

// Structure to store experiment results
typedef struct {
    unsigned int epoch;
    unsigned int iteration;
    double loss;
    double accuracy;
    double learning_rate;
} result_point_t;

// Function to initialize the results file
void init_results_file(const char* filename);

// Function to log a result point
void log_result(const char* filename, result_point_t result);

// Run all experiments
void run_experiments(void);

// Individual experiment runners
void run_sgd_experiments(void);
void run_momentum_experiments(void);
void run_lr_decay_experiments(void);
void run_combined_experiments(void);
void run_adam_experiments(void);
void run_rmsprop_experiments(void);

// Helper function to run a single experiment
void run_single_experiment(optimisation_method_t method, double learning_rate, 
                         int batch_size, int epochs, double momentum, 
                         double final_lr, const char* output_file);

// Save results to CSV file
void save_results(const char* filename);

#endif /* EXPERIMENT_UTILS_H */ 