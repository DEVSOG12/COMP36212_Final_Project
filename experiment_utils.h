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

// Function to run experiments with different configurations
void run_experiments();

#endif /* EXPERIMENT_UTILS_H */ 