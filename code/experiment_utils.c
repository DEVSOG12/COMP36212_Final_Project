#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "experiment_utils.h"
#include "mnist_helper.h"
#include "neural_network.h"
#include "optimiser.h"

void init_results_file(const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }
    
    // Write header row
    fprintf(file, "epoch,iteration,loss,accuracy,learning_rate\n");
    fclose(file);
}

void log_result(const char* filename, result_point_t result) {
    FILE* file = fopen(filename, "a");
    if (file == NULL) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }
    
    // Write result point
    fprintf(file, "%u,%u,%f,%f,%f\n", 
            result.epoch, 
            result.iteration, 
            result.loss, 
            result.accuracy, 
            result.learning_rate);
    fclose(file);
}

// Run experiment with a specific configuration and save results to a file
void run_experiment(const char* filename, 
                   double learning_rate, 
                   unsigned int batch_size, 
                   unsigned int epochs,
                   optimisation_method_t opt_method,
                   double momentum,
                   double final_lr,
                   double beta1,
                   double beta2,
                   double epsilon) {
    // Initialize the results file
    init_results_file(filename);
    
    // Initialize the neural network
    initialise_nn();
    
    // Initialize the optimizer
    initialise_optimiser(learning_rate, batch_size, epochs);
    
    // Set optimization method
    set_optimisation_method(opt_method, momentum, final_lr);
    
    // Set Adam parameters if using Adam
    if (opt_method == ADAM) {
        set_adam_parameters(beta1, beta2, epsilon);
    }
    
    // Variables for tracking
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;
    double mean_loss = 0.0;
    unsigned int log_freq = 1000; // More frequent logging for plots
    unsigned int log_counter = 0;
    
    // Calculate total number of batches
    unsigned int num_batches = epochs * (N_TRAINING_SET / batch_size);
    
    // Initial accuracy
    test_accuracy = evaluate_testing_accuracy();
    
    // Log initial point
    result_point_t result = {
        .epoch = 0,
        .iteration = 0,
        .loss = 0.0,
        .accuracy = test_accuracy,
        .learning_rate = learning_rate
    };
    log_result(filename, result);
    
    // Run optimization
    for (int i = 0; i < num_batches; i++) {
        for (int j = 0; j < batch_size; j++) {
            // Evaluate objective function
            obj_func = evaluate_objective_function(training_sample);
            mean_loss += obj_func;
            
            // Update counters
            total_iter++;
            log_counter++;
            training_sample++;
            
            // Log results periodically
            if (log_counter >= log_freq) {
                test_accuracy = evaluate_testing_accuracy();
                mean_loss = mean_loss / log_counter;
                
                // Log result
                result.epoch = epoch_counter;
                result.iteration = total_iter;
                result.loss = mean_loss;
                result.accuracy = test_accuracy;
                result.learning_rate = learning_rate;
                log_result(filename, result);
                
                // Reset for next period
                mean_loss = 0.0;
                log_counter = 0;
            }
            
            // Check if we need to move to the next epoch
            if (training_sample == N_TRAINING_SET) {
                training_sample = 0;
                epoch_counter++;
                
                // Update learning rate if using decay
                update_learning_rate(epoch_counter);
            }
        }
        
        // Update weights after batch
        update_parameters(batch_size);
    }
    
    // Log final result
    if (log_counter > 0) {
        test_accuracy = evaluate_testing_accuracy();
        mean_loss = mean_loss / log_counter;
        
        result.epoch = epoch_counter;
        result.iteration = total_iter;
        result.loss = mean_loss;
        result.accuracy = test_accuracy;
        result.learning_rate = learning_rate;
        log_result(filename, result);
    }
}

void run_experiments() {
    printf("Running experiments for analysis...\n");
    
    // Part I - Basic SGD with different configurations
    printf("Part I: Running SGD experiments\n");
    run_experiment("results_sgd_0.1_10.csv", 0.1, 10, 2, SGD, 0.0, 0.0, 0.0, 0.0, 0.0);
    
    // Part II - Exploring batch size and learning rate
    printf("Part II-A: Exploring batch size and learning rate\n");
    // Batch size variations
    run_experiment("results_sgd_0.1_1.csv", 0.1, 1, 2, SGD, 0.0, 0.0, 0.0, 0.0, 0.0);
    run_experiment("results_sgd_0.1_100.csv", 0.1, 100, 2, SGD, 0.0, 0.0, 0.0, 0.0, 0.0);
    
    // Learning rate variations
    run_experiment("results_sgd_0.01_10.csv", 0.01, 10, 2, SGD, 0.0, 0.0, 0.0, 0.0, 0.0);
    run_experiment("results_sgd_0.001_10.csv", 0.001, 10, 2, SGD, 0.0, 0.0, 0.0, 0.0, 0.0);
    
    // Part II - Learning rate decay
    printf("Part II-B: Learning rate decay\n");
    run_experiment("results_sgd_lr_decay_0.1_0.001.csv", 0.1, 10, 2, SGD_LR_DECAY, 0.0, 0.001, 0.0, 0.0, 0.0);
    
    // Part II - Momentum
    printf("Part II-C: Momentum\n");
    run_experiment("results_sgd_momentum_0.9.csv", 0.01, 10, 2, SGD_MOMENTUM, 0.9, 0.0, 0.0, 0.0, 0.0);
    
    // Part II - Combined approaches
    printf("Part II-D: Combined approaches\n");
    run_experiment("results_sgd_momentum_lr_decay.csv", 0.1, 10, 2, SGD_MOMENTUM_LR_DECAY, 0.9, 0.001, 0.0, 0.0, 0.0);
    
    // Part III - Adam
    printf("Part III: Adam optimizer\n");
    run_experiment("results_adam.csv", 0.001, 10, 2, ADAM, 0.0, 0.0, 0.9, 0.999, 1e-8);
    
    printf("All experiments completed.\n");
} 