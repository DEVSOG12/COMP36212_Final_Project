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

void run_experiments(void) {
    // Part I: Basic SGD
    printf("Running Part I: Basic SGD experiments...\n");
    run_sgd_experiments();
    
    // Part II: Improving convergence
    printf("\nRunning Part II: Improving convergence experiments...\n");
    run_momentum_experiments();
    run_lr_decay_experiments();
    run_combined_experiments();
    
    // Part III: Adaptive learning
    printf("\nRunning Part III: Adaptive learning experiments...\n");
    run_adam_experiments();
    run_rmsprop_experiments();
}

void run_sgd_experiments(void) {
    // Learning rates to try
    double learning_rates[] = {0.1, 0.01, 0.001};
    // Batch sizes to try
    int batch_sizes[] = {1, 10, 100};
    
    for (int i = 0; i < sizeof(learning_rates)/sizeof(double); i++) {
        for (int j = 0; j < sizeof(batch_sizes)/sizeof(int); j++) {
            char filename[100];
            snprintf(filename, sizeof(filename), "results_sgd_%.3f_%d.csv", 
                    learning_rates[i], batch_sizes[j]);
            
            run_single_experiment(SGD, learning_rates[i], batch_sizes[j], 2, 0.0, 0.0, filename);
        }
    }
}

void run_momentum_experiments(void) {
    // Momentum values to try
    double momentums[] = {0.5, 0.7, 0.9, 0.95, 0.99};
    double learning_rate = 0.1;
    int batch_size = 10;
    
    for (int i = 0; i < sizeof(momentums)/sizeof(double); i++) {
        char filename[100];
        snprintf(filename, sizeof(filename), "results_sgd_momentum_%.2f.csv", momentums[i]);
        
        run_single_experiment(SGD_MOMENTUM, learning_rate, batch_size, 5, momentums[i], 0.0, filename);
    }
}

void run_lr_decay_experiments(void) {
    // Learning rate decay configurations
    struct lr_decay_config {
        double initial_lr;
        double final_lr;
        int epochs;
    } configs[] = {
        {0.1, 0.001, 5},  // Linear decay
        {0.1, 0.0001, 5}, // Steeper decay
        {0.01, 0.001, 5}  // Lower initial rate
    };
    
    int batch_size = 10;
    
    for (int i = 0; i < sizeof(configs)/sizeof(struct lr_decay_config); i++) {
        char filename[100];
        snprintf(filename, sizeof(filename), "results_sgd_lr_decay_%.3f_%.3f.csv", 
                configs[i].initial_lr, configs[i].final_lr);
        
        run_single_experiment(SGD_LR_DECAY, configs[i].initial_lr, batch_size, 
                            configs[i].epochs, 0.0, configs[i].final_lr, filename);
    }
}

void run_combined_experiments(void) {
    // Combined momentum and learning rate decay
    double momentums[] = {0.5, 0.7, 0.9};
    double initial_lr = 0.1;
    double final_lr = 0.001;
    int batch_size = 10;
    int epochs = 5;
    
    for (int i = 0; i < sizeof(momentums)/sizeof(double); i++) {
        char filename[100];
        snprintf(filename, sizeof(filename), "results_sgd_momentum_lr_decay_%.2f.csv", momentums[i]);
        
        run_single_experiment(SGD_MOMENTUM_LR_DECAY, initial_lr, batch_size, 
                            epochs, momentums[i], final_lr, filename);
    }
}

void run_adam_experiments(void) {
    // Adam hyperparameter grid
    double beta1s[] = {0.8, 0.9, 0.95, 0.99};
    double beta2s[] = {0.999, 0.9999};
    double epsilons[] = {1e-8, 1e-7, 1e-6};
    double learning_rates[] = {0.001, 0.0005, 0.0001};
    int batch_size = 10;
    int epochs = 5;
    
    for (int i = 0; i < sizeof(beta1s)/sizeof(double); i++) {
        for (int j = 0; j < sizeof(beta2s)/sizeof(double); j++) {
            for (int k = 0; k < sizeof(epsilons)/sizeof(double); k++) {
                for (int l = 0; l < sizeof(learning_rates)/sizeof(double); l++) {
                    char filename[100];
                    snprintf(filename, sizeof(filename), 
                            "results_adam_b1%.2f_b2%.4f_eps%.0e_lr%.4f.csv",
                            beta1s[i], beta2s[j], epsilons[k], learning_rates[l]);
                    
                    run_single_experiment(ADAM, learning_rates[l], batch_size, epochs, 0.0, 0.0, filename);
                    set_adam_parameters(beta1s[i], beta2s[j], epsilons[k]);
                }
            }
        }
    }
}

void run_rmsprop_experiments(void) {
    // RMSProp hyperparameter grid
    double rhos[] = {0.8, 0.9, 0.95, 0.99};
    double epsilons[] = {1e-8, 1e-7, 1e-6};
    double learning_rates[] = {0.001, 0.0005, 0.0001};
    int batch_size = 10;
    int epochs = 5;
    
    for (int i = 0; i < sizeof(rhos)/sizeof(double); i++) {
        for (int j = 0; j < sizeof(epsilons)/sizeof(double); j++) {
            for (int k = 0; k < sizeof(learning_rates)/sizeof(double); k++) {
                char filename[100];
                snprintf(filename, sizeof(filename), 
                        "results_rmsprop_rho%.2f_eps%.0e_lr%.4f.csv",
                        rhos[i], epsilons[j], learning_rates[k]);
                
                run_single_experiment(RMSPROP, learning_rates[k], batch_size, epochs, 0.0, 0.0, filename);
                set_rmsprop_parameters(rhos[i], epsilons[j]);
            }
        }
    }
}

void run_single_experiment(optimisation_method_t method, double learning_rate, 
                         int batch_size, int epochs, double momentum, 
                         double final_lr, const char* output_file) {
    printf("Running experiment: method=%d, lr=%.4f, batch=%d, epochs=%d, momentum=%.2f, final_lr=%.4f\n",
           method, learning_rate, batch_size, epochs, momentum, final_lr);
    
    // Initialize network and dataset
    initialise_nn();
    initialise_optimiser(learning_rate, batch_size, epochs);
    set_optimisation_method(method, momentum, final_lr);
    
    // Run optimization
    run_optimisation();
    
    // Save results
    save_results(output_file);
    
    // Clean up
    free_dataset_data_structures();
}

void save_results(const char* filename) {
    // Get the current results from the optimizer
    result_point_t result = {
        .epoch = 0,
        .iteration = 0,
        .loss = 0.0,
        .accuracy = evaluate_testing_accuracy(),
        .learning_rate = learning_rate
    };
    
    // Initialize the file if it doesn't exist
    init_results_file(filename);
    
    // Log the result
    log_result(filename, result);
} 