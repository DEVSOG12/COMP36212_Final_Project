#include "optimiser.h"
#include "mnist_helper.h"
#include "neural_network.h"
#include "math.h"
#include <time.h>

// Function declarations
void update_parameters(unsigned int batch_size);
void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy);

// Optimisation parameters
unsigned int log_freq = 30000; // Compute and print accuracy every log_freq iterations

// Parameters passed from command line arguments
unsigned int num_batches;
unsigned int batch_size;
unsigned int total_epochs;
double learning_rate;
double initial_learning_rate; // Store initial learning rate for learning rate decay

// Parameters for advanced optimization methods
optimisation_method_t opt_method = SGD; // Default to basic SGD
double momentum_param = 0.0;        // Momentum parameter (alpha)
double final_learning_rate = 0.0;   // Final learning rate for decay

// Adam optimizer parameters
double beta1 = 0.9;             // Exponential decay rate for first moment estimates
double beta2 = 0.999;           // Exponential decay rate for second moment estimates
double epsilon = 1e-8;          // Small constant to prevent division by zero
unsigned int t = 0;             // Time step counter for Adam

// RMSProp parameters
double rho = 0.9;              // Decay rate for moving average
double rmsprop_epsilon = 1e-8; // Small constant to prevent division by zero

// Timing variables
struct timespec start_time, end_time;
double total_training_time = 0.0;
double epoch_times[100]; // Store times for each epoch
unsigned int epoch_time_idx = 0;

void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy){
    // Calculate elapsed time for this epoch
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double epoch_time = (end_time.tv_sec - start_time.tv_sec) + 
                       (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    epoch_times[epoch_time_idx++] = epoch_time;
    total_training_time += epoch_time;
    
    printf("Epoch: %u,  Total iter: %u,  Mean Loss: %0.12f,  Test Acc: %f,  LR: %f,  Time: %.2fs\n", 
           epoch_counter, total_iter, mean_loss, test_accuracy, learning_rate, epoch_time);
    
    // Reset timer for next epoch
    clock_gettime(CLOCK_MONOTONIC, &start_time);
}

void initialise_optimiser(double cmd_line_learning_rate, int cmd_line_batch_size, int cmd_line_total_epochs){
    batch_size = cmd_line_batch_size;
    learning_rate = cmd_line_learning_rate;
    initial_learning_rate = cmd_line_learning_rate; // Store for learning rate decay
    total_epochs = cmd_line_total_epochs;
    
    num_batches = total_epochs * (N_TRAINING_SET / batch_size);
    printf("Optimising with parameters: \n\tepochs = %u \n\tbatch_size = %u \n\tnum_batches = %u\n\tlearning_rate = %f\n\n",
           total_epochs, batch_size, num_batches, learning_rate);
}

void set_optimisation_method(optimisation_method_t method, double momentum, double final_lr) {
    opt_method = method;
    momentum_param = momentum;
    final_learning_rate = final_lr;
    
    printf("Setting optimization method: ");
    switch(opt_method) {
        case SGD:
            printf("SGD (basic)\n");
            break;
        case SGD_MOMENTUM:
            printf("SGD with Momentum (alpha=%.4f)\n", momentum_param);
            break;
        case SGD_LR_DECAY:
            printf("SGD with Learning Rate Decay (initial=%.4f, final=%.4f)\n", 
                  learning_rate, final_learning_rate);
            break;
        case SGD_MOMENTUM_LR_DECAY:
            printf("SGD with Momentum and Learning Rate Decay (momentum=%.4f, initial_lr=%.4f, final_lr=%.4f)\n", 
                  momentum_param, learning_rate, final_learning_rate);
            break;
        case ADAM:
            printf("Adam (beta1=%.4f, beta2=%.4f, epsilon=%.8f)\n", 
                  beta1, beta2, epsilon);
            break;
    }
}

void set_adam_parameters(double b1, double b2, double eps) {
    beta1 = b1;
    beta2 = b2;
    epsilon = eps;
    
    printf("Adam parameters set: beta1=%.4f, beta2=%.4f, epsilon=%.8f\n", 
           beta1, beta2, epsilon);
}

void set_rmsprop_parameters(double decay_rate, double eps) {
    rho = decay_rate;
    rmsprop_epsilon = eps;
    
    printf("RMSProp parameters set: rho=%.4f, epsilon=%.8f\n", 
           rho, rmsprop_epsilon);
}

// Update learning rate based on current epoch (linear decay)
void update_learning_rate(unsigned int epoch) {
    if (opt_method == SGD_LR_DECAY || opt_method == SGD_MOMENTUM_LR_DECAY) {
        double alpha = (double)epoch / (double)total_epochs;
        learning_rate = initial_learning_rate * (1.0 - alpha) + final_learning_rate * alpha;
    }
}

void run_optimisation(void){
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;  //evaluate_testing_accuracy();
    double mean_loss = 0.0;
    
    // Start timing
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // Run optimiser - update parameters after each minibatch
    for (int i=0; i < num_batches; i++){
        for (int j = 0; j < batch_size; j++){

            // Evaluate accuracy on testing set (expensive, evaluate infrequently)
            if (total_iter % log_freq == 0 || total_iter == 0){
                if (total_iter > 0){
                    mean_loss = mean_loss/((double) log_freq);
                }
                
                test_accuracy = evaluate_testing_accuracy();
                print_training_stats(epoch_counter, total_iter, mean_loss, test_accuracy);

                // Reset mean_loss for next reporting period
                mean_loss = 0.0;
            }
            
            // Evaluate forward pass and calculate gradients
            obj_func = evaluate_objective_function(training_sample);
            mean_loss+=obj_func;

            // Update iteration counters (reset at end of training set to allow multiple epochs)
            total_iter++;
            training_sample++;
            // On epoch completion:
            if (training_sample == N_TRAINING_SET){
                training_sample = 0;
                epoch_counter++;
                
                // Update learning rate if using decay
                update_learning_rate(epoch_counter);
            }
        }
        
        // Update weights on batch completion
        update_parameters(batch_size);
    }
    
    // Print final performance and timing summary
    test_accuracy = evaluate_testing_accuracy();
    print_training_stats(epoch_counter, total_iter, (mean_loss/((double) log_freq)), test_accuracy);
    
    printf("\nTraining Summary:\n");
    printf("Total training time: %.2fs\n", total_training_time);
    printf("Average epoch time: %.2fs\n", total_training_time / epoch_counter);
    printf("Time per batch: %.2fms\n", (total_training_time * 1000) / num_batches);
}

double evaluate_objective_function(unsigned int sample){

    // Compute network performance
    evaluate_forward_pass(training_data, sample);
    double loss = compute_xent_loss(training_labels[sample]);
    
    // Evaluate gradients
    //evaluate_backward_pass(training_labels[sample], sample);
    evaluate_backward_pass_sparse(training_labels[sample], sample);
    
    // Evaluate parameter updates
    store_gradient_contributions();
    
    return loss;
}

void update_parameters(unsigned int batch_size){
    // Learning rate normalized by batch size
    double lr = learning_rate / (double)batch_size;
    
    // Update weights based on the selected optimization method
    switch(opt_method) {
        case SGD:
            // Standard SGD update
            for (int i = 0; i < N_NEURONS_L3; i++) {
                for (int j = 0; j < N_NEURONS_LO; j++) {
                    w_L3_LO[i][j].w -= lr * w_L3_LO[i][j].dw;
                    w_L3_LO[i][j].dw = 0.0;
                }
            }
            
            for (int i = 0; i < N_NEURONS_L2; i++) {
                for (int j = 0; j < N_NEURONS_L3; j++) {
                    w_L2_L3[i][j].w -= lr * w_L2_L3[i][j].dw;
                    w_L2_L3[i][j].dw = 0.0;
                }
            }
            
            for (int i = 0; i < N_NEURONS_L1; i++) {
                for (int j = 0; j < N_NEURONS_L2; j++) {
                    w_L1_L2[i][j].w -= lr * w_L1_L2[i][j].dw;
                    w_L1_L2[i][j].dw = 0.0;
                }
            }
            
            for (int i = 0; i < N_NEURONS_LI; i++) {
                for (int j = 0; j < N_NEURONS_L1; j++) {
                    w_LI_L1[i][j].w -= lr * w_LI_L1[i][j].dw;
                    w_LI_L1[i][j].dw = 0.0;
                }
            }
            break;
            
        case SGD_MOMENTUM:
        case SGD_MOMENTUM_LR_DECAY:
            // SGD with momentum update
            for (int i = 0; i < N_NEURONS_L3; i++) {
                for (int j = 0; j < N_NEURONS_LO; j++) {
                    w_L3_LO[i][j].v = momentum_param * w_L3_LO[i][j].v - lr * w_L3_LO[i][j].dw;
                    w_L3_LO[i][j].w += w_L3_LO[i][j].v;
                    w_L3_LO[i][j].dw = 0.0;
                }
            }
            
            for (int i = 0; i < N_NEURONS_L2; i++) {
                for (int j = 0; j < N_NEURONS_L3; j++) {
                    w_L2_L3[i][j].v = momentum_param * w_L2_L3[i][j].v - lr * w_L2_L3[i][j].dw;
                    w_L2_L3[i][j].w += w_L2_L3[i][j].v;
                    w_L2_L3[i][j].dw = 0.0;
                }
            }
            
            for (int i = 0; i < N_NEURONS_L1; i++) {
                for (int j = 0; j < N_NEURONS_L2; j++) {
                    w_L1_L2[i][j].v = momentum_param * w_L1_L2[i][j].v - lr * w_L1_L2[i][j].dw;
                    w_L1_L2[i][j].w += w_L1_L2[i][j].v;
                    w_L1_L2[i][j].dw = 0.0;
                }
            }
            
            for (int i = 0; i < N_NEURONS_LI; i++) {
                for (int j = 0; j < N_NEURONS_L1; j++) {
                    w_LI_L1[i][j].v = momentum_param * w_LI_L1[i][j].v - lr * w_LI_L1[i][j].dw;
                    w_LI_L1[i][j].w += w_LI_L1[i][j].v;
                    w_LI_L1[i][j].dw = 0.0;
                }
            }
            break;
            
        case SGD_LR_DECAY:
            // Same as SGD but with learning rate decay (already handled in update_learning_rate)
            for (int i = 0; i < N_NEURONS_L3; i++) {
                for (int j = 0; j < N_NEURONS_LO; j++) {
                    w_L3_LO[i][j].w -= lr * w_L3_LO[i][j].dw;
                    w_L3_LO[i][j].dw = 0.0;
                }
            }
            
            for (int i = 0; i < N_NEURONS_L2; i++) {
                for (int j = 0; j < N_NEURONS_L3; j++) {
                    w_L2_L3[i][j].w -= lr * w_L2_L3[i][j].dw;
                    w_L2_L3[i][j].dw = 0.0;
                }
            }
            
            for (int i = 0; i < N_NEURONS_L1; i++) {
                for (int j = 0; j < N_NEURONS_L2; j++) {
                    w_L1_L2[i][j].w -= lr * w_L1_L2[i][j].dw;
                    w_L1_L2[i][j].dw = 0.0;
                }
            }
            
            for (int i = 0; i < N_NEURONS_LI; i++) {
                for (int j = 0; j < N_NEURONS_L1; j++) {
                    w_LI_L1[i][j].w -= lr * w_LI_L1[i][j].dw;
                    w_LI_L1[i][j].dw = 0.0;
                }
            }
            break;
            
        case ADAM:
            // Increment the time step for Adam
            t++;
            
            // Bias correction factors
            double bc1 = 1.0 - pow(beta1, t);
            double bc2 = 1.0 - pow(beta2, t);
            
            // Layer 3 -> Output
            for (int i = 0; i < N_NEURONS_L3; i++) {
                for (int j = 0; j < N_NEURONS_LO; j++) {
                    // Update biased first moment estimate
                    w_L3_LO[i][j].m = beta1 * w_L3_LO[i][j].m + (1.0 - beta1) * w_L3_LO[i][j].dw;
                    
                    // Update biased second raw moment estimate
                    w_L3_LO[i][j].v_adam = beta2 * w_L3_LO[i][j].v_adam + (1.0 - beta2) * pow(w_L3_LO[i][j].dw, 2);
                    
                    // Compute bias-corrected first moment estimate
                    double m_hat = w_L3_LO[i][j].m / bc1;
                    
                    // Compute bias-corrected second raw moment estimate
                    double v_hat = w_L3_LO[i][j].v_adam / bc2;
                    
                    // Update parameters
                    w_L3_LO[i][j].w -= lr * m_hat / (sqrt(v_hat) + epsilon);
                    
                    // Reset gradient for next batch
                    w_L3_LO[i][j].dw = 0.0;
                }
            }
            
            // Layer 2 -> Layer 3
            for (int i = 0; i < N_NEURONS_L2; i++) {
                for (int j = 0; j < N_NEURONS_L3; j++) {
                    w_L2_L3[i][j].m = beta1 * w_L2_L3[i][j].m + (1.0 - beta1) * w_L2_L3[i][j].dw;
                    w_L2_L3[i][j].v_adam = beta2 * w_L2_L3[i][j].v_adam + (1.0 - beta2) * pow(w_L2_L3[i][j].dw, 2);
                    
                    double m_hat = w_L2_L3[i][j].m / bc1;
                    double v_hat = w_L2_L3[i][j].v_adam / bc2;
                    
                    w_L2_L3[i][j].w -= lr * m_hat / (sqrt(v_hat) + epsilon);
                    w_L2_L3[i][j].dw = 0.0;
                }
            }
            
            // Layer 1 -> Layer 2
            for (int i = 0; i < N_NEURONS_L1; i++) {
                for (int j = 0; j < N_NEURONS_L2; j++) {
                    w_L1_L2[i][j].m = beta1 * w_L1_L2[i][j].m + (1.0 - beta1) * w_L1_L2[i][j].dw;
                    w_L1_L2[i][j].v_adam = beta2 * w_L1_L2[i][j].v_adam + (1.0 - beta2) * pow(w_L1_L2[i][j].dw, 2);
                    
                    double m_hat = w_L1_L2[i][j].m / bc1;
                    double v_hat = w_L1_L2[i][j].v_adam / bc2;
                    
                    w_L1_L2[i][j].w -= lr * m_hat / (sqrt(v_hat) + epsilon);
                    w_L1_L2[i][j].dw = 0.0;
                }
            }
            
            // Input -> Layer 1
            for (int i = 0; i < N_NEURONS_LI; i++) {
                for (int j = 0; j < N_NEURONS_L1; j++) {
                    w_LI_L1[i][j].m = beta1 * w_LI_L1[i][j].m + (1.0 - beta1) * w_LI_L1[i][j].dw;
                    w_LI_L1[i][j].v_adam = beta2 * w_LI_L1[i][j].v_adam + (1.0 - beta2) * pow(w_LI_L1[i][j].dw, 2);
                    
                    double m_hat = w_LI_L1[i][j].m / bc1;
                    double v_hat = w_LI_L1[i][j].v_adam / bc2;
                    
                    w_LI_L1[i][j].w -= lr * m_hat / (sqrt(v_hat) + epsilon);
                    w_LI_L1[i][j].dw = 0.0;
                }
            }
            break;
        
        case RMSPROP:
            // RMSProp update
            for (int i = 0; i < N_NEURONS_L3; i++) {
                for (int j = 0; j < N_NEURONS_LO; j++) {
                    // Update moving average of squared gradients
                    w_L3_LO[i][j].v = rho * w_L3_LO[i][j].v + (1 - rho) * w_L3_LO[i][j].dw * w_L3_LO[i][j].dw;
                    // Update weights
                    w_L3_LO[i][j].w -= lr * w_L3_LO[i][j].dw / (sqrt(w_L3_LO[i][j].v) + rmsprop_epsilon);
                    w_L3_LO[i][j].dw = 0.0;
                }
            }
            
            for (int i = 0; i < N_NEURONS_L2; i++) {
                for (int j = 0; j < N_NEURONS_L3; j++) {
                    w_L2_L3[i][j].v = rho * w_L2_L3[i][j].v + (1 - rho) * w_L2_L3[i][j].dw * w_L2_L3[i][j].dw;
                    w_L2_L3[i][j].w -= lr * w_L2_L3[i][j].dw / (sqrt(w_L2_L3[i][j].v) + rmsprop_epsilon);
                    w_L2_L3[i][j].dw = 0.0;
                }
            }
            
            for (int i = 0; i < N_NEURONS_L1; i++) {
                for (int j = 0; j < N_NEURONS_L2; j++) {
                    w_L1_L2[i][j].v = rho * w_L1_L2[i][j].v + (1 - rho) * w_L1_L2[i][j].dw * w_L1_L2[i][j].dw;
                    w_L1_L2[i][j].w -= lr * w_L1_L2[i][j].dw / (sqrt(w_L1_L2[i][j].v) + rmsprop_epsilon);
                    w_L1_L2[i][j].dw = 0.0;
                }
            }
            
            for (int i = 0; i < N_NEURONS_LI; i++) {
                for (int j = 0; j < N_NEURONS_L1; j++) {
                    w_LI_L1[i][j].v = rho * w_LI_L1[i][j].v + (1 - rho) * w_LI_L1[i][j].dw * w_LI_L1[i][j].dw;
                    w_LI_L1[i][j].w -= lr * w_LI_L1[i][j].dw / (sqrt(w_LI_L1[i][j].v) + rmsprop_epsilon);
                    w_LI_L1[i][j].dw = 0.0;
                }
            }
            break;
    }
}
