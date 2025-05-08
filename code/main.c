#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mnist_helper.h"
#include "neural_network.h"
#include "optimiser.h"
#include "experiment_utils.h"

void print_help_and_exit(char **argv) {
    printf("usage: %s <path_to_dataset> <learning_rate> <batch_size> <total_epochs> [options]\n", argv[0]);
    printf("Options:\n");
    printf("  -v 1                        Enable gradient verification\n");
    printf("  -m <method>                 Optimization method (0=SGD, 1=SGD_MOMENTUM, 2=SGD_LR_DECAY,\n");
    printf("                                                   3=SGD_MOMENTUM_LR_DECAY, 4=ADAM, 5=RMSPROP)\n");
    printf("  -momentum <value>           Momentum parameter (default: 0.9)\n");
    printf("  -final_lr <value>           Final learning rate for decay (default: 0.001)\n");
    printf("  -beta1 <value>              Beta1 for Adam (default: 0.9)\n");
    printf("  -beta2 <value>              Beta2 for Adam (default: 0.999)\n");
    printf("  -epsilon <value>            Epsilon for Adam/RMSProp (default: 1e-8)\n");
    printf("  -rho <value>                Decay rate for RMSProp (default: 0.9)\n");
    printf("  -run_experiments            Run all experiments for analysis\n");
    exit(0);
}

int main(int argc, char** argv) {
    
    if(argc < 5){
        printf("ERROR: incorrect number of arguments\n");
        print_help_and_exit(argv);
    }
    
    const char* path_to_dataset = argv[1];
    double learning_rate = atof(argv[2]);
    unsigned int batch_size = atoi(argv[3]);
    unsigned int total_epochs = atoi(argv[4]);
    
    // Optional parameters with defaults
    int verify_gradients_flag = 0;
    optimisation_method_t opt_method = SGD;
    double momentum = 0.9;
    double final_lr = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double rho = 0.9;
    int run_experiments_flag = 0;
    
    // Parse optional parameters
    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            verify_gradients_flag = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            int method_int = atoi(argv[i + 1]);
            if (method_int >= 0 && method_int <= 5) {
                opt_method = (optimisation_method_t)method_int;
            } else {
                printf("ERROR: invalid optimization method\n");
                print_help_and_exit(argv);
            }
            i++;
        } else if (strcmp(argv[i], "-momentum") == 0 && i + 1 < argc) {
            momentum = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-final_lr") == 0 && i + 1 < argc) {
            final_lr = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-beta1") == 0 && i + 1 < argc) {
            beta1 = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-beta2") == 0 && i + 1 < argc) {
            beta2 = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-epsilon") == 0 && i + 1 < argc) {
            eps = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-rho") == 0 && i + 1 < argc) {
            rho = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-run_experiments") == 0) {
            run_experiments_flag = 1;
        } else {
            printf("ERROR: unknown parameter: %s\n", argv[i]);
            print_help_and_exit(argv);
        }
    }
    
    if(!path_to_dataset || !learning_rate || !batch_size || !total_epochs) {
        printf("ERROR: invalid argument\n");
        print_help_and_exit(argv);
    }
    
    printf("********************************************************************************\n");
    printf("Initialising Dataset... \n");
    printf("********************************************************************************\n");
    initialise_dataset(path_to_dataset,
                       0 // print flag
                       );

    // Run all experiments if requested
    if (run_experiments_flag) {
        printf("********************************************************************************\n");
        printf("Running all experiments for analysis...\n");
        printf("********************************************************************************\n");
        run_experiments();
        printf("********************************************************************************\n");
        printf("All experiments completed. Run analyze_results.py to generate plots.\n");
        printf("********************************************************************************\n");
        free_dataset_data_structures();
        return 0;
    }

    printf("********************************************************************************\n");
    printf("Initialising neural network... \n");
    printf("********************************************************************************\n");
    initialise_nn();

    // Verify gradients if requested
    if (verify_gradients_flag) {
        printf("********************************************************************************\n");
        printf("Verifying gradients...\n");
        printf("********************************************************************************\n");
        // Choose a random sample for verification
        unsigned int random_sample = rand() % N_TRAINING_SET;
        verify_gradients(random_sample);
    }

    printf("********************************************************************************\n");
    printf("Initialising optimiser...\n");
    printf("********************************************************************************\n");
    initialise_optimiser(learning_rate, batch_size, total_epochs);
    
    // Set optimization method
    printf("********************************************************************************\n");
    printf("Setting optimization method...\n");
    printf("********************************************************************************\n");
    set_optimisation_method(opt_method, momentum, final_lr);
    
    // Set optimizer-specific parameters
    if (opt_method == ADAM) {
        set_adam_parameters(beta1, beta2, eps);
    } else if (opt_method == RMSPROP) {
        set_rmsprop_parameters(rho, eps);
    }

    printf("********************************************************************************\n");
    printf("Performing training optimisation...\n");
    printf("********************************************************************************\n");
    run_optimisation();
    
    printf("********************************************************************************\n");
    printf("Program complete... \n");
    printf("********************************************************************************\n");
    free_dataset_data_structures();
    return 0;
}
