all: mnist_optimiser.out

mnist_optimiser.out:
	gcc -o mnist_optimiser.out main.c mnist_helper.c neural_network.c optimiser.c experiment_utils.c -lm -O3

clean:
	rm -f mnist_optimiser.out
	rm -f *.csv
	rm -rf plots
