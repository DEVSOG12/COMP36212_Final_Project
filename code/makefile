CC = gcc
CFLAGS = -O3 -lm

all: mnist_optimiser.out

clean:
	rm -f mnist_optimiser.out
	rm -f *.csv
	rm -rf plots

mnist_optimiser.out: main.c mnist_helper.c neural_network.c optimiser.c experiment_utils.c
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: all clean
