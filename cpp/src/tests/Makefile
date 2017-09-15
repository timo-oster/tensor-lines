CC = g++
CFLAGS = --std=c++11 -g
INCS = -I$$CXXTEST -I../ -I. -I/usr/include/eigen3
INCLUDES = $(wildcard ../*.hh) $(wildcard ../*.tcc)

TEST_SRC = $(wildcard *Test.hh)
SRC = ParallelEigenvectors.cc
OBJ = $(SRC:.cc=.o)

.PHONY: all
all: runner

%.o: ../%.cc $(INCLUDES)
	$(CC) $(CFLAGS) $(INCS) -c $< -o $@

runner: runner.cpp $(OBJ)
	$(CC) $(CFLAGS) $(INCS) $^ -o $@

runner.cpp: $(TEST_SRC)
	cxxtestgen --have-std --have-eh --runner=ErrorPrinter $^ -o $@

.PHONY: test
test: runner
	./runner -v

.PHONY: clean
clean:
	rm -f *.o runner.cpp runner parsetab.py parsetab.pyc