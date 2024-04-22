all: interpret interpret_batch

interpret:
	mkdir bin
	g++ -Wall -g -std=c++20 ./interpret.cpp ./evaluate_mltl.cpp ./utils.cpp -o ./bin/interpret

interpret_batch:
	g++ -Wall -g -std=c++20 ./interpret_batch.cpp ./evaluate_mltl.cpp ./utils.cpp -o ./bin/interpret_batch

clean:
	rm -rf ./bin

