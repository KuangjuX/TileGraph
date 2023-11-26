CC := g++
EXAMPLE := subgraph_match

LD_FLAGS := -Lbuild/ -ltilegraph -Wl,-rpath=build/
INC_FLAGS := -Iinclude -I3rd-party/result -I3rd-party/fmt/include

build:
	@mkdir build
	@cd build && cmake .. && make -j8

test: build
	@cd build && make test

example: build
	@$(CC) examples/$(EXAMPLE).cpp $(INC_FLAGS) $(LD_FLAGS) -o build/$(EXAMPLE) 
	@./build/$(EXAMPLE)

clean:
	@rm -rf build