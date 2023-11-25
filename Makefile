CC := g++
EXAMPLE := subgraph_match

build:
	@mkdir build
	@cd build && cmake .. && make -j8

test: build
	@cd build && make test

example: build
	@$(CC) examples/$(EXAMPLE).cpp -Iinclude -Lbuild/ -ltilegraph -Wl,-rpath=build/ -o build/$(EXAMPLE) 
	@./build/$(EXAMPLE)

clean:
	@rm -rf build