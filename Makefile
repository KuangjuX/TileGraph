CC := g++
EXAMPLE := persistent_kernel_fusion
EXAMPLE_SRCS := $(wildcard examples/*.cpp)
EXAMPLES := $(patsubst examples/%.cpp, %, $(EXAMPLE_SRCS))

LD_FLAGS := -Lbuild/ -ltilegraph -Wl,-rpath=build/
INC_FLAGS := -Iinclude -I3rd-party/result -I3rd-party/fmt/include -I3rd-party/fmtlog
MACRO_FLAGS := -DFMTLOG_HEADER_ONLY -DFMT_HEADER_ONLY

build:
	@mkdir build
	@cd build && cmake .. && make -j8

test: build
	@cd build && make test

example: build
	@$(CC) examples/$(EXAMPLE).cpp $(INC_FLAGS) $(LD_FLAGS) $(MACRO_FLAGS) -o build/$(EXAMPLE) 
	@./build/$(EXAMPLE)

examples: build 
	@for example in $(EXAMPLES); do \
		$(CC) examples/$$example.cpp $(INC_FLAGS) $(LD_FLAGS) $(MACRO_FLAGS) -o build/$$example; \
		echo "Running example: $$example"; \
		./build/$$example; \
	done


clean:
	@rm -rf build