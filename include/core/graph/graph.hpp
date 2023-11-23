#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>

#include "core/type.hpp"
#include "core/platform.hpp"
#include "core/graph/subgraph.hpp"

namespace tilegraph::graph {

    class Data;
    class SubGraph;

    class Node {
       private:
        static int64_t count;

       public:
        std::string name;
        const int64_t index;
        int64_t indegree;
        int64_t outputs_num;
        std::vector<Data *> inputs;
        std::vector<Data *> outputs;
        std::vector<Node *> predecessors;
        std::vector<Node *> successors;
        //   std::unordered_map<std::string, Attribute> attributes;
        OperatorType operator_type;
        std::shared_ptr<SubGraph> subgraph;

       public:
        Node(std::vector<Data *> inputs_list = {},
             std::vector<Data *> outputs_list = {}, std::string name_value = "",
             int64_t outputs_num_value = 1);
        Node(std::vector<Data *> inputs_list = {},
             std::vector<Data *> outputs_list = {},
             std::shared_ptr<SubGraph> = nullptr, std::string name_value = "",
             int64_t outputs_num_value = 1);
        ~Node() = default;
        Data *getOutput(int64_t index);
        std::vector<Data *> getOutputs();
        //   void printNode();
        //   void printLink();
        //   void setAttribute(std::string key, Attribute attribute);
        //   Attribute getAttribute(std::string key);
        //   void deleteAttribute(std::string key);
        OperatorType getOperatorType();
    };

    class Data {
       private:
        static int64_t count;

       public:
        std::string name;
        const int64_t index;
        int remaining;
        Node *producer;
        std::vector<Node *> consumers;
        TensorDatatype tensor_datatype;
        TensorType tensor_type;
        //   TensorLayout tensor_layout;
        std::vector<int64_t> tensor_dimension;
        //   std::vector<int64_t> tensor_stride;
        //   int64_t data_offset;
        //   bool is_contiguous;

       public:
        Data() = delete;
        Data(const std::vector<int64_t> &dimension, std::string name_value = "",
             TensorDatatype dtype = TensorDatatype::FLOAT,
             TensorType type = TensorType::VARIABLE);
        //   Data(const std::vector<int64_t> &dimension,
        //        TensorDatatype dtype = TensorDatatype::FLOAT,
        //        TensorType type = TensorType::VARIABLE,
        //        TensorLayout layout = TensorLayout::ARRAY, int64_t offset = 0,
        //        std::string name_value = "");
        //   Data(const std::vector<int64_t> &dimension,
        //        const std::vector<int64_t> &stride,
        //        TensorDatatype dtype = TensorDatatype::FLOAT,
        //        TensorType type = TensorType::VARIABLE,
        //        TensorLayout layout = TensorLayout::ARRAY, int64_t offset = 0,
        //        std::string name_value = "");
        ~Data() = default;
        void setProducer(Node *producer_value);
        void addConsumer(Node *consumer_value);
        Data *clone(Data *);

        //   void printData();
        //   void printLink();
        //   bool isContiguous();
        //   void flatten(int64_t start = 0, int64_t end = -1);
        //   TileTensor tiling(const Split &split);
        //   TileTensor tiling(const std::vector<int64_t> &shape);
    };

    class Graph {
       private:
        static int64_t count;

       public:
        std::string name;
        const int64_t index;
        std::vector<Node *> operators;
        std::vector<Data *> inputs;
        std::vector<Data *> outputs;
        std::vector<Data *> temps;
        std::unordered_set<Data *> remaining_data;
        // Device
        Platform platform;
        //   std::vector<Task *> task_list;

       public:
        Graph(std::vector<Node *> operators_list = {},
              std::vector<Data *> inputs_list = {},
              std::vector<Data *> outputs_list = {},
              std::string name_value = "");
        ~Graph() = default;
        std::vector<Node *> topoSort();
        //   virtual std::string generatorHead(int64_t indent = 0) = 0;
        //   virtual std::string generatorTask(int64_t indent = 0) = 0;
        //   virtual std::string generatorHost(int64_t indent = 0) = 0;
        //   virtual std::string generatorCode(int64_t indent = 0) = 0;
        //   virtual std::string generatorHeadFile(int64_t indent = 0) = 0;
        //   virtual std::string generatorSourceFile(int64_t indent = 0) = 0;
        //   virtual void applyPlatform(Platform platform) = 0;
        void printGraph();
        bool fuseNode(std::vector<Node *> old_nodes, Node *subgraph_node);
    };

}  // namespace tilegraph::graph