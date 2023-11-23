#include <vector>

namespace tilegraph::graph {
    // class GraphBase {
    //    private:
    //     static int64_t count;

    //    public:
    //     std::string name;
    //     const int64_t index;
    //     std::vector<Node *> operators;
    //     std::vector<Data *> inputs;
    //     std::vector<Data *> outputs;
    //     std::vector<Data *> temps;
    //     std::unordered_set<Data *> remaining_data;
    //     // Device
    //     Platform platform;
    //     //   std::vector<Task *> task_list;

    //    public:
    //     Graph(std::vector<Node *> operators_list = {},
    //           std::vector<Data *> inputs_list = {},
    //           std::vector<Data *> outputs_list = {},
    //           std::string name_value = "");
    //     ~Graph() = default;
    //     std::vector<Node *> topoSort();
    //     //   virtual std::string generatorHead(int64_t indent = 0) = 0;
    //     //   virtual std::string generatorTask(int64_t indent = 0) = 0;
    //     //   virtual std::string generatorHost(int64_t indent = 0) = 0;
    //     //   virtual std::string generatorCode(int64_t indent = 0) = 0;
    //     //   virtual std::string generatorHeadFile(int64_t indent = 0) = 0;
    //     //   virtual std::string generatorSourceFile(int64_t indent = 0) = 0;
    //     //   virtual void applyPlatform(Platform platform) = 0;
    //     void printGraph();
    //     bool removeNode(int64_t index);
    //     void addNode(Node *node);
    // };
}  // namespace tilegraph