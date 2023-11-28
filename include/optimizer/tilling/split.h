#include <vector>

namespace tilegraph::tilling {
    class Split {
       public:
        std::vector<std::size_t> split_dims;

        Split(std::vector<std::size_t> split_dims);
        ~Split() = default;
    };
}  // namespace tilegraph::tilling