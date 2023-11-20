#pragma once
#include "core/type.h"
#include <vector>

namespace tilegraph {

class Tile {
 public:
  // Self information
  std::vector<int64_t> tile_dimension;
  std::vector<int64_t> tile_stride;
  std::vector<int64_t> tile_position;
  std::string tile_name;
  int64_t start_offset;

 public:
  // Constructor
  Tile() = delete;
  Tile(const std::vector<int64_t>& dimension,
       const std::vector<int64_t>& position, const std::string& name,
       const int64_t& offset);
  Tile(const std::vector<int64_t>& dimension,
       const std::vector<int64_t>& position, const std::vector<int64_t>& stride,
       const std::string& name, const int64_t& offset);
  // Destructor
  ~Tile() = default;
  // Information
  void printInformation();
  void printSummary();
};

class TileTensor {
  /* Tile Tensor is a class that contains tensor tiles after split
   *  tiles: vector of Tile
   *  stride: stride of tiles in TileTensor
   *  shape: num_tiles = IIshape
   */
 public:
  // all tiles
  std::vector<Tile> tiles;
  std::vector<Tile> remain_tiles;
  std::vector<int64_t> stride;
  std::vector<int64_t> shape;
  TensorLayout layout;
  TensorType type;
  std::string name;

 public:
  // Constructor
  TileTensor() = default;
  TileTensor(const std::vector<int64_t>& shape,
             const std::vector<int64_t>& stride, TensorType type,
             TensorLayout layout, std::string name);
  ~TileTensor() = default;
  // Add tile
  void addTile(const Tile& t);
  // Get tiles
  std::vector<Tile> getTiles();
  // Delete tile
  // FIXME: delete tile may cause the missing of finding
  Tile deleteTile(const std::vector<int64_t>& coord);
  // Get tile by multi-dimension coord
  Tile operator()(const std::vector<int64_t>& coord);
  // Clear tiles
  void clear();
  // Tiles are empty or not
  bool empty();
  // num tiles
  int64_t numTiles();
  // num neat tiles
  int64_t numNeatTiles();
  // num unneat tiles
  int64_t numRemainTiles();

  bool isNeat();
  // TileTensor coord under the range is neat
  std::vector<int64_t> neatRange();

  // print
  void printInformation();
  void printSummary();
};

}  // namespace tilegraph