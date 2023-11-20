#pragma once
#include "core/type.h"
#include <set>
#include <unordered_set>
#include <vector>

namespace tilegraph {

struct CacheData {
  // To identify a data tile in cache
 public:
  // Tensor name or identifier
  std::string name;
  // Offset in tensor for this data tile
  int64_t offset;
  // Size of data tile(in bytes)
  int64_t size;

 public:
  // Constructor
  CacheData();
  CacheData(std::string _name, int64_t _offset, int64_t _size);
  // Destructor
  ~CacheData() = default;
  // Operator overload
  bool operator==(const CacheData &other) const;
  // Empty
  bool isEmpty();
  // Information
  void printInformation();
};

struct CacheDataHash {
  // Hash Function for CacheData
  size_t operator()(const CacheData &data) const;
};

struct Block {
 public:
  // Flag
  bool allocated;
  // Block start address in nram (aligned)
  int64_t block_offset;
  // Block size
  int64_t block_size;
  // Ptr to next block
  Block *next;
  // Ptr to previous block
  Block *prev;
  // Name/identifier of cache
  std::string cache_name;
  // Type: cache or ldram
  CacheType cache_type;
  // Data cached in block
  CacheData data;
  // Count for cache replacement
  int data_count;

 public:
  // Constructor
  Block() = delete;
  Block(bool _allocated, int64_t _block_offset, int64_t _block_size,
        Block *_next, Block *_prev, std::string _cache_name,
        CacheType _cache_type, CacheData _data, int _data_count);
  // Destructor
  ~Block() = default;
  // Operator overload
  bool operator==(const Block &block) const;
  // Information
  void printInformation(int indent);
};

struct CompareBlockSize {
  // Comparison Function for block size
  bool operator()(const Block *block1, const Block *block2) const;
};

struct CacheHit {
  // Result for cache queries. Five situations in total:
  //
  // 1. Location == CACHE, which means target data is found in nram cache
  //    - cache_offset: start address of target data in cache
  //    - other variables are null
  // 2. Location == LDRAM, which means target data is missing in nram cache
  //                       but found in ldram
  //    - ldram_from_offset: start address in ldram where target data is
  //                         stored
  //    - cache_offset: alloc a space in cache for target data to load to
  //    2.1. cache_offset is empty
  //        - other variables are null
  //    2.2. cache_offset is occupied
  //        - ldram_to_offset: alloc a space in ldram for cache swapping
  //        - replaced_data_cache_offset: replaced data start address in cache
  //        - replaced_data_size: size of replaced data from cache
  // 3. Location == NOT_FOUND, which means target data is not found in either
  //                nram cache or ldram
  //    - ldram_from_offset: set to null
  //    - cache_offset: alloc a space in cache for target data to load to
  //    3.1. cache_offset is empty
  //        - other variables are null
  //    3.2. cache_offset is occupied
  //        - ldram_to_offset: alloc a space in ldram for cache swapping
  //        - replaced_data_cache_offset: replaced data start address in cache
  //        - replaced_data_size: size of replaced data from cache

 public:
  CacheHitLocation location;
  int64_t cache_offset;
  int64_t ldram_from_offset;
  std::vector<int64_t> ldram_to_offset;
  std::vector<int64_t> replaced_data_cache_offset;
  std::vector<int64_t> replaced_data_size;

 public:
  // Constructor
  CacheHit() = delete;
  CacheHit(CacheHitLocation _location, int64_t _cache_offset,
           int64_t _ldram_from_offset, std::vector<int64_t> _ldram_to_offset,
           std::vector<int64_t> _replaced_data_cache_offset,
           std::vector<int64_t> _replaced_data_size);
  // Destructor
  ~CacheHit() = default;
  // Information
  void printInformation();
};

class Cache {
 public:
  // Cache configurations
  std::string name;
  // Cache Size (in Bytes)
  int64_t cache_size;
  int64_t cache_align_size;
  // LDRAM Size (in Bytes)
  int64_t ldram_size;
  MemoryDispatch cache_dispatch;

 private:
  // Cache block list
  Block *cache_head;
  Block *cache_tail;
  Block *ldram_head;
  Block *ldram_tail;

  // Free block List
  std::set<Block *, CompareBlockSize> free_cache_blocks;
  std::set<Block *, CompareBlockSize> free_ldram_blocks;
  // Hashmap used to check if data is in ldram
  std::unordered_set<CacheData, CacheDataHash> storedInLdram;
  // Hashmap used to check if data is locked
  std::unordered_set<CacheData, CacheDataHash> lockedData;
  // Flag that indicates lock all following data
  bool lock_on;
  // Clock count for FIFO
  int64_t clock;

 public:
  // Constructor
  Cache() = delete;
  Cache(int64_t total_nram, int64_t total_ldram, int64_t align_size,
        std::string name, MemoryDispatch dispatch);
  // Destructor
  ~Cache();
  // Clear cache/ldram
  void reset();
  // Reset cache dispatch algorithm
  void resetDispatch(MemoryDispatch dispatch);

  // Cache Primitives
  // Load data to cache
  CacheHit load(CacheData data);
  // Allocate memory for data
  CacheHit allocate(CacheData data);
  // Free data from cache
  CacheHit free(CacheData data);
  // Find data in cache (read-only)
  CacheHit find(CacheData data);
  // Lock & unlock data
  void lock();
  void unlock();

  // Information
  void printInformation();
  void printBlocks(Block *head);
  void printMemoryGraph(Block *head, int height, int width);

 private:
  void initBlockCount(Block *block);
  void updateBlockCount(Block *block, bool match);
  bool cacheReplaceable(Block *curr, Block *target);
  void freeBlock(Block *target);
  void safeEraseFreeBlock(Block *block);
  void safeInsertFreeBlock(Block *block);
  Block *cacheAlloc(CacheData target_data, int indent);
  std::vector<std::tuple<CacheData, int64_t>> loadData2Block(
      CacheData replacer_data, Block *replacee);
  CacheHit loadData(CacheData data, bool alloc);
  CacheHit findData(CacheData data, bool free);
};

}  // namespace tilegraph