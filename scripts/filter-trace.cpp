#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <map>
#include <sstream>
#include <string.h>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <iostream>
// (taeklim)
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <getopt.h>


#define PAGE_SIZE_BITS 0xFFF
#define CACHE_LINE_BITS 0x1F

int main (int argc, char** argv) {
  std::ofstream file;
  std::string filename;
  std::string dmesg_file, trace_file;
  std::string output;
  std::string vertex_file, edge_file;
  int c = 0;
  int arg_num = 0;

  while ((c = getopt(argc, argv, "f:h")) != -1) {
    switch (c) {
      case 'f':
        filename = optarg;
        arg_num++;
        break;
      case 'h':
        printf("Converting edgelist to CSR\n");
        printf("\t-f | input fuile name\n");
        printf("\t-h | help message\n");
        return 0;
      case '?':
        break;
      default:
        break;
    }
  }
  if (arg_num != 1) {
    printf("Read dmesg and nvbit trace for filtering\n");
    printf("\t-f | input fuile name\n");
    printf("\t-h | help message\n");
    return 0;
  }
  dmesg_file = filename + "_dmesg.log";
  printf("dmesg filename %s\n", dmesg_file.c_str());
  
  std::ifstream dmesgFile(dmesg_file);
  if (!dmesgFile.is_open()) {
      fprintf(stderr, "Failed file opening\n");
      return 1;
  }

  printf("Start post processing\n");
  std::string line;
  std::vector<uint64_t> page_addr_list;
  while (getline(dmesgFile, line)) {
      std::vector<std::string> rows;
      std::stringstream ss(line);
      std::string field;

      while (getline(ss, field, ' ')) {
          rows.push_back(field);
      }
      if (rows.size() == 3) {
          if (strcmp(rows[1].c_str(), "vaddr2,") == 0) {
              page_addr_list.push_back(std::strtoull(rows[2].c_str(), NULL, 0));
          }
      }
  }
  printf("Done reading dmesg file page_addr_list size:%ld\n", 
      page_addr_list.size());

  trace_file = filename + "_trace.log";
  printf("trace filename %s\n", trace_file.c_str());
  
  std::ifstream traceFile(trace_file);
  if (!traceFile.is_open()) {
      fprintf(stderr, "Failed file opening\n");
      return 1;
  }

  std::string trace_line;
  std::unordered_set<uint64_t> cl_accessed;
  std::unordered_set<uint64_t> cl_transfer;
  uint64_t cl_addr;
  while (getline(traceFile, trace_line)) {
      std::istringstream iss(trace_line);
      while (iss >> cl_addr) {
        cl_accessed.insert(cl_addr);
      }
  }
  printf("Done reading trace file cl_accessed size:%ld\n", 
      cl_accessed.size());

  std::unordered_set<uint64_t>::iterator itr;
  uint64_t masked_page_addr = 0;
  uint64_t prev_masked_addr = 0;
  uint64_t cl_access_num = 0;
  bool access_flag = false;
  for (itr = cl_accessed.begin(); itr != cl_accessed.end(); itr++) {
    masked_page_addr = (*itr & ~PAGE_SIZE_BITS);
    if (masked_page_addr != prev_masked_addr) {
      auto find_it = 
        std::find(page_addr_list.begin(), page_addr_list.end(), masked_page_addr);
      if (find_it != page_addr_list.end()) {
        cl_access_num++;
        access_flag = true;
      } else {
        access_flag = false;
      }
      prev_masked_addr = masked_page_addr;
    } else if (access_flag) {
      cl_access_num++;
    }
  }
  printf("cache-line accessed from transfer: %ld\n", cl_access_num);

  return 0;
}
