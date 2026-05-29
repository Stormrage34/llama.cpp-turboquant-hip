#include "speculative.h"
#include <iostream>

int main() {
    auto type = common_speculative_type_from_name("draft-mtp");
    if (type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
        std::cerr << "FAIL: expected COMMON_SPECULATIVE_TYPE_DRAFT_MTP, got " << type << std::endl;
        return 1;
    }
    std::cout << "PASS" << std::endl;
    return 0;
}
