#include <iostream>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

struct GrepOptions {
    bool ignore_case = false;
    bool use_regex = false;
    bool line_number = false;
    bool with_filename = false;
    bool no_filename = false;
    bool count = false;
    bool invert_match = false;
};

int main(int argc, char* argv[]) {
    // Simple argument parsing (placeholder). Real implementation omitted for brevity.
    GrepOptions opts;
    std::string pattern;
    std::vector<std::string> files;
    // TODO: parse args to fill opts, pattern, and files.
    // For now, just demonstrate reading from stdin.
    std::istream* in = &std::cin;
    std::string line;
    size_t line_no = 0;
    while (std::getline(*in, line)) {
        ++line_no;
        // Placeholder match: always true.
        bool match = true;
        if (opts.invert_match) match = !match;
        if (match) {
            if (opts.count) continue; // count handling later
            if (opts.line_number) std::cout << line_no << ":";
            std::cout << line << '\n';
        }
    }
    return 0;
}
