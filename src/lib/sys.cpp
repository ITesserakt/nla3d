// This file is a part of nla3d project. For information about authors and
// licensing go to project's repository on github:
// https://github.com/dmitryikh/nla3d

#include "sys.h"

// obligatory macros to initialize easyloggingpp system
INITIALIZE_EASYLOGGINGPP

namespace nla3d {

static LogInitializer logInitializer;

uint32 tick() { return static_cast<uint32>(clock()); }

int32 npow(const int16 dig, const uint16 power) {
    int32 res = 1; // TODO: if too big number?
    for (uint16 i = 0; i < power; i++) {
        res *= dig;
    }
    return res;
}

static std::vector<std::string> read_tokens(const char* input) {
    std::vector<std::string> vec = {};
    std::string tmp = {};
    constexpr char delimeters[] = "(),";
    const char* p = input;
    const char* start = p;
    while (*p) {
        if (*p == '!') {
            break;
        }
        bool isfind = false;
        const char* delp = delimeters;
        while (*delp) {
            if (*delp == *p) {
                isfind = true;
                break;
            }
            delp++;
        }
        if (isfind) {
            tmp.assign(start, static_cast<int16>(p - start));
            del_spaces(tmp);
            vec.push_back(tmp);
            vec.emplace_back(delp, 1);
            p++;
            start = p;
            continue;
        }
        p++;
    }
    tmp.assign(start, static_cast<int16>(p - start));
    del_spaces(tmp);
    vec.push_back(tmp);
    return vec;
}

void del_spaces(std::string& str) {
    uint16 start = 0;
    if (str.empty())
        return;
    while (str[start] == ' ') {
        start++;
        if (start == str.length()) {
            str = " ";
            return;
        }
    }
    auto end = static_cast<uint16>(str.length() - 1);
    while (str[end] == ' ') {
        end--;
    }
    str = std::string(str, start, end - start + 1);
}

char* getCmdOption(char** begin, char** end, const std::string& option) {
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return nullptr;
}

std::vector<char*> getCmdManyOptions(char** begin, char** end, const std::string& option) {
    std::vector<char*> _vec;
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        while (!(itr[0][0] == '-' && (itr[0][1] < '0' || itr[0][1] > '9'))) {
            _vec.push_back(*itr);
            if (++itr == end) {
                break;
            }
        }
    }
    return _vec;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

// TODO: this functions only truncate file extension.
// But it was intended to leave only a file name (delete path and extension)
std::string getFileNameFromPath(const std::string& filename) {
    const auto pivot = std::find(filename.rbegin(), filename.rend(), '.');
    return pivot == filename.rend() ? filename : std::string(filename.begin(), pivot.base() - 1);
}

} // namespace nla3d
