// This file is a part of nla3d project. For information about authors and
// licensing go to project's repository on github:
// https://github.com/dmitryikh/nla3d

#include <string>

#include "PostProcessor.h"

namespace nla3d {

PostProcessor::PostProcessor(FEStorage& st) : storage(st) {
    active = false;
    failed = false;
    nPost_proc = 0;
}

std::string PostProcessor::getStatus() const {
    std::stringstream ss;
    ss << "PostProcessor No " << nPost_proc << ": " << name << "\n\tActive - " << (active ? "true" : "false")
       << ", Failed - " << (failed ? "true" : "false");
    return ss.str();
}

} // namespace nla3d
