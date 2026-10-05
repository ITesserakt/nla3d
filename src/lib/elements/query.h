// This file is a part of nla3d project. For information about authors and
// licensing go to project's repository on github:
// https://github.com/dmitryikh/nla3d

#pragma once

namespace nla3d {

enum class scalarQuery { UNDEF = 0, SP, W, WU, WP, VOL, LAST };

constexpr const char* scalarQueryLabels[] = {"UNDEFINED", "S_P", "W", "WU", "WP", "VOL", "LAST"};

#ifndef SWIG // got swig 3.0.12 sytax error
static_assert(static_cast<int>(scalarQuery::LAST) == sizeof(scalarQueryLabels) / sizeof(scalarQueryLabels[0]) - 1,
              "scalarQuery enumeration and scalarQueryLabels must have the same number of entries");
#endif

enum class vectorQuery { UNDEF, IC, FLUX, GRADT, LAST };

constexpr const char* vectorQueryLabels[] = {"UNDEFINEDS", "IC", "FLUX", "GRADT", "LAST"};

#ifndef SWIG // got swig 3.0.12 sytax error
static_assert(static_cast<int>(vectorQuery::LAST) == sizeof(vectorQueryLabels) / sizeof(vectorQueryLabels[0]) - 1,
              "vectorQuery enumeration and vectorQueryLabels must have the same number of entries");
#endif

enum class tensorQuery {
    UNDEF,
    // usual stress tensor
    COUCHY,
    // second Piola-Kirchgoff stress tensor (symmetric 3x3)
    PK2,
    // Lagrange deformations
    E,
    // C = F^T F
    C,
    TSTRAIN,
    LAST
};

constexpr const char* tensorQueryLabels[] = {"UNDEFINED", "COUCHY", "PK2", "E", "C", "TSTRAIN", "LAST"};

#ifndef SWIG // got swig 3.0.12 sytax error
static_assert(static_cast<int>(tensorQuery::LAST) == sizeof(tensorQueryLabels) / sizeof(tensorQueryLabels[0]) - 1,
              "tensorQuery enumeration and tensorQueryLabels must have the same number of entries");
#endif

// means averaged values of the element
constexpr uint16 GP_MEAN = 100;

inline char const* query2label(scalarQuery query) {
    assert(query >= scalarQuery::UNDEF && query < scalarQuery::LAST);
    return scalarQueryLabels[static_cast<int>(query)];
}

inline char const* query2label(vectorQuery query) {
    assert(query >= vectorQuery::UNDEF && query < vectorQuery::LAST);
    return vectorQueryLabels[static_cast<int>(query)];
}

inline char const* query2label(tensorQuery query) {
    assert(query >= tensorQuery::UNDEF && query < tensorQuery::LAST);
    return tensorQueryLabels[static_cast<int>(query)];
}

} // namespace nla3d
