//
// Created by mike on 19-5-13.
//

#pragma once

#include <sstream>
#include <string>
#include <utility>

template<typename ...Args>
std::string serialize(Args &&...args) {
    std::stringstream ss;
    (ss << ... << std::forward<Args>(args));
    return ss.str();
}
