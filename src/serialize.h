//
// Created by mike on 19-5-13.
//

#ifndef LEARNOPENGL_SERIALIZE_H
#define LEARNOPENGL_SERIALIZE_H

#include <sstream>
#include <string>
#include <utility>

template<typename ...Args>
std::string serialize(Args &&...args) {
    std::stringstream ss;
    (ss << ... << std::forward<Args>(args));
    return ss.str();
}

#endif //LEARNOPENGL_SERIALIZE_H
