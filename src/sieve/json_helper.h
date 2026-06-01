// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.


#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iostream>

// ---------------- JSON String Builder ----------------
class JSONString {
public:
    enum class Type { Object, Array };

    JSONString(Type t) : type(t) {
        if(type == Type::Object){
            content = "{}";
        }
        if(type == Type::Array){
            content = "[]";
        }
    }

    // Add a named element to an object
    void addNamedData(const std::string& name, const std::string& value) {
        if (type != Type::Object) return;
        if (content != "{}")
        {
            std::string newEntry = "," + ("\""+name)+"\":"+value; //The parentheses avoid char additions
            content.insert(content.size()-1, newEntry);
        }
        else{
            std::string newEntry = "\""+name+"\":"+value;
            content.insert(content.size()-1, newEntry);
        }
    }

    // Append an element to an array
    void appendData(const std::string& value) {
        if (type != Type::Array) return;
        if (content != "[]")
        {
            std::string newEntry = "," + value;
            content.insert(content.size()-1, newEntry);
        }
        else{
            std::string newEntry = value;
            content.insert(content.size()-1, newEntry);
        }
    }

    // Get the full JSON string
    std::string str() const {
        return content;
    }

private:
    Type type;
    std::string content;
};

// ---------------- JSON File Controller ----------------
class JSON_IO {
public:
    JSON_IO() {}

    // Append a JSONString (object or array) to a top-level JSON array in a file
    void appendToFile(const std::string& filename, const JSONString& json) {
    ensureFile(filename);

    std::fstream file(filename, std::ios::in | std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return;
    }

    // Seek to last two characters to check if array is empty
    file.seekg(-4, std::ios::end);
    char ch1 = file.get(); // [
    bool arrayEmpty = (ch1 == '[');
    // Move write pointer just before final ']'
    file.seekp(-2, std::ios::end);

    if (!arrayEmpty) file << ",\n"; // only add comma if array already has elements

    // Write the new JSON element
    file << json.str() << "\n]";

    file.close();
}

private:
    std::vector<std::string> files; // track files already initialized

    void ensureFile(const std::string& filename) {
        if (std::find(files.begin(), files.end(), filename) == files.end()) {
            files.push_back(filename);
            std::ofstream out(filename);
            if (!out.is_open()) std::cerr << "Cannot create file: " << filename << "\n";
            else out << "[\n\n]"; // initialize empty top-level array
        }
    }
};
