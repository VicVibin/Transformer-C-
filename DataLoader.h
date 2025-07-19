#pragma once
#include <vector>
#include <iostream>
#include <map>
#include <unordered_set>
#include <string>

using str = std::string;
using text = std::vector<std::string>;

class TextProcessor {
public:
    str toLower(const str& string);                    // DECLARATION only
    str removePunctuation(const str& string);          // DECLARATION only
    bool isAlpha(const str& word);                     // DECLARATION only
    text tokenize(const str& line);                    // DECLARATION only
    text getFilesInDirectory(const str& folderPath);   // DECLARATION only
    text readAllStories(const str& folderPath);        // DECLARATION only
    text cleanText(const text& lines);                 // DECLARATION only

};

