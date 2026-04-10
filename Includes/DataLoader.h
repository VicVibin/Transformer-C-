#pragma once
#include <vector>
#include <iostream>
#include <map>
#include <unordered_set>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#ifdef _WIN32
    #include <windows.h>
#else
    #include <dirent.h>
    #include <sys/stat.h>
#endif

using str = std::string;
using Text = std::vector<std::string>;

class TextProcessor {
public:
    str toLower(const str& string);                   
    str removePunctuation(const str& string);        
    bool isAlpha(const str& word);                 
    Text tokenize(const str& line);                    
    Text getFilesInDirectory(const str& folderPath);  
    Text readAllStories(const str& folderPath);      
    Text cleanText(const Text& lines);                

};

Text LoadStory(const str& path);
void Reading(Text string);
Text read_words(Text words, const int start, const int end);
