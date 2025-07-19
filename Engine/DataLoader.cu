#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include "includes/DataLoader.h"
#ifdef _WIN32
    #include <windows.h>
#else
    #include <dirent.h>
    #include <sys/stat.h>
#endif

str TextProcessor::toLower(const str& string) 
{   

    str result = string;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

str TextProcessor::removePunctuation(const str& string) {
    str result;
    str punctuation = ",.\"'!@#$%^&*(){}?/;`~:<>+=-\\";
    for (char c : string) {
        if (punctuation.find(c) == str::npos) {
            result += c;
        }
    }
    return result;
}

bool TextProcessor::isAlpha(const str& word) {
    if (word.empty()) return false;
    return std::all_of(word.begin(), word.end(), ::isalpha);
}

text TextProcessor::tokenize(const str& line) {
    text tokens;
    std::istringstream iss(line);
    str word;
    while (iss >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

text TextProcessor::getFilesInDirectory(const str& folderPath) {
    text filenames;
    str searchPath = folderPath + "\\*";
    WIN32_FIND_DATAA findFileData;
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                filenames.push_back(folderPath + "\\" + findFileData.cFileName);
            }
        } while (FindNextFileA(hFind, &findFileData) != 0);
        FindClose(hFind);
    }
    return filenames;
}

text TextProcessor::readAllStories(const str& folderPath) {
    text allLines;
    text filenames = getFilesInDirectory(folderPath);
    if (filenames.empty()) {
        std::cerr << "No files found in directory: " << folderPath << std::endl;
        return allLines;
    }

    std::cout << "Found " << filenames.size() << " files in directory" << std::endl;

    for (const str& filename : filenames) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open file " << filename << std::endl;
            continue;
        }

        str line;
        while (std::getline(file, line)) {
            if (line == "----------") break;
            if (!line.empty()) allLines.push_back(line);
        }

        file.close();
    }

    return allLines;
}

text TextProcessor::cleanText(const text& lines) {
    text cleanedWords;
    for (const str& line : lines) {
        str lowerLine = toLower(line);
        str cleanLine = removePunctuation(lowerLine);
        text tokens = tokenize(cleanLine);
        for (const str& word : tokens) {
            if (isAlpha(word)) {
                cleanedWords.push_back(word);
            }
        }
    }
    return cleanedWords;
}

